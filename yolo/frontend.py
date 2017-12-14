import numpy as np
import os
import cv2

from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from yolo.preprocessing import BatchGenerator
from yolo.backend import create_feature_extractor
from yolo.decoder import YoloDecoder
from yolo.loss import YoloLoss


class YOLO(object):
    def __init__(self, architecture,
                       input_size, 
                       labels, 
                       max_box_per_image,
                       anchors):

        self.input_size = input_size
        
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box   = 5
        self.anchors  = anchors
        self.max_box_per_image = max_box_per_image

        # create feature extractor
        self.feature_extractor = create_feature_extractor(architecture, input_size)
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()        

        # truth tensor
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image , 4))

        self.model = self._create_network()
        
        # print a summary of the whole model
        self.model.summary()

    def _create_network(self):
        def _init_layer(layer):
            weights = layer.get_weights()
    
            new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
            new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)
    
            layer.set_weights([new_kernel, new_bias])
        
        # make the feature extractor layers
        input_tensor = Input(shape=(self.input_size, self.input_size, 3))
        features = self.feature_extractor.extract(input_tensor)      

        # make the object detection layer
        output_tensor = Conv2D(self.nb_box * (4 + 1 + self.nb_class), 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='conv_23', 
                        kernel_initializer='lecun_normal')(features)
        output_tensor = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output_tensor)
        output_tensor = Lambda(lambda args: args[0])([output_tensor, self.true_boxes])

        model = Model([input_tensor, self.true_boxes], output_tensor)
        _init_layer(model.layers[-4])
        return model

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def predict(self, image):
        """
        # Args
            image : 3d-array
        
        # Returns
            boxes : list of BoundBox instance
        """
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        # (13,13,5,6)
        netout = self.model.predict([input_image, dummy_array])[0]
        yolo_decoder = YoloDecoder(self.anchors)
        boxes = yolo_decoder.run(netout)
        return boxes

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    valid_times,    # the number of times to repeat the validation set, often used for small datasets
                    nb_epoch,       # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    saved_weights_name='best_weights.h5',
                    debug=False):     

        self.batch_size = batch_size
        self.debug = debug

        if warmup_epochs > 0: nb_epoch = warmup_epochs # if it's warmup stage, don't train more than warmup_epochs

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        warmup_bs  = warmup_epochs * (train_times*(len(train_imgs)/batch_size+1) + valid_times*(len(valid_imgs)/batch_size+1))
        yolo_loss = YoloLoss(self.grid_w,
                             self.grid_h,
                             self.batch_size,
                             self.anchors,
                             self.nb_box,
                             self.nb_class,
                             warmup_bs,
                             self.true_boxes)
        self.model.compile(loss=yolo_loss.custom_loss, optimizer=optimizer)

        ############################################
        # Make train and validation generators
        ############################################
        train_batch, valid_batch = self._create_batch_generator(train_imgs, valid_imgs)

        ############################################
        # Start the training process
        ############################################        
        self.model.fit_generator(generator        = train_batch, 
                                 steps_per_epoch  = len(train_batch) * train_times, 
                                 epochs           = nb_epoch, 
                                 verbose          = 1,
                                 validation_data  = valid_batch,
                                 validation_steps = len(valid_batch) * valid_times,
                                 callbacks        = self._create_callbacks(saved_weights_name), 
                                 workers          = 3,
                                 max_queue_size   = 8)

    def _create_batch_generator(self, train_imgs, valid_imgs):
        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,  
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }    

        train_batch = BatchGenerator(train_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize,
                                     jitter=False)
        valid_batch = BatchGenerator(valid_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize,
                                     jitter=False)
        return train_batch, valid_batch

    def _create_callbacks(self, saved_weights_name):
        # Make a few callbacks
        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        tb_counter  = len([log for log in os.listdir(os.path.expanduser('~/logs/')) if 'yolo' in log]) + 1
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/') + 'yolo' + '_' + str(tb_counter), 
                                  histogram_freq=0, 
                                  write_graph=True, 
                                  write_images=False)
        callbacks = [early_stop, checkpoint, tensorboard]
        return callbacks

