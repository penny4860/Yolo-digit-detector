# -*- coding: utf-8 -*-
import os

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from yolo.batch_gen import BatchGenerator


class YoloTrainer(object):
    
    def __init__(self, model, loss_func, normalize_func, generator_config):
        self.model = model
        self.loss_func = loss_func
        self.normalize_func = normalize_func
        self.generator_config = generator_config
        
    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    valid_times,    # the number of times to repeat the validation set, often used for small datasets
                    nb_epoch,       # number of epoches
                    learning_rate,  # the learning rate
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    saved_weights_name='best_weights.h5'):

        if warmup_epochs > 0: nb_epoch = warmup_epochs # if it's warmup stage, don't train more than warmup_epochs

        # 1. create optimizer
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        # 2. create loss function
        self.model.compile(loss=self.loss_func, optimizer=optimizer)

        # 3. batch generator
        train_batch, valid_batch = self._create_batch_generator(train_imgs, valid_imgs)

        # 4. training
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
        train_batch = BatchGenerator(train_imgs, 
                                     self.generator_config, 
                                     norm=self.normalize_func)
        valid_batch = BatchGenerator(valid_imgs, 
                                     self.generator_config, 
                                     norm=self.normalize_func,
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
