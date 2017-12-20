# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda
import numpy as np

from yolo.backend import create_feature_extractor


nb_box = 5

class YoloNetwork(object):
    
    def __init__(self,
                 architecture,
                 input_size,
                 nb_classes,
                 max_box_per_image=10):
        
        # 1. create feature extractor
        self._feature_extractor = create_feature_extractor(architecture, input_size)
        
        # 2. create full network
        input_tensor = Input(shape=(input_size, input_size, 3))
        features = self._feature_extractor.extract(input_tensor)
        grid_size = self._feature_extractor.get_output_size()
        
        # truth tensor
        true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))
    
        # make the object detection layer
        output_tensor = Conv2D(nb_box * (4 + 1 + nb_classes), 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='conv_23', 
                        kernel_initializer='lecun_normal')(features)
        output_tensor = Reshape((grid_size, grid_size, nb_box, 4 + 1 + nb_classes))(output_tensor)
        output_tensor = Lambda(lambda args: args[0])([output_tensor, true_boxes])
    
        model = Model([input_tensor, true_boxes], output_tensor)

        self.model = model
        self.true_boxes = true_boxes
        self._init_layer(grid_size)

    def _init_layer(self, grid_size):
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(grid_size*grid_size)
        new_bias   = np.random.normal(size=weights[1].shape)/(grid_size*grid_size)

        layer.set_weights([new_kernel, new_bias])

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        
    def forward(self, image):
        import cv2
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self._feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        # (13,13,5,6)
        netout = self.model.predict([input_image, dummy_array])[0]
        return netout
    
    def get_input_size(self):
        image_shape, _ = self.model.get_input_shape_at(0)
        _, h, w, _ = image_shape
        assert h == w
        return h
    
    def get_max_box_per_image(self):
        _, true_boxes_shape = self.model.get_input_shape_at(0)
        _, _, _, _, max_box_per_image, _ = true_boxes_shape
        return max_box_per_image

    def get_grid_size(self):
        _, h, w, _, _ = self.model.get_output_shape_at(-1)
        assert h == w
        return h

    def get_nb_boxes(self):
        _, _, _, nb_boxes, _ = self.model.get_output_shape_at(-1)
        return nb_boxes



#     true_boxes : tensor
#     model








    