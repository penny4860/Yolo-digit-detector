# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda
import numpy as np

from yolo.backend import create_feature_extractor

class YoloNetwork(object):
    
    def __init__(self, architecture, input_size, nb_box, nb_classes, max_box_per_image=10):
        self._feature_extractor = create_feature_extractor(architecture, input_size)
    
        def _init_layer(layer):
            weights = layer.get_weights()
    
            new_kernel = np.random.normal(size=weights[0].shape)/(grid_h*grid_w)
            new_bias   = np.random.normal(size=weights[1].shape)/(grid_h*grid_w)
    
            layer.set_weights([new_kernel, new_bias])
        
        # 1. create feature extractor
        feature_extractor = create_feature_extractor(architecture, input_size)
        
        # 2. create full network
        input_tensor = Input(shape=(input_size, input_size, 3))
        features = feature_extractor.extract(input_tensor)
        grid_h, grid_w = feature_extractor.get_output_shape()
        
        # truth tensor
        true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))
    
        # make the object detection layer
        output_tensor = Conv2D(nb_box * (4 + 1 + nb_classes), 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='conv_23', 
                        kernel_initializer='lecun_normal')(features)
        output_tensor = Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_classes))(output_tensor)
        output_tensor = Lambda(lambda args: args[0])([output_tensor, true_boxes])
    
        model = Model([input_tensor, true_boxes], output_tensor)
        _init_layer(model.layers[-4])
        self.model = model
    
    
    