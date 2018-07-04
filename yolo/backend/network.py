# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda
import numpy as np
import cv2
import os

from .utils.feature import create_feature_extractor


def create_yolo_network(architecture,
                        input_size,
                        nb_classes,
                        nb_box):
    feature_extractor = create_feature_extractor(architecture, input_size)
    yolo_net = YoloNetwork(feature_extractor,
                           input_size,
                           nb_classes,
                           nb_box)
    return yolo_net


class YoloNetwork(object):
    
    def __init__(self,
                 feature_extractor,
                 input_size,
                 nb_classes,
                 nb_box):
        
        # 1. create full network
        grid_size = feature_extractor.get_output_size()
        
        # make the object detection layer
        output_tensor = Conv2D(nb_box * (4 + 1 + nb_classes), (1,1), strides=(1,1),
                               padding='same', 
                               name='detection_layer_{}'.format(nb_box * (4 + 1 + nb_classes)), 
                               kernel_initializer='lecun_normal')(feature_extractor.feature_extractor.output)
        output_tensor = Reshape((grid_size, grid_size, nb_box, 4 + 1 + nb_classes))(output_tensor)
    
        model = Model(feature_extractor.feature_extractor.input, output_tensor)
        self._norm = feature_extractor.normalize
        self._model = model
        self._model.summary()
        self._init_layer()

    def _init_layer(self):
        layer = self._model.layers[-2]
        weights = layer.get_weights()
        
        input_depth = weights[0].shape[-2] # 2048
        new_kernel = np.random.normal(size=weights[0].shape)/ input_depth
        new_bias   = np.zeros_like(weights[1])

        layer.set_weights([new_kernel, new_bias])

    def load_weights(self, weight_path, by_name):
        self._model.load_weights(weight_path, by_name=by_name)
        
    def forward(self, image):
        def _get_input_size():
            input_shape = self._model.get_input_shape_at(0)
            _, h, w, _ = input_shape
            return h
            
        input_size = _get_input_size()
        image = cv2.resize(image, (input_size, input_size))
        image = self._norm(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)

        # (13,13,5,6)
        netout = self._model.predict(input_image)[0]
        return netout

    def get_model(self, first_trainable_layer=None):
        layer_names = [layer.name for layer in self._model.layers]
        fixed_layers = []
        if first_trainable_layer in layer_names:
            for layer in self._model.layers:
                if layer.name == first_trainable_layer:
                    break
                layer.trainable = False
                fixed_layers.append(layer.name)

        if fixed_layers != []:
            print("The following layers do not update weights!!!")
            print("    ", fixed_layers)
        return self._model

    def get_grid_size(self):
        _, h, w, _, _ = self._model.get_output_shape_at(-1)
        assert h == w
        return h

    def get_normalize_func(self):
        return self._norm



