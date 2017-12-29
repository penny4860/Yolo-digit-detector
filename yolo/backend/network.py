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
                        max_box_per_image,
                        nb_box,
                        feature_weights=None):
    if feature_weights is not None and os.path.exists(feature_weights):
        print("The network is initialized pretrained feature weights in {}".format(feature_weights))
        weights = feature_weights
    else:
        print("The network is initialized random weights")
        weights = None

    feature_extractor = create_feature_extractor(architecture, input_size, weights)
    yolo_net = YoloNetwork(feature_extractor,
                           input_size,
                           nb_classes,
                           max_box_per_image,
                           nb_box)
    return yolo_net


class YoloNetwork(object):
    
    def __init__(self,
                 feature_extractor,
                 input_size,
                 nb_classes,
                 max_box_per_image,
                 nb_box):
        
        # 1. create full network
        input_tensor = Input(shape=(input_size, input_size, 3))
        features = feature_extractor.extract(input_tensor)
        grid_size = feature_extractor.get_output_size()
        
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

        self._norm = feature_extractor.normalize
        self._model = model
        self._true_boxes = true_boxes
        self._model.summary()
        self._init_layer(grid_size)

    def _init_layer(self, grid_size):
        layer = self._model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(grid_size*grid_size)
        new_bias   = np.random.normal(size=weights[1].shape)/(grid_size*grid_size)

        layer.set_weights([new_kernel, new_bias])

    def load_weights(self, weight_path):
        self._model.load_weights(weight_path)
        
    def forward(self, image):
        def _get_dummy_true_boxes():
            _, true_boxes_shape = self._model.get_input_shape_at(0)
            array = np.zeros(true_boxes_shape[1:])
            return np.expand_dims(array, 0)
        
        def _get_input_size():
            input_shape, _ = self._model.get_input_shape_at(0)
            _, h, w, _ = input_shape
            return h
            
        input_size = _get_input_size()
        image = cv2.resize(image, (input_size, input_size))
        image = self._norm(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = _get_dummy_true_boxes()

        # (13,13,5,6)
        netout = self._model.predict([input_image, dummy_array])[0]
        return netout

    def get_model(self):
        return self._model

    def get_grid_size(self):
        _, h, w, _, _ = self._model.get_output_shape_at(-1)
        assert h == w
        return h

    def get_normalize_func(self):
        return self._norm

    def get_true_boxes(self):
        return self._true_boxes
