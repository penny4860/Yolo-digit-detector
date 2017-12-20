# -*- coding: utf-8 -*-

import os
from yolo.decoder import YoloDecoder
from yolo.network import YoloNetwork
from yolo.loss import YoloLoss

# create_feature_extractor(architecture, input_size)
# client : predict.py
# client : train.py
class YOLO(object):
    def __init__(self,
                 architecture,
                 input_size = 416,
                 n_classes = 1,
                 max_box_per_image = 10,
                 anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]):
        """
        # Args
            feature_extractor : BaseFeatureExtractor instance
        """
        
        # 1. inference model
        self._yolo_network = YoloNetwork(architecture, input_size, n_classes, max_box_per_image, anchors)
        
        # 2. loss function
        self._yolo_loss = YoloLoss(self._yolo_network.get_true_boxes(),
                                   self._yolo_network.get_grid_size(),
                                   n_classes,
                                   anchors)

        # 3. decoding
        self._yolo_decoder = YoloDecoder(anchors)
        self._anchors = anchors

    def load_weights(self, weight_path):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights in", weight_path)
            self._yolo_network.load_weights(weight_path)
        else:
            print("Fail to load pre-trained weights. Make sure weight file path.")

    def predict(self, image):
        """
        # Args
            image : 3d-array
        
        # Returns
            boxes : list of BoundBox instance
        """
        netout = self._yolo_network.forward(image)
        boxes = self._yolo_decoder.run(netout)
        return boxes

    def get_model(self):
        return self._yolo_network.get_model()

    def get_grid_size(self):
        return self._yolo_network.get_grid_size()

    def get_nb_boxes(self):
        return int(len(self._anchors)/2)
        
    def get_normalize_func(self):
        return self._yolo_network.get_normalize_func()

    def get_loss_func(self):
        return self._yolo_loss.custom_loss
        
