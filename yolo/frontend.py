# -*- coding: utf-8 -*-

from yolo.decoder import YoloDecoder
from yolo.network import YoloNetwork

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
        self._yolo_network = YoloNetwork(architecture, input_size, n_classes, max_box_per_image, anchors)
        self._yolo_decoder = YoloDecoder(anchors)

    def load_weights(self, weight_path):
        self._yolo_network.load_weights(weight_path)

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

    def get_grid_size(self):
        return self._yolo_network.get_grid_size()

    def get_nb_boxes(self):
        return self._yolo_network.get_nb_boxes()
        
    def get_model(self):
        return self._yolo_network.get_model()

    def get_normalize_func(self):
        return self._yolo_network.get_normalize_func()

    def get_loss_func(self):
        return self._yolo_network.get_loss_func()
        
