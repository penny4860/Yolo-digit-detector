# -*- coding: utf-8 -*-

from yolo.decoder import YoloDecoder
from yolo.loss import YoloLoss
from yolo.trainer import YoloTrainer
from yolo.batch_gen import GeneratorConfig

# create_feature_extractor(architecture, input_size)
# client : predict.py
# client : train.py
class YOLO(object):
    def __init__(self,
                 network,
                 anchors):
        """
        # Args
            feature_extractor : BaseFeatureExtractor instance
        """
        self._yolo_network = network
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
