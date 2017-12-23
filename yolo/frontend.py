# -*- coding: utf-8 -*-

import os
from yolo.decoder import YoloDecoder
from yolo.network import YoloNetwork
from yolo.loss import YoloLoss
from yolo.batch_gen import BatchGenerator

# create_feature_extractor(architecture, input_size)
# client : predict_driver.py
# client : train_driver.py
class YOLO(object):
    def __init__(self,
                 architecture,
                 labels,
                 input_size = 416,
                 max_box_per_image = 10,
                 anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]):
        """
        # Args
            feature_extractor : BaseFeatureExtractor instance
        """
        
        # 1. inference model
        n_classes = len(labels)
        self._yolo_network = YoloNetwork(architecture, input_size, n_classes, max_box_per_image, anchors)
        
        # 2. loss function
        self._yolo_loss = YoloLoss(self._yolo_network.get_true_boxes(),
                                   self._yolo_network.get_grid_size(),
                                   n_classes,
                                   anchors)

        # 3. decoding
        self._yolo_decoder = YoloDecoder(anchors)
        self._anchors = anchors
        self._labels = labels
        self._input_size = input_size
        self._max_box_per_image = max_box_per_image

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

    def get_loss_func(self,
                      batch_size,
                      warmup_epochs,
                      train_times = 1,
                      valid_times = 1):
        warmup_bs  = warmup_epochs * (train_times*(batch_size+1) + valid_times*(batch_size+1))
        return self._yolo_loss.custom_loss(batch_size, warmup_bs)

    def get_normalize_func(self):
        return self._yolo_network.get_normalize_func()

    def get_grid_size(self):
        return self._yolo_network.get_grid_size()
    
    def get_batch_generator(self, images, batch_size, jitter=True):
        config = GeneratorConfig(self._input_size,
                                 self.get_grid_size(),
                                 self._labels,
                                 batch_size,
                                 self._max_box_per_image,
                                 self._anchors)
        
        batch_generator = BatchGenerator(images,
                                         config,
                                         jitter=jitter,
                                         norm=self.get_normalize_func())
        return batch_generator
        
        
        
