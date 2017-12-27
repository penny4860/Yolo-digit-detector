# -*- coding: utf-8 -*-
# This module is responsible for communicating with the outside of the yolo package.
# Outside the package, someone can use yolo detector accessing with this module.

import os

from yolo.backend.decoder import YoloDecoder
from yolo.backend.network import YoloNetwork
from yolo.backend.loss import YoloLoss
from yolo.backend.batch_gen import create_batch_generator
from yolo.backend.utils.fit import train
from yolo.backend.utils.annotation import get_train_annotations


def create_yolo(architecture,
                labels,
                input_size = 416,
                max_box_per_image = 10,
                anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                weights_path=None):

    n_classes = len(labels)
    yolo_network = YoloNetwork(architecture, input_size, n_classes, max_box_per_image, anchors)
    yolo_loss = YoloLoss(yolo_network.get_true_boxes(),
                         yolo_network.get_grid_size(),
                         n_classes, anchors)
    yolo_decoder = YoloDecoder(anchors)
    yolo = YOLO(yolo_network, yolo_loss, yolo_decoder, labels, input_size, max_box_per_image)
    if weights_path:
        yolo.load_weights(weights_path)
    return yolo


class YOLO(object):
    def __init__(self,
                 yolo_network,
                 yolo_loss,
                 yolo_decoder,
                 labels,
                 input_size = 416,
                 max_box_per_image = 10):
        """
        # Args
            feature_extractor : BaseFeatureExtractor instance
        """
        self._yolo_network = yolo_network
        self._yolo_loss = yolo_loss
        self._yolo_decoder = yolo_decoder
        
        self._labels = labels
        # Batch를 생성할 때만 사용한다.
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

    def train(self,
              img_folder,
              ann_folder,
              nb_epoch,
              saved_weights_name,
              batch_size=8,
              jitter=True,
              learning_rate=1e-4, 
              train_times=1,
              valid_times=1,
              warmup_epochs=None,
              valid_img_folder="",
              valid_ann_folder=""):

        # 1. get annotations        
        train_annotations, valid_annotations = get_train_annotations(self._labels,
                                                                     img_folder,
                                                                     ann_folder,
                                                                     valid_img_folder,
                                                                     valid_ann_folder)
        
        # 1. get batch generator
        train_batch_generator = self._get_batch_generator(train_annotations, batch_size, jitter=jitter)
        valid_batch_generator = self._get_batch_generator(valid_annotations, batch_size, jitter=False)
        
        # 2. To train model get keras model instance & loss fucntion
        model = self._yolo_network.get_model()
        loss = self._get_loss_func(batch_size,
                                  warmup_epochs,
                                  train_times,
                                  valid_times)
        
        # 3. Run training loop
        train(model,
                loss,
                train_batch_generator,
                valid_batch_generator,
                learning_rate      = learning_rate, 
                nb_epoch           = nb_epoch,
                train_times        = train_times,
                valid_times        = valid_times,
                saved_weights_name = saved_weights_name)

    def _get_loss_func(self,
                      batch_size,
                      warmup_epochs,
                      train_times = 1,
                      valid_times = 1):
        warmup_bs  = warmup_epochs * (train_times*(batch_size+1) + valid_times*(batch_size+1))
        return self._yolo_loss.custom_loss(batch_size, warmup_bs)

    def _get_batch_generator(self, annotations, batch_size, jitter=True):
        """
        # Args
            annotations : Annotations instance
            batch_size : int
            jitter : bool
        
        # Returns
            batch_generator : BatchGenerator instance
        """
        batch_generator = create_batch_generator(annotations,
                                                 self._input_size,
                                                 self._yolo_network.get_grid_size(),
                                                 batch_size,
                                                 self._max_box_per_image,
                                                 self._yolo_loss.anchors,
                                                 jitter=jitter,
                                                 norm=self._yolo_network.get_normalize_func())
        return batch_generator

