# -*- coding: utf-8 -*-
# This module is responsible for communicating with the outside of the yolo package.
# Outside the package, someone can use yolo detector accessing with this module.

import os
import numpy as np

from yolo.backend.decoder import YoloDecoder
from yolo.backend.loss import YoloLoss
from yolo.backend.network import create_yolo_network
from yolo.backend.batch_gen import create_batch_generator

from yolo.backend.utils.fit import train
from yolo.backend.utils.annotation import get_train_annotations, get_unique_labels
from yolo.backend.utils.box import to_minmax


def get_object_labels(ann_directory):
    files = os.listdir(ann_directory)
    files = [os.path.join(ann_directory, fname) for fname in files]
    return get_unique_labels(files)


def create_yolo(architecture,
                labels,
                input_size = 416,
                anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                coord_scale=1.0,
                class_scale=1.0,
                object_scale=5.0,
                no_object_scale=1.0):

    n_classes = len(labels)
    n_boxes = int(len(anchors)/2)
    yolo_network = create_yolo_network(architecture, input_size, n_classes, n_boxes)
    yolo_loss = YoloLoss(yolo_network.get_grid_size(),
                         n_classes,
                         anchors,
                         coord_scale,
                         class_scale,
                         object_scale,
                         no_object_scale)

    yolo_decoder = YoloDecoder(anchors)
    yolo = YOLO(yolo_network, yolo_loss, yolo_decoder, labels, input_size)
    return yolo


class YOLO(object):
    def __init__(self,
                 yolo_network,
                 yolo_loss,
                 yolo_decoder,
                 labels,
                 input_size = 416):
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

    def load_weights(self, weight_path, by_name=False):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights in", weight_path)
            self._yolo_network.load_weights(weight_path, by_name=by_name)
        else:
            print("Fail to load pre-trained weights. Make sure weight file path.")

    def predict(self, image, threshold=0.3):
        """
        # Args
            image : 3d-array (BGR ordered)
        
        # Returns
            boxes : array, shape of (N, 4)
            probs : array, shape of (N, nb_classes)
        """
        def _to_original_scale(boxes):
            height, width = image.shape[:2]
            minmax_boxes = to_minmax(boxes)
            minmax_boxes[:,0] *= width
            minmax_boxes[:,2] *= width
            minmax_boxes[:,1] *= height
            minmax_boxes[:,3] *= height
            return minmax_boxes.astype(np.int)

        netout = self._yolo_network.forward(image)
        boxes, probs = self._yolo_decoder.run(netout, threshold)
        
        if len(boxes) > 0:
            boxes = _to_original_scale(boxes)
            return boxes, probs
        else:
            return [], []

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
              valid_img_folder="",
              valid_ann_folder="",
              first_trainable_layer=None,
              is_only_detect=False):

        # 1. get annotations        
        train_annotations, valid_annotations = get_train_annotations(self._labels,
                                                                     img_folder,
                                                                     ann_folder,
                                                                     valid_img_folder,
                                                                     valid_ann_folder,
                                                                     is_only_detect)
        
        # 1. get batch generator
        train_batch_generator = self._get_batch_generator(train_annotations, batch_size, train_times, jitter=jitter)
        valid_batch_generator = self._get_batch_generator(valid_annotations, batch_size, valid_times, jitter=False)
        
        # 2. To train model get keras model instance & loss fucntion
        model = self._yolo_network.get_model(first_trainable_layer)
        loss = self._get_loss_func(batch_size)
        
        # 3. Run training loop
        train(model,
                loss,
                train_batch_generator,
                valid_batch_generator,
                learning_rate      = learning_rate, 
                nb_epoch           = nb_epoch,
                saved_weights_name = saved_weights_name)

    def _get_loss_func(self, batch_size):
        return self._yolo_loss.custom_loss(batch_size)

    def _get_batch_generator(self, annotations, batch_size, repeat_times=1, jitter=True):
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
                                                 self._yolo_loss.anchors,
                                                 repeat_times,
                                                 jitter=jitter,
                                                 norm=self._yolo_network.get_normalize_func())
        return batch_generator
    
