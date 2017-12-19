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
                 loss,
                 trainer,
                 labels, 
                 anchors):
        """
        # Args
            feature_extractor : BaseFeatureExtractor instance
        """
        self._yolo_network = network
        self._yolo_decoder = YoloDecoder(anchors)
        self._yolo_loss = loss
        self._yolo_trainer = trainer
        
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.anchors  = anchors

        # print a summary of the whole model
        self._yolo_network.model.summary()

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

    def train(self,
              train_batch,
              valid_batch,
              train_times,
              valid_times,
              nb_epoch,
              warmup_epochs,
              learning_rate,
              saved_weights_name='best_weights.h5'):
        self._yolo_trainer.train(train_batch,
                                 valid_batch,
                                 train_times,
                                 valid_times,
                                 nb_epoch,
                                 warmup_epochs,
                                 learning_rate,
                                 saved_weights_name)
