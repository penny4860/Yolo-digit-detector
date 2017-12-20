# -*- coding: utf-8 -*-
import json
import os
import numpy as np
from yolo.annotation import parse_annotation
from yolo import YOLO
from yolo.network import YoloNetwork
from yolo.loss import YoloLoss
from yolo.batch_gen import GeneratorConfig, BatchGenerator


def _parse(config):
    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'], 
                                                config['train']['train_image_folder'], 
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                    config['valid']['valid_image_folder'], 
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8*len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    
    overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

    print('Seen labels:\t', train_labels)
    print('Given labels:\t', config['model']['labels'])
    print('Overlap labels:\t', overlap_labels)    

    if len(overlap_labels) < len(config['model']['labels']):
        print('Some labels have no images! Please revise the list of labels in the config.json file!')
        return
    
    return train_imgs, valid_imgs

def train(conf):

    with open(conf) as config_buffer:
        config = json.loads(config_buffer.read())

    # 1. Construct the model 
    yolo = YOLO(config['model']['architecture'],
                config['model']['input_size'],
                len(config['model']['labels']),
                config['model']['max_box_per_image'],
                anchors = config['model']['anchors'])
    # 2. Load the pretrained weights (if any) 
    yolo.load_weights(config['train']['pretrained_weights'])

    # 3. Parse the annotations 
    train_imgs, valid_imgs = _parse(config)

    ###############################
    #   Start the training process 
    ###############################
    generator_config = GeneratorConfig(config['model']['input_size'],
                                       yolo.get_grid_size(),
                                       yolo.get_nb_boxes(),
                                       config['model']['labels'],
                                       config['train']['batch_size'],
                                       config['model']['max_box_per_image'],
                                       config['model']['anchors'])

    from yolo.trainer import YoloTrainer
    yolo_trainer = YoloTrainer(yolo.get_model(),
                               yolo.get_loss_func(),
                               yolo.get_normalize_func(),
                               generator_config)
    yolo_trainer.train(train_imgs,
                       valid_imgs,
                       train_times        = config['train']['train_times'],
                       valid_times        = config['valid']['valid_times'],
                       nb_epoch           = config['train']['nb_epoch'],
                       warmup_epochs      = config['train']['warmup_epochs'],
                       learning_rate      = config['train']['learning_rate'], 
                       saved_weights_name = config['train']['saved_weights_name'])
