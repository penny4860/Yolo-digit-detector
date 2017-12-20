# -*- coding: utf-8 -*-
import json
import os
import numpy as np
from yolo.annotation import parse_annotation
from yolo import YOLO
from yolo.network import YoloNetwork
from yolo.loss import YoloLoss
from yolo.batch_gen import GeneratorConfig, BatchGenerator

def train(conf):

    config_path = conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################

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
        
    ###############################
    #   Construct the model 
    ###############################

    yolo_network = YoloNetwork(config['model']['architecture'],
                               config['model']['input_size'],
                               len(config['model']['labels']),
                               config['model']['anchors'],
                               max_box_per_image=10)
    
    from yolo.trainer import YoloTrainer
    yolo_trainer = YoloTrainer(yolo_network.get_model(),
                               yolo_network.get_loss_func())

    yolo = YOLO(network             = yolo_network,
                trainer             = yolo_trainer,
                labels              = config['model']['labels'], 
                anchors             = config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################
    input_size = yolo_network.get_input_size()
    grid_size = yolo_network.get_grid_size()
    nb_box = yolo_network.get_nb_boxes()
    max_box_per_image = yolo_network.get_max_box_per_image()
    
    generator_config = GeneratorConfig(input_size,
                                       grid_size,
                                       nb_box,
                                       config['model']['labels'],
                                       config['train']['batch_size'],
                                       max_box_per_image,
                                       config['model']['anchors'])

    train_batch = BatchGenerator(train_imgs,
                                 generator_config,
                                 norm=yolo_network.get_normalize_func())

    valid_batch = BatchGenerator(valid_imgs,
                                 generator_config,
                                 norm=yolo_network.get_normalize_func(),
                                 jitter=False)

    yolo.train(train_batch,
               valid_batch,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epoch           = config['train']['nb_epoch'],
               warmup_epochs      = config['train']['warmup_epochs'],
               learning_rate      = config['train']['learning_rate'], 
               saved_weights_name = config['train']['saved_weights_name'])
