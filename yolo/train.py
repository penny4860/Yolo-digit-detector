# -*- coding: utf-8 -*-
import json
import os
import numpy as np
from yolo.utils.annotation import parse_annotation
from yolo.utils.trainer import train_yolo
from yolo import YOLO


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
                config['model']['labels'],
                config['model']['input_size'],
                config['model']['max_box_per_image'],
                config['model']['anchors'])
    # 2. Load the pretrained weights (if any) 
    yolo.load_weights(config['train']['pretrained_weights'])

    # 3. Parse the annotations 
    train_annotations, valid_annotations = _parse(config)

    # 4. get batch generator
    # Todo : train_imgs 를 class 로 정의하자.
    train_batch_generator = yolo.get_batch_generator(train_annotations,
                                                    config["train"]["batch_size"],
                                                    jitter=False)
    valid_batch_generator = yolo.get_batch_generator(valid_annotations,
                                                    config["train"]["batch_size"],
                                                    jitter=False)
    
    # 5. To train model get keras model instance & loss fucntion
    model = yolo.get_model()
    loss = yolo.get_loss_func(config['train']['batch_size'],
                              config['train']['warmup_epochs'],
                              config['train']['train_times'],
                              config['valid']['valid_times'])
    
    # 6. Run training loop
    train_yolo(model,
               loss,
               train_batch_generator,
               valid_batch_generator,
               learning_rate      = config['train']['learning_rate'], 
               nb_epoch           = config['train']['nb_epoch'],
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               saved_weights_name = config['train']['saved_weights_name'],
               )
