# -*- coding: utf-8 -*-
import json
from yolo.frontend import create_yolo

def train_yolo(conf):

    with open(conf) as config_buffer:
        config = json.loads(config_buffer.read())

    # 1. Construct the model 
    yolo = create_yolo(config['model']['architecture'],
                       config['model']['labels'],
                       config['model']['input_size'],
                       config['model']['max_box_per_image'],
                       config['model']['anchors'])
    
    # 2. Load the pretrained weights (if any) 
    yolo.load_weights(config['train']['pretrained_weights'])
    
    # 3. Parse the annotations 
    yolo.train(config['train']['train_image_folder'],
               config['train']['train_annot_folder'],
               config['train']['nb_epoch'],
               config['train']['saved_weights_name'],
               config["train"]["batch_size"],
               False,
               config['train']['learning_rate'], 
               config['train']['train_times'],
               config['valid']['valid_times'],
               config['train']['warmup_epochs'],
               config['valid']['valid_image_folder'],
               config['valid']['valid_annot_folder'])
           


