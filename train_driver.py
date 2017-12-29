# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(111)
import argparse
import os
import json
from yolo.frontend import create_yolo

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default="config.json",
    help='path to configuration file')

if __name__ == '__main__':
    args = argparser.parse_args()
    
    with open(args.conf) as config_buffer:
        config = json.loads(config_buffer.read())

    # 1. Construct the model 
    yolo = create_yolo(config['model']['architecture'],
                       config['model']['labels'],
                       config['model']['input_size'],
                       config['model']['max_box_per_image'],
                       config['model']['anchors'],
                       config['weights']['feature'])
    
    # 2. Load the pretrained weights (if any) 
    yolo.load_weights(config['weights']['pretrained'])
    
    # 3. Parse the annotations 
    yolo.train(config['train']['train_image_folder'],
               config['train']['train_annot_folder'],
               config['train']['nb_epoch'],
               config['train']['saved_weights_name'],
               config["train"]["batch_size"],
               config["train"]["jitter"],
               config['train']['learning_rate'], 
               config['train']['train_times'],
               config['train']['valid_times'],
               config['train']['warmup_epochs'],
               config['train']['valid_image_folder'],
               config['train']['valid_annot_folder'])
    # loss: 2.1691, train batch jitter=False

