# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(111)
import argparse
import os
import json
from yolo.frontend import create_yolo, get_object_labels

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default="configs/from_scratch.json",
    help='path to configuration file')

def setup_training(config_file):
    """make directory to save weights & its configuration """
    import shutil
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    dirname = config['train']['saved_folder']
    if os.path.isdir(dirname):
        print("{} is already exists. Weight file in directory will be overwritten".format(dirname))
    else:
        print("{} is created.".format(dirname, dirname))
        os.makedirs(dirname)
    print("Weight file and Config file will be saved in \"{}\"".format(dirname))
    shutil.copyfile(config_file, os.path.join(dirname, "config.json"))
    return config, os.path.join(dirname, "weights.h5")


if __name__ == '__main__':
    args = argparser.parse_args()
    config, weight_file = setup_training(args.conf)
    
    if config['train']['is_only_detect']:
        labels = ["object"]
    else:
        if config['model']['labels']:
            labels = config['model']['labels']
        else:
            labels = get_object_labels(config['train']['train_annot_folder'])
    print(labels)

    # 1. Construct the model 
    yolo = create_yolo(config['model']['architecture'],
                       labels,
                       config['model']['input_size'],
                       config['model']['anchors'],
                       config['model']['coord_scale'],
                       config['model']['class_scale'],
                       config['model']['object_scale'],
                       config['model']['no_object_scale'])
    
    # 2. Load the pretrained weights (if any) 
    yolo.load_weights(config['pretrained']['full'], by_name=True)

    # 3. actual training 
    yolo.train(config['train']['train_image_folder'],
               config['train']['train_annot_folder'],
               config['train']['actual_epoch'],
               weight_file,
               config["train"]["batch_size"],
               config["train"]["jitter"],
               config['train']['learning_rate'], 
               config['train']['train_times'],
               config['train']['valid_times'],
               config['train']['valid_image_folder'],
               config['train']['valid_annot_folder'],
               config['train']['first_trainable_layer'],
               config['train']['is_only_detect'])
    # loss: 2.1691, train batch jitter=False
