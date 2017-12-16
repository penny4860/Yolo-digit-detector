# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(111)
import argparse
import os
from yolo.train import train
from yolo.predict import predict

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
#     args = argparser.parse_args()
#     train(args.conf)
#     # loss: 2.1691, train batch jitter=False
    
    predict("sample//raccoon_train_imgs//raccoon-72.jpg",
            "test_raccoon_weight.h5",
            config_path="config.json")

