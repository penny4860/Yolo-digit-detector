#! /usr/bin/env python

import argparse
import os

from yolo.predict import predict

DEFAULT_CONFIG_FILE = "sample//config.json"
DEFAULT_PRETRAINED_WEIGHTS = "sample//mobilenet_raccoon.h5"
DEFAULT_INPUT_IMAGE = "sample//raccoon.jpg"


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default=DEFAULT_CONFIG_FILE,
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default=DEFAULT_PRETRAINED_WEIGHTS,
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    default=DEFAULT_INPUT_IMAGE,
    help='path to an image or an video (mp4 format)')

if __name__ == '__main__':
    args = argparser.parse_args()
    predict(args.input, args.weights, args.conf)


    
