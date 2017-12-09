#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np

from yolo.predict import predict

DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_PRETRAINED_WEIGHTS = "mobilenet_raccoon.h5"
DEFAULT_INPUT_IMAGE = "tests//raccoon.jpg"


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
    def test_predict():
        def is_equal_image_file(output_file, desired_file):
            output_img = cv2.imread(output_file)
            desired_img = cv2.imread(desired_file)
            return np.allclose(output_img, desired_img)

        # 1. Given 
        output_file = "tests//raccoon_detected.jpg"
        desired_file = "tests//raccoon_detected_gt.jpg"
        if os.path.exists(output_file):
            os.remove(output_file)
        # 2. When run main()
        args = argparser.parse_args()
        args.conf = "tests//config.json"
        args.weights = "tests//mobilenet_raccoon.h5"
        args.input = "tests//raccoon.jpg"

        predict(args.input, args.weights, args.input)

        # 3. Should 
        return is_equal_image_file(output_file, desired_file)

    if test_predict() == True:
        print("predict passed")
    else:
        print("predict failed")


    
