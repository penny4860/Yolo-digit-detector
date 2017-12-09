#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from yolo.preprocessing import parse_annotation
from yolo.utils import draw_boxes
from yolo.frontend import YOLO


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

def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(list(range(nb_frames))):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  
    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

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

        _main_(args)

        # 3. Should 
        return is_equal_image_file(output_file, desired_file)

    if test_predict() == True:
        print("predict passed")
    else:
        print("predict failed")


    
