# -*- coding: utf-8 -*-
# This driver stores the prediction results of the images specified by annotation.
import argparse
import json
import cv2
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_scaled_boxes
from yolo.backend.utils.annotation import parse_annotation
import os
import yolo


DEFAULT_CONFIG_FILE = os.path.join(yolo.PROJECT_ROOT, "configs", "raccoon.json")
DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "weights", "raccoon.h5")
DEFAULT_THRESHOLD = 0.3

argparser = argparse.ArgumentParser(
    description='Predict digits driver')

argparser.add_argument(
    '-c',
    '--conf',
    default=DEFAULT_CONFIG_FILE,
    help='path to configuration file')

argparser.add_argument(
    '-t',
    '--threshold',
    default=DEFAULT_THRESHOLD,
    help='detection threshold')

argparser.add_argument(
    '-w',
    '--weights',
    default=DEFAULT_WEIGHT_FILE,
    help='trained weight files')

if __name__ == '__main__':
    # 1. extract arguments
    args = argparser.parse_args()
    with open(args.conf) as config_buffer:
        config = json.loads(config_buffer.read())
    model_config = config['model']

    # 2. create yolo instance & predict
    yolo = create_yolo(model_config['architecture'],
                       model_config['labels'],
                       model_config['input_size'],
                       model_config['anchors'])
    yolo.load_weights(args.weights)

    # 3. read image
    write_dname = "detected"
    if not os.path.exists(write_dname): os.makedirs(write_dname)
    annotations = parse_annotation(config['train']['valid_annot_folder'],
                                   config['train']['valid_image_folder'],
                                   config['model']['labels'],
                                   is_only_detect=config['train']['is_only_detect'])
    for i in range(len(annotations)):
        img_path = annotations.fname(i)
        img_fname = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        boxes, probs = yolo.predict(image, float(args.threshold))
      
        # 4. save detection result
        image = draw_scaled_boxes(image, boxes, probs, model_config['labels'])
        output_path = os.path.join(write_dname, os.path.split(img_fname)[-1])
        
        cv2.imwrite(output_path, image)
        print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))

    
