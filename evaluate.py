# -*- coding: utf-8 -*-

# This driver performs 2-functions for the validation images specified in configuration file:
#     1) evaluate fscore of validation images.
#     2) stores the prediction results of the validation images.

import argparse
import json
import cv2
import numpy as np
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_scaled_boxes
from yolo.backend.utils.annotation import parse_annotation
from yolo.backend.utils.eval.fscore import count_true_positives, calc_score

import os
import yolo


DEFAULT_CONFIG_FILE = os.path.join(yolo.PROJECT_ROOT, "svhn", "config.json")
DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "svhn", "weights.h5")
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

    # 2. create yolo instance & predict
    yolo = create_yolo(config['model']['architecture'],
                       config['model']['labels'],
                       config['model']['input_size'],
                       config['model']['anchors'])
    yolo.load_weights(args.weights)

    # 3. read image
    write_dname = "detected"
    if not os.path.exists(write_dname): os.makedirs(write_dname)
    annotations = parse_annotation(config['train']['valid_annot_folder'],
                                   config['train']['valid_image_folder'],
                                   config['model']['labels'],
                                   is_only_detect=config['train']['is_only_detect'])

    n_true_positives = 0
    n_truth = 0
    n_pred = 0
    for i in range(len(annotations)):
        img_path = annotations.fname(i)
        img_fname = os.path.basename(img_path)
        image = cv2.imread(img_path)
        true_boxes = annotations.boxes(i)
        true_labels = annotations.code_labels(i)
        
        boxes, probs = yolo.predict(image, float(args.threshold))
        labels = np.argmax(probs, axis=1) if len(probs) > 0 else [] 
      
        # 4. save detection result
        image = draw_scaled_boxes(image, boxes, probs, config['model']['labels'])
        output_path = os.path.join(write_dname, os.path.split(img_fname)[-1])
        
        cv2.imwrite(output_path, image)
        print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))

        n_true_positives += count_true_positives(boxes, true_boxes, labels, true_labels)
        n_truth += len(true_boxes)
        n_pred += len(boxes)
    print(calc_score(n_true_positives, n_truth, n_pred))

