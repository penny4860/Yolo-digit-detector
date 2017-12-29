#! /usr/bin/env python
# Todo : sample driver file => ipython notebook description
import argparse
import json
import cv2
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_boxes
import os
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset")


DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_INPUT_IMAGE = os.path.join(TEST_SAMPLE_DIR, "raccoon.jpg")

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default=DEFAULT_CONFIG_FILE,
    help='path to configuration file')

argparser.add_argument(
    '-i',
    '--input',
    default=DEFAULT_INPUT_IMAGE,
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-t',
    '--threshold',
    default=0.3,
    help='detection threshold')

if __name__ == '__main__':
    # 1. extract arguments
    args = argparser.parse_args()
    with open(args.conf) as config_buffer:
        config = json.loads(config_buffer.read())
    model_config = config['model']

    # 2. read image
    image = cv2.imread(args.input)

    # Todo : pretrained feature message
    # 3. create yolo instance & predict
    yolo = create_yolo(model_config['architecture'],
                       model_config['labels'],
                       model_config['input_size'],
                       model_config['max_box_per_image'],
                       model_config['anchors'],
                       feature_weights_path=None)
    yolo.load_weights(config['pretrained']['full'])

    boxes, probs = yolo.predict(image, args.threshold)

    # 4. save detection result
    output_path = args.input[:-4] + '_detected' + args.input[-4:]
    image = draw_boxes(image, boxes, probs, model_config['labels'])
    cv2.imwrite(output_path, image)
    print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))

    
