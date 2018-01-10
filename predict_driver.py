#! /usr/bin/env python
# Todo : sample driver file => ipython notebook description
import argparse
import json
import cv2
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_boxes
import os
import yolo
import glob


DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_INPUT_IMAGE = os.path.join("tests", "dataset", "svhn", "imgs")

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
    help='image directory path')

argparser.add_argument(
    '-t',
    '--threshold',
    default=0.4,
    help='detection threshold')

argparser.add_argument(
    '-w',
    '--weights',
    default="overfit//weights.h5",
    help='trained weight files')

def get_write_dir(image_dir):
    par_dname = os.path.dirname(image_dir)
    cur_dname = os.path.split(image_dir)[-1]
    detected_dname = os.path.join(par_dname, cur_dname + "_detected")
    if not os.path.exists(detected_dname): os.makedirs(detected_dname)
    return detected_dname

if __name__ == '__main__':
    # 1. extract arguments
    args = argparser.parse_args()
    with open(args.conf) as config_buffer:
        config = json.loads(config_buffer.read())
    model_config = config['model']

    # Todo : pretrained feature message
    # 2. create yolo instance & predict
    yolo = create_yolo(model_config['architecture'],
                       model_config['labels'],
                       model_config['input_size'],
                       model_config['anchors'],
                       feature_weights_path=None)
    yolo.load_weights(args.weights)

    # 3. read image
    write_dname = get_write_dir(args.input)
    image_files = glob.glob(os.path.join(args.input, "*.png"))
    
    
    # 4. 
    files = os.listdir(config['train']['train_annot_folder'])
    
    for fname in files:
        fname_ =  os.path.splitext(fname)[0]
        img_fname = fname_ + ".png"
        img_path = os.path.join(config['train']['train_image_folder'], img_fname)
    
        image = cv2.imread(fname)
        boxes, probs = yolo.predict(image, float(args.threshold))
      
        # 4. save detection result
        image = draw_boxes(image, boxes, probs, model_config['labels'])
        output_path = os.path.join(write_dname, os.path.split(fname)[-1])
        cv2.imwrite(output_path, image)
        print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))

    
