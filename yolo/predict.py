
import json
from yolo.utils.box import draw_boxes
from yolo import YOLO
import cv2

def predict(image_path, weights_path, conf="config.json"):

    with open(conf) as config_buffer:
        config = json.loads(config_buffer.read())

    # 1. Construct the model 
    yolo = YOLO(config['model']['architecture'],
                config['model']['labels'],
                config['model']['input_size'],
                config['model']['max_box_per_image'],
                config['model']['anchors'])
    # 2. Load the pretrained weights (if any) 
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################
    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, config['model']['labels'])

    print(len(boxes), 'boxes are found')

    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
