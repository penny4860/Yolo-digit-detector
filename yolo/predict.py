
import json
from yolo.box import draw_boxes
from yolo import YOLO
import cv2
from yolo.network import YoloNetwork

def predict(image_path, weights_path, config_path="config.json"):

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################
    yolo_network = YoloNetwork(config['model']['architecture'],
                               config['model']['input_size'],
                               len(config['model']['labels']),
                               max_box_per_image=10)

    from yolo.loss import YoloLoss
    yolo_loss = YoloLoss(yolo_network.get_grid_size(),
                         config['model']['anchors'],
                         yolo_network.get_nb_boxes(),
                         len(config['model']['labels']),
                         yolo_network.true_boxes)

    yolo = YOLO(network             = yolo_network,
                loss                = yolo_loss,
                trainer             = None,
                labels              = config['model']['labels'], 
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################
    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, config['model']['labels'])

    print(len(boxes), 'boxes are found')

    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
