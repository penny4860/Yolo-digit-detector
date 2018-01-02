# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(111)
import os
import pytest
import cv2
from yolo.frontend import create_yolo
from yolo.backend.utils.box import to_centroid, centroid_box_iou
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn")


@pytest.fixture(scope='function')
def setup_weights_file(request):
    pretrained_feature_file = os.path.join(yolo.PROJECT_ROOT,
                                           "tests",
                                           "dataset",
                                           "mobilenet_features.h5")
    weight_file = os.path.join(TEST_SAMPLE_DIR, "weights.h5")
    def teardown():
        os.remove(weight_file)
    request.addfinalizer(teardown)
    
    return pretrained_feature_file, weight_file

@pytest.fixture(scope='function')
def setup_model_config(request):
    model_config = {"architecture":         "MobileNet",
                    "input_size":           288,
                    "anchors":              [0.57273, 0.677385,
                                             1.87446, 2.06253,
                                             3.33843, 5.47434,
                                             7.88282, 3.52778,
                                             9.77052, 9.16828],
                    "max_box_per_image":    10,        
                    "labels":               ["1", "2", "3", "9"]}
    return model_config
    
@pytest.fixture(scope='function')
def setup_train_config(request):
    config = {
        "train_times":          10,
        "valid_times":          1,
        "batch_size":           2,
        "learning_rate":        1e-4,
        "nb_epoch":             50,
        "jitter":    False
    }
    return config

@pytest.fixture(scope='function')
def setup_dataset_folder(request):
    img_folder = os.path.join(TEST_SAMPLE_DIR, "imgs/")
    ann_folder = os.path.join(TEST_SAMPLE_DIR, "anns/")
    return img_folder, ann_folder

@pytest.fixture(scope='function')
def setup_input_image(request):
    input_file = os.path.join(TEST_SAMPLE_DIR, "imgs", "1.png")
    image = cv2.imread(input_file)
    return image

@pytest.fixture(scope='function')
def setup_centroid_true_boxes(request):
    true_boxes = to_centroid(np.array([[246, 77, 327, 296],
                                       [323, 81, 419, 300]]))
    return true_boxes


def test_train_yolo_framework(setup_model_config,
                              setup_weights_file,
                              setup_dataset_folder,
                              setup_input_image,
                              setup_centroid_true_boxes):
    model_config = setup_model_config
    pretrained_feature_file, weight_file = setup_weights_file
    img_folder, ann_folder = setup_dataset_folder

    # 1. Construct the model 
    yolo = create_yolo(model_config['architecture'],
                       model_config['labels'],
                       model_config['input_size'],
                       model_config['max_box_per_image'],
                       model_config['anchors'],
                       pretrained_feature_file)
    
    # 2. warmup training
    yolo.train(img_folder, ann_folder,
               3,
               weight_file,
               2,
               False,
               1e-4, 
               10,
               1,
               3,
               img_folder, ann_folder)
    # 3. Load the warmup trained weights
    yolo.load_weights(weight_file)
    
    # 4. actual training 
    yolo.train(img_folder, ann_folder,
               50,
               weight_file,
               2,
               False,
               1e-4, 
               10,
               1,
               0,
               img_folder, ann_folder)

    # 5. Load training image & predict objects
    image = setup_input_image
    boxes, probs = yolo.predict(image)
    boxes = to_centroid(boxes)
    true_boxes = setup_centroid_true_boxes

    assert len(boxes) == 2
    assert len(probs) == 2
    assert np.allclose(np.argmax(probs, axis=1), [0, 3])
    for box, true_box in zip(boxes, true_boxes):
        iou = centroid_box_iou(box, true_box)
        assert iou > 0.5

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

