# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np
import cv2

from yolo.backend.utils.box import to_centroid
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn")

# Todo : test_train duplicated code
@pytest.fixture(scope='module')
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

@pytest.fixture(scope='module')
def setup_image_and_its_boxes(request):
    input_file = os.path.join(TEST_SAMPLE_DIR, "imgs", "1.png")
    image = cv2.imread(input_file)
    true_boxes = to_centroid(np.array([[246, 77, 327, 296],
                                       [323, 81, 419, 300]]))
    
    return image, true_boxes




