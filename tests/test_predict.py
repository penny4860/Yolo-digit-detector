# -*- coding: utf-8 -*-

import pytest
import numpy as np
import os
from yolo.frontend import create_yolo
from yolo.backend.utils.box import to_centroid, centroid_box_iou
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn")

@pytest.fixture(scope='function')
def setup_outputs(request):
    desired_boxes = np.array([[104, 86, 546, 402]])
    desired_probs = np.array([[ 0.57606441]])
    return desired_boxes, desired_probs

def test_predict(setup_input_image, setup_outputs, setup_centroid_true_boxes, setup_model_config):

    # 1. Given 
    image = setup_input_image
    model_config = setup_model_config

    desired_boxes, desired_probs = setup_outputs
    
    # 2. When run
    # 1. Construct the model 
    yolo = create_yolo(model_config['architecture'],
                       model_config['labels'],
                       model_config['input_size'],
                       model_config['max_box_per_image'],
                       model_config['anchors'])
    
    yolo.load_weights(os.path.join(TEST_SAMPLE_DIR, "mobile_288_weights.h5"))
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


    


