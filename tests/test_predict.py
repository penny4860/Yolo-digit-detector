

import pytest
import cv2
import numpy as np
import os
from yolo.frontend import create_yolo
from yolo.backend.utils.box import to_centroid, centroid_box_iou
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn")

@pytest.fixture(scope='function')
def setup_centroid_true_boxes(request):
    true_boxes = to_centroid(np.array([[246, 77, 327, 296],
                                       [323, 81, 419, 300]]))
    return true_boxes

@pytest.fixture(scope='function')
def setup_inputs(request):
    input_image_file = os.path.join(TEST_SAMPLE_DIR, "imgs", "1.png")
    image = cv2.imread(input_image_file)
    return image

@pytest.fixture(scope='function')
def setup_outputs(request):
    desired_boxes = np.array([[104, 86, 546, 402]])
    desired_probs = np.array([[ 0.57606441]])
    return desired_boxes, desired_probs

def test_predict(setup_inputs, setup_outputs, setup_centroid_true_boxes, setup_model_config):

    # 1. Given 
    image = setup_inputs
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


    


