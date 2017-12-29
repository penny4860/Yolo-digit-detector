

import pytest
import cv2
import numpy as np
import os
from yolo.frontend import create_yolo
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset")


@pytest.fixture(scope='function')
def setup_inputs(request):
    input_file = os.path.join(TEST_SAMPLE_DIR, "raccoon.jpg")
    image = cv2.imread(input_file)
    return image


@pytest.fixture(scope='function')
def setup_outputs(request):
    desired_boxes = np.array([[104, 86, 546, 402]])
    desired_probs = np.array([[ 0.57606441]])
    return desired_boxes, desired_probs

def test_predict(setup_inputs, setup_outputs):

    # 1. Given 
    image = setup_inputs
    desired_boxes, desired_probs = setup_outputs
    
    # 2. When run
    yolo = create_yolo("MobileNet",
                       ["raccoon"])
    yolo.load_weights(os.path.join(TEST_SAMPLE_DIR, "mobilenet_raccoon.h5"))
    boxes, probs = yolo.predict(image)

    # 3. should
    assert np.allclose(boxes, desired_boxes)
    assert np.allclose(probs, desired_probs)
    
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])


    


