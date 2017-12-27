

import pytest
import cv2
import numpy as np
import os
from yolo.frontend import create_yolo
from yolo.backend.utils.box import BoundBox


THIS_DIRECTORY = os.path.dirname(__file__)
SAMPLE_DIRECTORY = os.path.join(os.path.dirname(THIS_DIRECTORY),
                                "sample")

@pytest.fixture(scope='function')
def setup_inputs(request):
    input_file = os.path.join(SAMPLE_DIRECTORY, "raccoon.jpg")
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
                       ["raccoon"],
                       weights_path=os.path.join(SAMPLE_DIRECTORY, "mobilenet_raccoon.h5"))
    boxes, probs = yolo.predict(image)

    # 3. should
    assert np.allclose(boxes, desired_boxes)
    assert np.allclose(probs, desired_probs)
    
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])


    


