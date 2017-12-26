

import pytest
import cv2
import numpy as np
import os
from yolo.frontend import create_yolo
from yolo.utils.box import BoundBox


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
    desired_boxes = [BoundBox(0.50070397927, 0.585420268209, 0.680594700387, 0.758197716846, 0.576064, [ 0.57606441])]
    return desired_boxes

def test_predict(setup_inputs, setup_outputs):

    # 1. Given 
    image = setup_inputs
    desired_boxes = setup_outputs
    
    # 2. When run
    yolo = create_yolo("MobileNet",
                       ["raccoon"],
                       weights_path=os.path.join(SAMPLE_DIRECTORY, "mobilenet_raccoon.h5"))
    boxes = yolo.predict(image)
    
    # 3. should
    for box, desired_box in zip(boxes, desired_boxes):    
        variables = box.__dict__.keys()
        for v in variables:
            assert np.allclose(getattr(box, v), getattr(desired_box, v))
    
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])


    


