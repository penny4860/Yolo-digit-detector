

from yolo.predict import predict
import pytest
import cv2
import numpy as np
import os


THIS_DIRECTORY = os.path.dirname(__file__)
SAMPLE_DIRECTORY = os.path.join(os.path.dirname(THIS_DIRECTORY),
                                "sample")

@pytest.fixture(scope='function')
def setup_inputs(request):
    input_file = os.path.join(SAMPLE_DIRECTORY, "raccoon.jpg")
    conf = os.path.join(SAMPLE_DIRECTORY, "config.json")
    weights = os.path.join(SAMPLE_DIRECTORY, "mobilenet_raccoon.h5")
    return input_file, conf, weights


@pytest.fixture(scope='function')
def setup_outputs(request):
    def _delete_file(fname):
        if os.path.exists(fname):
            os.remove(fname)
    output_file = os.path.join(SAMPLE_DIRECTORY, "raccoon_detected.jpg")
    desired_file = os.path.join(SAMPLE_DIRECTORY, "raccoon_detected_gt.jpg")
    _delete_file(output_file)
    def teardown():
        _delete_file(output_file)
    request.addfinalizer(teardown)
    return output_file, desired_file


def test_predict(setup_inputs, setup_outputs):
    def is_equal_image_file(output_file, desired_file):
        output_img = cv2.imread(output_file)
        desired_img = cv2.imread(desired_file)
        return np.allclose(output_img, desired_img)

    # 1. Given 
    input_file, conf, weights = setup_inputs
    output_file, desired_file = setup_outputs
    
    # 2. When run main()
    predict(input_file, weights, conf)

    # 3. Should 
    assert(is_equal_image_file(output_file, desired_file) == True)
    
    
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])


    


