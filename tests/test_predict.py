

from yolo.predict import predict
import pytest
import cv2
import numpy as np
import os

SAMPLE_DIRECTORY = "..//sample"


def test_predict():
    def is_equal_image_file(output_file, desired_file):
        output_img = cv2.imread(output_file)
        desired_img = cv2.imread(desired_file)
        return np.allclose(output_img, desired_img)

    # 1. Given 
    output_file = os.path.join(SAMPLE_DIRECTORY, "raccoon_detected.jpg")
    desired_file = os.path.join(SAMPLE_DIRECTORY, "raccoon_detected_gt.jpg")
    if os.path.exists(output_file):
        os.remove(output_file)
    # 2. When run main()
    conf = os.path.join(SAMPLE_DIRECTORY, "config.json")
    weights = os.path.join(SAMPLE_DIRECTORY, "mobilenet_raccoon.h5")
    predict(os.path.join(SAMPLE_DIRECTORY, "raccoon.jpg"),
            weights,
            conf)

    # 3. Should 
    return is_equal_image_file(output_file, desired_file)
    
if __name__ == '__main__':
    pytest.main([__file__])


    


