# -*- coding: utf-8 -*-

import pytest
import os
import numpy as np
from yolo.backend.utils.annotation import parse_annotation
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn")

@pytest.fixture(scope='function')
def setup_inputs(request):
    annotation_dir = os.path.join(TEST_SAMPLE_DIR, "anns//")
    image_dir = os.path.join(TEST_SAMPLE_DIR, "imgs//")
    return annotation_dir, image_dir

@pytest.fixture(scope='function')
def setup_expected_outputs(request, setup_inputs):
    _, image_dir = setup_inputs
    seen_labels = {'raccoon': 1}
    filenames = [os.path.join(image_dir, 'raccoon-1.jpg')]
    minmax_boxes = np.array([[81, 88, 522, 408]])
    labels = [["raccoon"]]
    return filenames, minmax_boxes, labels, seen_labels


def test_parse_annotation(setup_inputs, setup_expected_outputs):
    
    # Given
    annotation_dir, image_dir = setup_inputs
    
    # When
    annotations = parse_annotation(annotation_dir, image_dir)
    
    filenames, minmax_boxes, labels, seen_labels_ = setup_expected_outputs
    
    assert seen_labels == seen_labels_
    assert annotations.fname(0) == filenames[0]
    assert np.allclose(annotations.boxes(0), minmax_boxes[0])
    

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
