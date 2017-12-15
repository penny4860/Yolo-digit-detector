# -*- coding: utf-8 -*-

import pytest
import os
from yolo.annotation import parse_annotation

SAMPLE_DIRECTORY = "..//sample"

@pytest.fixture(scope='function')
def setup_inputs(request):
    annotation_dir = os.path.join(SAMPLE_DIRECTORY, "raccoon_train_annotations//")
    image_dir = os.path.join(SAMPLE_DIRECTORY, "raccoon_train_imgs//")
    return annotation_dir, image_dir

@pytest.fixture(scope='function')
def setup_expected_outputs(request, setup_inputs):
    _, image_dir = setup_inputs
    
    images = [{'object': [{'name': 'raccoon', 'xmin': 81, 'ymin': 88, 'xmax': 522, 'ymax': 408}],
               'filename': os.path.join(image_dir, 'raccoon-1.jpg'),
               'width': 650,
               'height': 417}]
    labels = {'raccoon': 1}
    return images, labels


def test_parse_annotation(setup_inputs, setup_expected_outputs):
    
    # Given
    annotation_dir, image_dir = setup_inputs
    
    # When
    all_imgs, seen_labels = parse_annotation(annotation_dir, image_dir)
    
    images, labels = setup_expected_outputs
    
    assert all_imgs == images
    assert seen_labels == labels
    

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
