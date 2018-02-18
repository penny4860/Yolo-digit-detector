# -*- coding: utf-8 -*-
import numpy as np
import pytest
from yolo.backend.utils.eval.fscore import count_true_positives

@pytest.fixture(scope='function')
def setup_boxes(request):
    detect_boxes = np.array([(100, 100, 200, 200), (105, 105, 210, 210), (120, 120, 290, 290)])
    true_boxes = np.array([(90, 90, 200, 200), (140, 140, 300, 300)])
    return detect_boxes, true_boxes

@pytest.fixture(scope='function')
def setup_labels(request):
    detect_labels = np.array(["raccoon", "raccoon", "human"])
    true_labels = np.array(["human", "raccoon"])
    return detect_labels, true_labels


def test_count_true_positives_in_case_of_only_matching_box_coord(setup_boxes):
    detect_boxes, true_boxes = setup_boxes
    n_true_positives = count_true_positives(detect_boxes, true_boxes)
    assert n_true_positives == 2

def test_count_true_positives_in_case_of_existing_labels(setup_boxes, setup_labels):
    detect_boxes, true_boxes = setup_boxes
    detect_labels, true_labels = setup_labels
    n_true_positives = count_true_positives(detect_boxes, true_boxes, detect_labels, true_labels)
    assert n_true_positives == 0

import pytest
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
