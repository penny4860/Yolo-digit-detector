
import pytest
import numpy as np
from yolo.backend.utils.box import find_match_box, centroid_box_iou

@pytest.fixture(scope='module')
def setup_for_find_match_box():
    input_box = [0, 0, 9.96875, 9.96875]
    boxes = [[0, 0, 0.57273, 0.677385],
             [0, 0, 1.87446, 2.06253],
             [0, 0, 3.33843, 5.47434],
             [0, 0, 7.88282, 3.52778],
             [0, 0, 9.77052, 9.16828]]
    
    input_box = np.array(input_box)
    boxes = np.array(boxes)
    matching_idx = 4
    return input_box, boxes, matching_idx

@pytest.fixture(scope='module')
def setup_for_iou():
    box1 = [0, 0, 9.96875, 9.96875]
    box2 = [0, 0, 9.77052, 9.16828]
    
    box1 = np.array(box1)
    box2 = np.array(box2)
    iou = 0.901413
    return box1, box2, iou

def test_find_match_box(setup_for_find_match_box):
    input_box, boxes, expected_idx = setup_for_find_match_box
    matching_idx = find_match_box(input_box, boxes)
    assert matching_idx == expected_idx

def test_centroid_box_iou(setup_for_iou):
    box1, box2, expected_iou = setup_for_iou
    iou = centroid_box_iou(box1, box2)
    assert np.isclose(iou, expected_iou)

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
