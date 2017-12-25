
import numpy as np
import cv2

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

    def get_label(self):
        return np.argmax(self.classes)
    
    def get_score(self):
        return self.classes[self.get_label()]
    
    def iou(self, bound_box):
        b1 = self.as_centroid()
        b2 = bound_box.as_centroid()
        return centroid_box_iou(b1, b2)

    def as_centroid(self):
        return np.array([self.x, self.y, self.w, self.h])
        

def draw_boxes(image, boxes, labels):
    
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image, 
                    labels[box.get_label()] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,255,0), 2)
        
    return image        


def centroid_box_iou(box1, box2):
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
    
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
    
    _, _, w1, h1 = box1.reshape(-1,)
    _, _, w2, h2 = box2.reshape(-1,)
    x1_min, y1_min, x1_max, y1_max = to_minmax(box1.reshape(-1,4)).reshape(-1,)
    x2_min, y2_min, x2_max, y2_max = to_minmax(box2.reshape(-1,4)).reshape(-1,)
            
    intersect_w = _interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = _interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    intersect = intersect_w * intersect_h
    union = w1 * h1 + w2 * h2 - intersect
    
    return float(intersect) / union


def to_centroid(minmax_boxes):
    """
    minmax_boxes : (N, 4)
    """
    minmax_boxes = minmax_boxes.astype(np.float)
    centroid_boxes = np.zeros_like(minmax_boxes)
    
    x1 = minmax_boxes[:,0]
    y1 = minmax_boxes[:,1]
    x2 = minmax_boxes[:,2]
    y2 = minmax_boxes[:,3]
    
    centroid_boxes[:,0] = (x1 + x2) / 2
    centroid_boxes[:,1] = (y1 + y2) / 2
    centroid_boxes[:,2] = x2 - x1
    centroid_boxes[:,3] = y2 - y1
    return centroid_boxes

def to_minmax(centroid_boxes):
    centroid_boxes = centroid_boxes.astype(np.float)
    minmax_boxes = np.zeros_like(centroid_boxes)
    
    cx = centroid_boxes[:,0]
    cy = centroid_boxes[:,1]
    w = centroid_boxes[:,2]
    h = centroid_boxes[:,3]
    
    minmax_boxes[:,0] = cx - w/2
    minmax_boxes[:,1] = cy - h/2
    minmax_boxes[:,2] = cx + w/2
    minmax_boxes[:,3] = cy + h/2
    return minmax_boxes

def to_normalize(boxes, scale):
    """
    box coordinates -> (scale, scale)
    """
    return boxes / float(scale)

def create_anchor_boxes(anchors):
    """
    # Args
        anchors : list of floats
    # Returns
        boxes : array, shape of (len(anchors)/2, 4)
            centroid-type
    """
    boxes = []
    n_boxes = int(len(anchors)/2)
    for i in range(n_boxes):
        boxes.append(np.array([0, 0, anchors[2*i], anchors[2*i+1]]))
    return np.array(boxes)

def find_match_box(centroid_box, centroid_boxes):
    """Find the index of the boxes with the largest overlap among the N-boxes.

    # Args
        box : array, shape of (1, 4)
        boxes : array, shape of (N, 4)
    
    # Return
        match_index : int
    """
    match_index = -1
    max_iou     = -1
    
    for i, box in enumerate(centroid_boxes):
        iou = centroid_box_iou(centroid_box, box)
        
        if max_iou < iou:
            match_index = i
            max_iou     = iou
    return match_index

import pytest
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

