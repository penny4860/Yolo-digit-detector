
import numpy as np
# from .utils.box import BoundBox
from yolo.backend.utils.box import BoundBox, nms_boxes

class YoloDecoder(object):
    
    def __init__(self,
                 anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                 obj_threshold=0.3,
                 nms_threshold=0.3):
        self._anchors = anchors
        self._obj_threshold = obj_threshold
        self._nms_threshold = nms_threshold

    def run(self, netout):
        """Convert Yolo network output to bounding box
        
        # Args
            netout : 4d-array, shape of (grid_h, grid_w, num of boxes per grid, 5 + n_classes)
                YOLO neural network output array
        
        # Returns
            boxes : list of BoundBox instance
        """
        grid_h, grid_w, nb_box = netout.shape[:3]

        boxes = []
        
        # decode the output by the network
        netout[..., 4]  = _sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > self._obj_threshold
        
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,b,5:]
                    
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row,col,b,:4]

                        x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                        w = self._anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = self._anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                        confidence = netout[row,col,b,4]
                        
                        box = BoundBox(x, y, w, h, confidence, classes)
                        
                        boxes.append(box)
        
        boxes = nms_boxes(boxes, len(classes), self._nms_threshold, self._obj_threshold)
        return boxes

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

import pytest
def test_yolo_decoding():
    netout = np.load("netout.npy")
    yolo_decoder = YoloDecoder()
    boxes = yolo_decoder.run(netout)
    assert np.allclose(boxes[0].x, 0.50070397927)
    assert np.allclose(boxes[0].y, 0.585420268209)
    assert np.allclose(boxes[0].w, 0.680594700387)
    assert np.allclose(boxes[0].h, 0.758197716846)
    assert np.allclose(boxes[0].c, 0.576064)
    assert np.allclose(boxes[0].classes, [ 0.57606441])
    

if __name__ == '__main__':
    pytest.main([__file__])
    
    


    
    
    
    
    
    
    
    
    
    
    

