
import numpy as np
from yolo.box import BoundBox, bbox_iou

class YoloDecoder(object):
    
    def __init__(self, anchors, obj_threshold=0.3, nms_threshold=0.3):
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
        netout[..., 4]  = self._sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * self._softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > self._obj_threshold
        
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,b,5:]
                    
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row,col,b,:4]

                        x = (col + self._sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + self._sigmoid(y)) / grid_h # center position, unit: image height
                        w = self._anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = self._anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                        confidence = netout[row,col,b,4]
                        
                        box = BoundBox(x, y, w, h, confidence, classes)
                        
                        boxes.append(box)

        # suppress non-maximal boxes
        for c in range(len(classes)):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                if boxes[index_i].classes[c] == 0: 
                    continue
                else:
                    for j in range(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        if bbox_iou(boxes[index_i], boxes[index_j]) >= self._nms_threshold:
                            boxes[index_j].classes[c] = 0
                            
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.get_score() > self._obj_threshold]
        
        # Todo : python primitive type?
        return boxes

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)
        if np.min(x) < t:
            x = x/np.min(x)*t
        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)

