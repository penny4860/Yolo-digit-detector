#-*- coding: utf-8 -*-
import numpy as np

class OverlapCalculator:
    
    def __init__(self):
        pass
    
    def calc_ious_per_truth(self, boxes, true_boxes):
        return self._calc(boxes, true_boxes)
    
    def calc_maximun_ious(self, boxes, true_boxes):
        ious_for_each_gt = self._calc(boxes, true_boxes)
        ious = np.max(ious_for_each_gt, axis=0)
        return ious
    
    def _calc(self, boxes, true_boxes):
        ious_for_each_gt = []
        
        for truth_box in true_boxes:
            y1 = boxes[:, 0]
            y2 = boxes[:, 1]
            x1 = boxes[:, 2]
            x2 = boxes[:, 3]
            
            y1_gt = truth_box[0]
            y2_gt = truth_box[1]
            x1_gt = truth_box[2]
            x2_gt = truth_box[3]
            
            xx1 = np.maximum(x1, x1_gt)
            yy1 = np.maximum(y1, y1_gt)
            xx2 = np.minimum(x2, x2_gt)
            yy2 = np.minimum(y2, y2_gt)
        
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            intersections = w*h
            As = (x2 - x1 + 1) * (y2 - y1 + 1)
            B = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
            
            ious = intersections.astype(float) / (As + B -intersections)
            ious_for_each_gt.append(ious)
        
        # (n_truth, n_boxes)
        ious_for_each_gt = np.array(ious_for_each_gt)
        return ious_for_each_gt

if __name__ == '__main__':
    pass
    