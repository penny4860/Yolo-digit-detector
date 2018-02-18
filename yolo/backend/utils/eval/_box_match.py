# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

class BoxMatcher(object):
    """
    # Args
        boxes1 : ndarray, shape of (N, 4)
            (x1, y1, x2, y2) ordered

        boxes2 : ndarray, shape of (M, 4)
            (x1, y1, x2, y2) ordered
    """
    
    def __init__(self, boxes1, boxes2, labels1=None, labels2=None):
        self._boxes1 = boxes1
        self._boxes2 = boxes2

        if len(boxes1) == 0 or len(boxes2) == 0:
            pass
        else:
            
            if labels1 is None or labels2 is None:
                self._iou_matrix = self._calc(boxes1,
                                              boxes2,
                                              np.ones((len(boxes1),)),
                                              np.ones((len(boxes2),)))
            else:
                self._iou_matrix = self._calc(boxes1, boxes2, labels1, labels2)
            self._match_pairs = linear_assignment(-1*self._iou_matrix)
    
    def match_idx_of_box1_idx(self, box1_idx):
        """
        # Args
            box1_idx : int
        
        # Returns
            box2_idx : int or None
                if matching index does not exist, return None
            iou : float
                IOU (intersection over union) between the box corresponding to the box1 index and the box2 matching it
        """
        assert box1_idx < len(self._boxes1)
        if len(self._boxes2) == 0:
            return None, 0
        
        box1_matching_idx_list = self._match_pairs[:, 0]
        box2_matching_idx_list = self._match_pairs[:, 1]
        box2_idx = self._find(box1_idx, box1_matching_idx_list, box2_matching_idx_list)
        if box2_idx is None:
            iou = 0
        else:
            iou = self._iou_matrix[box1_idx, box2_idx]
        return box2_idx, iou

    def match_idx_of_box2_idx(self, box2_idx):
        """
        # Args
            box2_idx : int
         
        # Returns
            box1_idx : int or None
                if matching index does not exist, return None
            iou : float
                IOU (intersection over union) between the box corresponding to the box2 index and the box1 matching it
        """
        assert box2_idx < len(self._boxes2)
        if len(self._boxes1) == 0:
            return None, 0

        box1_matching_idx_list = self._match_pairs[:, 0]
        box2_matching_idx_list = self._match_pairs[:, 1]
        box1_idx = self._find(box2_idx, box2_matching_idx_list, box1_matching_idx_list)
        if box1_idx is None:
            iou = 0
        else:
            iou = self._iou_matrix[box1_idx, box2_idx]
        return box1_idx, iou

    def _find(self, input_idx, input_idx_list, output_idx_list):
        if input_idx in input_idx_list:
            loc = np.where(input_idx_list == input_idx)[0][0]
            output_idx = int(output_idx_list[loc])
        else:
            output_idx = None
        return output_idx
    
    def _calc_maximun_ious(self):
        ious_for_each_gt = self._calc(self._boxes1, self._boxes2)
        ious = np.max(ious_for_each_gt, axis=0)
        return ious
    
    def _calc(self, boxes, true_boxes, labels, true_labels):
        ious_for_each_gt = []
        
        for truth_box, truth_label in zip(true_boxes, true_labels):
            
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            
            x1_gt = truth_box[0]
            y1_gt = truth_box[1]
            x2_gt = truth_box[2]
            y2_gt = truth_box[3]
            
            xx1 = np.maximum(x1, x1_gt)
            yy1 = np.maximum(y1, y1_gt)
            xx2 = np.minimum(x2, x2_gt)
            yy2 = np.minimum(y2, y2_gt)
        
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            intersections = w*h
            As = (x2 - x1 + 1) * (y2 - y1 + 1)
            B = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
            
            label_score = (labels == truth_label).astype(np.float)
            
            ious = label_score * intersections.astype(float) / (As + B -intersections)
            ious_for_each_gt.append(ious)
        
        # (n_truth, n_boxes)
        ious_for_each_gt = np.array(ious_for_each_gt)
        return ious_for_each_gt.T


if __name__ == "__main__":
    labels = np.array([1,2,3,4])
    label = np.array([4])
    expected = np.array([0, 0, 0, 1])
    label_score = (labels == label).astype(np.float)
    print(label_score)
    
    
    labels = np.array(["a","bb","a","cc"])
    label = np.array(["cc"])
    label_score = (labels == label).astype(np.float)
    print(label_score)
    
    
    