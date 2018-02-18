
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
    
    def __init__(self, boxes1, boxes2):
        self._boxes1 = boxes1
        self._boxes2 = boxes2

        if len(boxes1) == 0 or len(boxes2) == 0:
            pass
        else:
            self._iou_matrix = self._calc(boxes1, boxes2)
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
    
    def _calc(self, boxes, true_boxes):
        ious_for_each_gt = []
        
        for truth_box in true_boxes:
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
            
            ious = intersections.astype(float) / (As + B -intersections)
            ious_for_each_gt.append(ious)
        
        # (n_truth, n_boxes)
        ious_for_each_gt = np.array(ious_for_each_gt)
        return ious_for_each_gt.T


def count_true_positives(detect_boxes, true_boxes):
    """
    # Args
        detect_boxes : array, shape of (n_detected_boxes, 4)
        true_boxes : array, shape of (n_true_boxes, 4)
    """
    n_true_positives = 0
 
    matcher = BoxMatcher(detect_boxes, true_boxes)
    for i in range(len(detect_boxes)):
        matching_idx, iou = matcher.match_idx_of_box1_idx(i)
        # print(matching_idx, iou)
        if matching_idx is not None and iou > 0.5:
            n_true_positives += 1
    return n_true_positives


def eval_imgs(list_detect_boxes, list_true_boxes):
    """
    # Args
        detect_boxes : list of box-arrays
        true_boxes : list of box-arrays
    """
    n_true_positives = 0
    n_true_boxes = 0
    n_pred_boxes = 0
    for detect_boxes, true_boxes in zip(list_detect_boxes, list_true_boxes):
        n_true_positives += count_true_positives(detect_boxes, true_boxes)
        n_true_boxes += len(true_boxes)
        n_pred_boxes += len(detect_boxes)

    precision = n_true_positives / n_pred_boxes
    recall = n_true_positives / n_true_boxes
    fscore = 2* precision * recall / (precision + recall)
    score = {"fscore": fscore, "precision": precision, "recall": recall}
    return score
    

if __name__ == "__main__":
    detect_boxes = np.array([(100, 100, 200, 200), (105, 105, 210, 210), (120, 120, 290, 290)])
    true_boxes = np.array([(90, 90, 200, 200), (140, 140, 300, 300)])
 
    print(eval_imgs([detect_boxes], [true_boxes]))
    
    