# -*- coding: utf-8 -*-
import numpy as np
from yolo.backend.utils.eval._box_match import BoxMatcher

def count_true_positives(detect_boxes, true_boxes, detect_labels=None, true_labels=None):
    """
    # Args
        detect_boxes : array, shape of (n_detected_boxes, 4)
        true_boxes : array, shape of (n_true_boxes, 4)
        detected_labels : array, shape of (n_detected_boxes,)
        true_labels :
    """
    n_true_positives = 0
 
    matcher = BoxMatcher(detect_boxes, true_boxes, detect_labels, true_labels)
    for i in range(len(detect_boxes)):
        matching_idx, iou = matcher.match_idx_of_box1_idx(i)
        print("detect_idx: {}, true_idx: {}, matching-score: {}".format(i, matching_idx, iou))
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
    

def test_count_true_positives_in_case_of_only_matching_box_coord():
    detect_boxes = np.array([(100, 100, 200, 200), (105, 105, 210, 210), (120, 120, 290, 290)])
    true_boxes = np.array([(90, 90, 200, 200), (140, 140, 300, 300)])
    n_true_positives = count_true_positives(detect_boxes, true_boxes)
    print(n_true_positives)
    assert n_true_positives == 2

def test_count_true_positives_in_case_of_existing_labels():
    detect_boxes = np.array([(100, 100, 200, 200), (105, 105, 210, 210), (120, 120, 290, 290)])
    true_boxes = np.array([(90, 90, 200, 200), (140, 140, 300, 300)])
    detect_labels = np.array(["raccoon", "raccoon", "human"])
    true_labels = np.array(["human", "raccoon"])
    
    n_true_positives = count_true_positives(detect_boxes, true_boxes, detect_labels, true_labels)
    assert n_true_positives == 0


if __name__ == "__main__":
    
    test_count_true_positives_in_case_of_only_matching_box_coord()
    test_count_true_positives_in_case_of_existing_labels()
    
