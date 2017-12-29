# -*- coding: utf-8 -*-

import os
import numpy as np
from xml.etree.ElementTree import parse


def get_train_annotations(labels,
                          img_folder,
                          ann_folder,
                          valid_img_folder = "",
                          valid_ann_folder = ""):
    """
    # Args
        labels : list of strings
            ["raccoon", "human", ...]
        img_folder : str
        ann_folder : str
        valid_img_folder : str
        valid_ann_folder : str

    # Returns
        train_anns : Annotations instance
        valid_anns : Annotations instance
    """
    # parse annotations of the training set
    train_anns, _ = parse_annotation(ann_folder,
                                     img_folder,
                                     labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_ann_folder):
        valid_anns, _ = parse_annotation(valid_ann_folder,
                                         valid_img_folder,
                                         labels)
    else:
        train_valid_split = int(0.8*len(train_anns))
        train_anns.shuffle()
        
        valid_anns = train_anns[train_valid_split:]
        train_anns = train_anns[:train_valid_split]
    
    return train_anns, valid_anns


class PascalVocXmlParser(object):
    """Parse annotation for 1-annotation file """
    
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            filename : str
        """
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            width : int
        """
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            height : int
        """
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            labels : list of strs
        """

        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels
    
    def get_boxes(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            bbs : 2d-array, shape of (N, 4)
                (x1, y1, x2, y2)-ordered
        """
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([int(x1), int(y1), int(x2), int(y2)])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree

def parse_annotation(ann_dir, img_dir, labels_naming=[]):
    """
    # Args
        ann_dir : str
        img_dir : str
        labels_naming : list of strings
    
    # Returns
        all_imgs : list of dict
        seen_labels : dict
            {'raccoon': 1, ...}
    """
    seen_labels = {}
    
    parser = PascalVocXmlParser()
    
    annotations = Annotations(labels_naming)
    for ann in sorted(os.listdir(ann_dir)):
        annotation_file = os.path.join(ann_dir, ann)
        fname = parser.get_fname(annotation_file)

        ##################################################################################
        annotation = Annotation(os.path.join(img_dir, fname))
        ##################################################################################

        labels = parser.get_labels(annotation_file)
        boxes = parser.get_boxes(annotation_file)
        
        objects = []
        for label, box in zip(labels, boxes):
            x1, y1, x2, y2 = box
            objects.append({'name': label, 'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2})

            ##################################################################################
            annotation.add_object(x1, y1, x2, y2, name=label)
            ##################################################################################
            
            if label in seen_labels:
                seen_labels[label] += 1
            else:
                seen_labels[label] = 1

        ##################################################################################
        annotations.add(annotation)
        ##################################################################################
                        
    return annotations, seen_labels
            

class Annotation(object):
    """
    # Attributes
        fname : image file path
        labels : list of strings
        boxes : Boxes instance
    """
    def __init__(self, filename):
        self.fname = filename
        self.labels = []
        self.boxes = None

    def add_object(self, x1, y1, x2, y2, name):
        self.labels.append(name)
        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2]).reshape(-1,4)
        else:
            box = np.array([x1, y1, x2, y2]).reshape(-1,4)
            self.boxes = np.concatenate([self.boxes, box])

class Annotations(object):
    def __init__(self, label_namings):
        self._components = []
        self._label_namings = label_namings

    def n_classes(self):
        return len(self._label_namings)

    def add(self, annotation):
        self._components.append(annotation)

    def shuffle(self):
        np.random.shuffle(self._components)
    
    def fname(self, i):
        index = self._valid_index(i)
        return self._components[index].fname
    
    def boxes(self, i):
        index = self._valid_index(i)
        return self._components[index].boxes

    def labels(self, i):
        """
        # Returns
            labels : list of strings
        """
        index = self._valid_index(i)
        return self._components[index].labels

    def code_labels(self, i):
        """
        # Returns
            code_labels : list of int
        """
        str_labels = self.labels(i)
        labels = []
        for label in str_labels:
            labels.append(self._label_namings.index(label))
        return labels

    def _valid_index(self, i):
        if i >= len(self._components):
            valid_index = i - len(self._components)
        else:
            valid_index = i
        return valid_index

    def __len__(self):
        return len(self._components)

    def __getitem__(self, idx):
        return self._components[idx]

