# -*- coding: utf-8 -*-

import os
import numpy as np
from xml.etree.ElementTree import parse


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
    


def parse_annotation(ann_dir, img_dir, labels=[]):
    """
    # Args
        ann_dir : str
        img_dir : str
    
    # Returns
        all_imgs : list of dict
        seen_labels : dict
            {'raccoon': 1, ...}
    """
    all_imgs = []
    seen_labels = {}
    
    parser = PascalVocXmlParser()
    
    annotations = Annotations()
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        
        annotation_file = os.path.join(ann_dir, ann)
        
        fname = parser.get_fname(annotation_file)
        img['filename'] = os.path.join(img_dir, fname)

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
        
        img['object'] = objects
        
        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return annotations, seen_labels
            

# Todo : parse_annotatiaon의 리턴을 이 객체로하자.
class AnnHandler(object):
    def __init__(self, image_anns, batch_size, shuffle):
        """
        # Args
            image_anns : list of dictionary including following keys
                "filename"  : str
                "object"    : list of dictionary
                    'name' : str
                    'xmin' : int
                    'ymin' : int
                    'xmax' : int
                    'ymax' : int
        """
        self.image_anns = image_anns
        self.batch_size = batch_size
        self.shuffle = shuffle

        if shuffle:
            np.random.shuffle(self.image_anns)

    def get_ann(self, batch_idx, index):
        batch_anns = self._get_batch(batch_idx)
        fname = self._get_fname(batch_anns, index)        
        labels = self._get_labels(batch_anns, index)        
        boxes = self._get_boxes(batch_anns, index)        
        return fname, boxes, labels

    def get_batch_size(self, batch_idx):
        batch_anns = self._get_batch(batch_idx)
        return len(batch_anns)

    def len_batches(self):
        return int(np.ceil(float(len(self.image_anns))/self.batch_size))

    def end_epoch(self):
        if self.shuffle:
            np.random.shuffle(self.image_anns)

    def _get_fname(self, batch_anns, index):
        return batch_anns[index]["filename"]

    def _get_labels(self, batch_anns, index):
        labels = []
        for obj in batch_anns[index]["object"]:
            labels.append(obj["name"])
        return labels

    def _get_boxes(self, batch_anns, index):
        boxes = []
        for obj in batch_anns[index]["object"]:
            x1, y1, x2, y2 = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
            boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes)
        return boxes

    def _get_batch(self, batch_idx):
        l_bound = batch_idx * self.batch_size
        r_bound = (batch_idx+1) * self.batch_size

        if r_bound > len(self.image_anns):
            r_bound = len(self.image_anns)
            l_bound = r_bound - self.batch_size
        return self.image_anns[l_bound:r_bound]


from yolo.box import Box, Boxes
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
            self.boxes = Boxes(minmax = [x1, y1, x2, y2])
        else:
            self.boxes.add(Box(minmax = [x1, y1, x2, y2]))


class Annotations(object):
    def __init__(self, filename=None):
        if filename is None:
            self._components = []
        else:
            ann = Annotation(filename)
            self._components = [ann]

    def add(self, annotation):
        self._components.append(annotation)

    def shuffle(self):
        np.random.shuffle(self._components)
    
    def get_fname(self, i):
        index = self._valid_index(i)
        return self._components[index].fname
    
    def get_boxes(self, i):
        index = self._valid_index(i)
        return self._components[index].boxes

    def get_labels(self, i):
        index = self._valid_index(i)
        return self._components[index].labels

    def _valid_index(self, i):
        if i >= len(self._components):
            valid_index = i - len(self._components)
        else:
            valid_index = i
        return valid_index

