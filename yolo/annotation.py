# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET


import numpy as np
from xml.etree.ElementTree import parse


class PascalVocXmlParser(object):
    
    def __init__(self):
        pass
    
    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root
    
    def get_fname(self, annotation_file):
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_labels(self, annotation_file):
        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels
    
    def get_boxes(self, annotation_file):
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


def parse_annotation(ann_dir, img_dir, labels=[]):
    """
    # Args
        ann_dir : str
        img_dir : str
        labels : list of strs
    
    # Returns
        all_imgs : list of dict
            [{'object': [{'name': 'raccoon', 'xmin': 81, 'ymin': 88, 'xmax': 522, 'ymax': 408}],
      '       filename': 'sample//raccoon_train_imgs/raccoon-1.jpg',
              'width': 650,
              'height': 417}]

        seen_labels : dict
            {'raccoon': 1}
    """
    all_imgs = []
    seen_labels = {}
    
    parser = PascalVocXmlParser()
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        
        tree = ET.parse(ann_dir + ann)
        
        img['filename'] = os.path.join(img_dir, parser.get_fname(ann_dir + ann))
        
        for elem in tree.iter():
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels



import pytest
SAMPLE_DIRECTORY = "..//sample"

@pytest.fixture(scope='function')
def setup_inputs(request):
    annotation_dir = os.path.join(SAMPLE_DIRECTORY, "raccoon_train_annotations//")
    image_dir = os.path.join(SAMPLE_DIRECTORY, "raccoon_train_imgs//")
    return annotation_dir, image_dir

@pytest.fixture(scope='function')
def setup_expected_outputs(request, setup_inputs):
    _, image_dir = setup_inputs
    
    images = [{'object': [{'name': 'raccoon', 'xmin': 81, 'ymin': 88, 'xmax': 522, 'ymax': 408}],
               'filename': os.path.join(image_dir, 'raccoon-1.jpg'),
               'width': 650,
               'height': 417}]
    labels = {'raccoon': 1}
    return images, labels


def test_parse_annotation(setup_inputs, setup_expected_outputs):
    
    # Given
    annotation_dir, image_dir = setup_inputs
    
    # When
    all_imgs, seen_labels = parse_annotation(annotation_dir, image_dir)
    
    images, labels = setup_expected_outputs
    
    assert all_imgs == images
    assert seen_labels == labels
    

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
    
    
    
    
    


