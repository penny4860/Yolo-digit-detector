# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        
        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
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
    
    
    
    
    


