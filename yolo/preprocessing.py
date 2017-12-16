
import cv2
import numpy as np
np.random.seed(1337)
import yolo.augment as augment
from keras.utils import Sequence
from yolo.box import BoundBox, bbox_iou

class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        """
        # Args
            images : list of dictionary including following keys
                "filename"  : str
                "width"     : int
                "height"    : int
                "object"    : list of dictionary
                    'name' : str
                    'xmin' : int
                    'ymin' : int
                    'xmax' : int
                    'ymax' : int
        """
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.counter = 0        
        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])/2))]

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def _get_annotations_batch(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']
            
        return self.images[l_bound:r_bound]

    def _is_valid_obj(self, x1, y1, x2, y2, label, grid_x, grid_y):
        """
        # Args
            x1, y1, x2, y2 : int
            label : str
            grid_x, grid_y : float
        """
        is_valid = False
        if x2 > x1 and y2 > y1:
            if label in self.config['LABELS']:
                if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                    is_valid = True
        return is_valid

    def _generate_x(self, img, all_objs):
        # assign input image to x_batch
        if self.norm != None: 
            x = self.norm(img)
        else:
            # plot image and bounding boxes for sanity check
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(img[:,:,::-1], obj['name'], 
                                (obj['xmin']+2, obj['ymin']+12), 
                                0, 1.2e-3 * img.shape[0], 
                                (0,255,0), 2)
                    
            x = img
        return x
    
    def _generate_y(self, grid_x, grid_y, best_anchor, obj_indx, box):
        y = np.zeros((self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))
        y[grid_y, grid_x, best_anchor, 0:4] = box
        y[grid_y, grid_x, best_anchor, 4  ] = 1.
        y[grid_y, grid_x, best_anchor, 5+obj_indx] = 1
        return y

    def _get_anchor_idx(self, box):
        _, _, center_w, center_h = box
        
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou     = -1
        
        shifted_box = BoundBox(0, 
                               0, 
                               center_w, 
                               center_h)
        
        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou    = bbox_iou(shifted_box, anchor)
            
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
        return best_anchor

    def _generate_box(self, obj):
        center_x = .5*(obj['xmin'] + obj['xmax'])
        center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
        center_y = .5*(obj['ymin'] + obj['ymax'])
        center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))
        
        return [center_x, center_y, center_w, center_h], grid_x, grid_y

    def __getitem__(self, idx):
        
        anns = self._get_annotations_batch(idx)
        batch_size = len(anns)

        instance_count = 0

        x_batch = np.zeros((batch_size, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((batch_size, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((batch_size, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))                # desired network output

        # loop over batch
        for annotation in anns:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(annotation, jitter=self.jitter)
            
            # assign input image to x_batch
            x_batch[instance_count] = self._generate_x(img, all_objs)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            # loop over objects in one image
            for obj in all_objs:
                box, grid_x, grid_y = self._generate_box(obj)

                if self._is_valid_obj(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['name'], grid_x, grid_y):
                    obj_indx  = self.config['LABELS'].index(obj['name'])
                    best_anchor = self._get_anchor_idx(box)

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[instance_count] = self._generate_y(grid_x, grid_y, best_anchor, obj_indx, box)
                    
                    # assign the true box to b_batch
                    b_batch[instance_count, 0, 0, 0, true_box_index] = box
                    
                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # increase instance counter in current batch
            instance_count += 1  

        self.counter += 1
        #print ' new batch created', self.counter
        
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
        self.counter = 0

    def aug_image(self, train_instance, jitter):
        """
        # Args
            train_instance : dict
            jitter : bool
        
        # Returns
            image : 3d-array
            all_objs : list of dicts
                "name"
                "xmin"
                "xmax"
                "ymin"
                "ymax"
        """
        boxes = []
        for obj in train_instance["object"]:
            x1, y1, x2, y2 = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
            boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes)
        
        image, boxes = augment.imread(train_instance["filename"],
                              boxes,
                              self.config["IMAGE_W"],
                              self.config["IMAGE_H"],
                              jitter)
        
        objs = []
        for obj, box in zip(train_instance["object"], boxes):
            obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"] = box.astype(np.int)
            objs.append(obj)
        return image, objs

import pytest
@pytest.fixture(scope='function')
def setup():
    import json
    from yolo.annotation import parse_annotation
    with open("config.json") as config_buffer:    
        config = json.loads(config_buffer.read())
    generator_config = {
            'IMAGE_H'         : config["model"]["input_size"], 
            'IMAGE_W'         : config["model"]["input_size"],
            'GRID_H'          : int(config["model"]["input_size"]/32),  
            'GRID_W'          : int(config["model"]["input_size"]/32),
            'BOX'             : 5,
            'LABELS'          : config["model"]["labels"],
            'CLASS'           : len(config["model"]["labels"]),
            'ANCHORS'         : config["model"]["anchors"],
            'BATCH_SIZE'      : 8,
            'TRUE_BOX_BUFFER' : config["model"]["max_box_per_image"],
    }
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'], 
                                                config['train']['train_image_folder'], 
                                                config['model']['labels'])
    return train_imgs, generator_config

@pytest.fixture(scope='function')
def expected():
    x_batch_gt = np.load("x_batch_gt.npy")
    b_batch_gt = np.load("b_batch_gt.npy")
    y_batch_gt = np.load("y_batch_gt.npy")
    return x_batch_gt, b_batch_gt, y_batch_gt

def test_generate_batch(setup, expected):
    images, config = setup
    x_batch_gt, b_batch_gt, y_batch_gt = expected

    batch_gen = BatchGenerator(images, config, False, False)
    
    # (8, 416, 416, 3) (8, 1, 1, 1, 10, 4) (8, 13, 13, 5, 6)
    (x_batch, b_batch), y_batch = batch_gen[0]
    
    assert np.array_equal(x_batch, x_batch_gt) == True 
    assert np.array_equal(b_batch, b_batch_gt) == True 
    assert np.array_equal(y_batch, y_batch_gt) == True 


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

        
