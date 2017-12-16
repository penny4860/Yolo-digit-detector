
import cv2
import numpy as np
np.random.seed(1337)
import yolo.augment as augment
from keras.utils import Sequence
from yolo.box import BoundBox, bbox_iou, to_cxcy_wh

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

    def _generate_x(self, img, boxes, labels):
        # assign input image to x_batch
        if self.norm != None: 
            x = self.norm(img)
        else:
            # plot image and bounding boxes for sanity check
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(img[:,:,::-1], (x1, y1), (x2, y2), (255,0,0), 3)
                    cv2.putText(img[:,:,::-1], label, 
                                (x1+2, y1+12), 
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
    
    def _normalize_box(self, cx, cy, w, h):
        """Noramlize box : 1-grid == 1.0"""
        norm_cx = cx / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
        norm_cy = cy / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
        norm_w = w / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
        norm_h = h / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
        return norm_cx, norm_cy, norm_w, norm_h

    
    def _generate_ann_batch(self, boxes, labels):
        
        # construct output from object's x, y, w, h
        true_box_index = 0
        y = np.zeros((13,13,5,6))
        b_ = np.zeros((1,1,1,10,4))
        
        # loop over objects in one image
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cx, cy, w, h = to_cxcy_wh(x1, y1, x2, y2)
            norm_box = self._normalize_box(cx, cy, w, h)
            
            grid_x = int(np.floor(norm_box[0]))
            grid_y = int(np.floor(norm_box[1]))

            if self._is_valid_obj(x1, y1, x2, y2, label, grid_x, grid_y):
                obj_indx  = self.config['LABELS'].index(label)
                best_anchor = self._get_anchor_idx(norm_box)

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                y = self._generate_y(grid_x, grid_y, best_anchor, obj_indx, norm_box)
                
                # assign the true box to b_batch
                b_[0, 0, 0, true_box_index] = norm_box
                
                true_box_index += 1
                true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
        return y, b_

    def __getitem__(self, idx):
        
        anns = self._get_annotations_batch(idx)
        batch_size = len(anns)

        instance_count = 0

        x_batch = np.zeros((batch_size, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((batch_size, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((batch_size, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))                # desired network output

        # loop over batch
        for annotation in anns:
            
            # Todo : annotation handling logic 
            boxes = []
            labels = []
            for obj in annotation["object"]:
                x1, y1, x2, y2 = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
                label = obj["name"]
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
            boxes = np.array(boxes)
            
            # augment input image and fix object's position and size
            img, boxes = augment.imread(annotation["filename"],
                                          boxes,
                                          self.config["IMAGE_W"],
                                          self.config["IMAGE_H"],
                                          self.jitter)
            
            x_batch[instance_count] = self._generate_x(img, boxes, labels)
            y_batch[instance_count], b_batch[instance_count] = self._generate_ann_batch(boxes, labels)
            instance_count += 1

        self.counter += 1
        #print ' new batch created', self.counter
        
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
        self.counter = 0


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

        
