
import numpy as np
np.random.seed(1337)
import yolo.augment as augment
from keras.utils import Sequence
from yolo.box import BoundBox, bbox_iou, to_cxcy_wh


class GeneratorConfig(object):
    
    def __init__(self,
                 input_size,
                 grid_size,
                 labels,
                 batch_size,
                 max_box_per_image,
                 anchors):
        self.input_size = input_size
        self.grid_size = grid_size
        self.nb_box = int(len(anchors)/2)
        self.labels = labels
        self.anchors = anchors
        self.batch_size = batch_size
        self.max_box_per_image = max_box_per_image
        self.n_classes = len(self.labels)


class LabelBatchGenerator(object):
    
    def __init__(self, config):
        self.config = config
        self.anchors = self._create_anchor_boxes(config.anchors)

    def _create_anchor_boxes(self, anchors):
        n_anchor_boxes = int(len(anchors)/2)
        return [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) 
                for i in range(n_anchor_boxes)]

    def _normalize_box(self, cx, cy, w, h):
        """Noramlize box : 1-grid == 1.0"""
        norm_cx = cx / (float(self.config.input_size) / self.config.grid_size)
        norm_cy = cy / (float(self.config.input_size) / self.config.grid_size)
        norm_w = w / (float(self.config.input_size) / self.config.grid_size) # unit: grid cell
        norm_h = h / (float(self.config.input_size) / self.config.grid_size) # unit: grid cell
        return norm_cx, norm_cy, norm_w, norm_h
    
    def _is_valid_obj(self, x1, y1, x2, y2, label, grid_x, grid_y):
        """
        # Args
            x1, y1, x2, y2 : int
            label : str
            grid_x, grid_y : float
        """
        is_valid = False
        if x2 > x1 and y2 > y1:
            if label in self.config.labels:
                if grid_x < self.config.grid_size and grid_y < self.config.grid_size:
                    is_valid = True
        return is_valid

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
    
    def _generate_y(self, grid_x, grid_y, best_anchor, obj_indx, box):
        y = np.zeros((self.config.grid_size,  self.config.grid_size, self.config.nb_box, 4+1+self.config.n_classes))
        y[grid_y, grid_x, best_anchor, 0:4] = box
        y[grid_y, grid_x, best_anchor, 4  ] = 1.
        y[grid_y, grid_x, best_anchor, 5+obj_indx] = 1
        return y
    
    def generate(self, boxes, labels):
        
        # construct output from object's x, y, w, h
        true_box_index = 0
        
        y = np.zeros((self.config.grid_size,
                      self.config.grid_size,
                      self.config.nb_box,
                      4+1+self.config.n_classes))
        b_ = np.zeros((1,1,1,
                       self.config.max_box_per_image,
                       4))
        
        # loop over objects in one image
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cx, cy, w, h = to_cxcy_wh(x1, y1, x2, y2)
            norm_box = self._normalize_box(cx, cy, w, h)
            
            grid_x = int(np.floor(norm_box[0]))
            grid_y = int(np.floor(norm_box[1]))

            if self._is_valid_obj(x1, y1, x2, y2, label, grid_x, grid_y):
                obj_indx  = self.config.labels.index(label)
                best_anchor = self._get_anchor_idx(norm_box)

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                y += self._generate_y(grid_x, grid_y, best_anchor, obj_indx, norm_box)
                
                # assign the true box to b_batch
                b_[0, 0, 0, true_box_index] = norm_box
                
                true_box_index += 1
                true_box_index = true_box_index % self.config.max_box_per_image
        return y, b_



class BatchGenerator(Sequence):
    def __init__(self, annotations, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        """
        # Args
            annotations : Annotations instance
        
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
        self.annotations = annotations
        self.generator = None
        self.batch_size = config.batch_size
        self._label_generator = LabelBatchGenerator(config)

        self.config = config
        self.jitter  = jitter
        if norm is None:
            self.norm = lambda x: x
        else:
            self.norm = norm
        self.counter = 0

    def __len__(self):
        return len(self.annotations._components)

    def __getitem__(self, idx):
        """
        # Args
            idx : int
                batch index
        """
        # batch_size = self._ann_handler.get_batch_size(idx)
        instance_count = 0

        x_batch = np.zeros((self.batch_size, self.config.input_size, self.config.input_size, 3))                         # input images
        b_batch = np.zeros((self.batch_size, 1     , 1     , 1    ,  self.config.max_box_per_image, 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((self.batch_size, self.config.grid_size,  self.config.grid_size, self.config.nb_box, 4+1+self.config.n_classes))                # desired network output

        for i in range(self.batch_size):
            # 1. get input file & its annotation
            fname = self.annotations.get_fname(self.batch_size*idx + i)
            boxes = self.annotations.get_boxes(self.batch_size*idx + i)
            labels = self.annotations.get_labels(self.batch_size*idx + i)
            
            # 2. read image in fixed size
            img, boxes = augment.imread(fname,
                                        boxes.as_minmax(),
                                        self.config.input_size,
                                        self.config.input_size,
                                        self.jitter)
            
            # 3. generate x_batch
            x_batch[instance_count] = self.norm(img)
            
            # 4. generate y_batch, b_batch
            y_batch[instance_count], b_batch[instance_count] = self._label_generator.generate(boxes, labels)
            instance_count += 1

        self.counter += 1
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        self.annotations.shuffle()
        self.counter = 0


import pytest
@pytest.fixture(scope='function')
def setup():
    import json
    from yolo.annotation import parse_annotation
    with open("config.json") as config_buffer:    
        config = json.loads(config_buffer.read())
        
    generator_config = GeneratorConfig(config["model"]["input_size"],
                                       int(config["model"]["input_size"]/32),
                                       labels = config["model"]["labels"],
                                       batch_size = 8,
                                       max_box_per_image = config["model"]["max_box_per_image"],
                                       anchors = config["model"]["anchors"])

    train_annotations, train_labels = parse_annotation(config['train']['train_annot_folder'], 
                                                config['train']['train_image_folder'], 
                                                config['model']['labels'])
    return train_annotations, generator_config

@pytest.fixture(scope='function')
def expected():
    x_batch_gt = np.load("x_batch_gt.npy")
    b_batch_gt = np.load("b_batch_gt.npy")
    y_batch_gt = np.load("y_batch_gt.npy")
    return x_batch_gt, b_batch_gt, y_batch_gt

def test_generate_batch(setup, expected):
    train_annotations, config = setup
    x_batch_gt, b_batch_gt, y_batch_gt = expected

    batch_gen = BatchGenerator(train_annotations, config, False, False)
    
    # (8, 416, 416, 3) (8, 1, 1, 1, 10, 4) (8, 13, 13, 5, 6)
    (x_batch, b_batch), y_batch = batch_gen[0]
    
    assert np.array_equal(x_batch, x_batch_gt) == True 
    assert np.array_equal(b_batch, b_batch_gt) == True 
    assert np.array_equal(y_batch, y_batch_gt) == True 


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

        
