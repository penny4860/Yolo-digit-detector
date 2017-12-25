
import numpy as np
np.random.seed(1337)
import yolo.augment as augment
from keras.utils import Sequence
from yolo.box import to_centroid, to_normalize, centroid_box_iou


def create_anchor_boxes(anchors):
    """
    # Args
        anchors : list of floats
    # Returns
        boxes : array, shape of (len(anchors), 4)
            centroid-type
    """
    boxes = []
    n_boxes = int(len(anchors)/2)
    for i in range(n_boxes):
        boxes.append(np.array([0, 0, anchors[2*i], anchors[2*i+1]]))
    return np.array(boxes)

class LabelBatchGenerator(object):
    
    def __init__(self, anchors):
        self.anchors = create_anchor_boxes(anchors)

    def _get_anchor_idx(self, norm_box):
        _, _, center_w, center_h = norm_box
        
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou     = -1
        
        shifted_box = np.array([0, 0, center_w, center_h])
        
        for i, anchor_box in enumerate(self.anchors):
            iou = centroid_box_iou(shifted_box, anchor_box)
            
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
        return best_anchor
    
    def _generate_y(self, best_anchor, obj_indx, box, y_shape):
        y = np.zeros(y_shape)
        grid_x, grid_y, _, _ = box.astype(int)
        y[grid_y, grid_x, best_anchor, 0:4] = box
        y[grid_y, grid_x, best_anchor, 4  ] = 1.
        y[grid_y, grid_x, best_anchor, 5+obj_indx] = 1
        return y
    
    def generate(self, norm_boxes, labels, y_shape, b_shape):
        """
        # Args
            labels : list of integers
            
            y_shape : tuple
                (grid_size, grid_size, nb_boxes, 4+1+nb_classes)
            b_shape : tuple
                (1, 1, 1, max_box_per_image, 4)
        """
        
        # construct output from object's x, y, w, h
        true_box_index = 0
        
        y = np.zeros(y_shape)
        b_ = np.zeros(b_shape)
        
        # loop over objects in one image
        for norm_box, label in zip(norm_boxes, labels):
            best_anchor = self._get_anchor_idx(norm_box)

            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            y += self._generate_y(best_anchor, label, norm_box, y_shape)
            
            # assign the true box to b_batch
            b_[0, 0, 0, true_box_index] = norm_box
            
            max_box_per_image = b_.shape[-1]
            true_box_index += 1
            true_box_index = true_box_index % max_box_per_image
        return y, b_


class BatchGenerator(Sequence):
    def __init__(self, annotations, 
                       input_size=416,
                       grid_size=13,
                       batch_size=8,
                       max_box_per_image=10,
                       anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828], 
                       jitter=True, 
                       norm=None):
        """
        # Args
            annotations : Annotations instance
        
        """
        self.annotations = annotations
        self.input_size = input_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.max_box_per_image = max_box_per_image
        self.n_classes = annotations.n_classes()
        self.nb_box = int(len(anchors) / 2)
        self._label_generator = LabelBatchGenerator(anchors)

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
        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, 3))                         # input images
        b_batch = np.zeros((self.batch_size, 1     , 1     , 1    ,  self.max_box_per_image, 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((self.batch_size, self.grid_size,  self.grid_size, self.nb_box, 4+1+self.n_classes))                # desired network output

        for i in range(self.batch_size):
            # 1. get input file & its annotation
            fname = self.annotations.fname(self.batch_size*idx + i)
            boxes = self.annotations.boxes(self.batch_size*idx + i)
            labels = self.annotations.code_labels(self.batch_size*idx + i)
            
            # 2. read image in fixed size
            img, boxes = augment.imread(fname,
                                        boxes,
                                        self.input_size,
                                        self.input_size,
                                        self.jitter)
            norm_boxes = self._centroid_scale_box(boxes)
            
            # 3. generate x_batch
            x_batch[i] = self.norm(img)
            y_batch[i], b_batch[i] = self._label_generator.generate(norm_boxes,
                                                                    labels,
                                                                    y_batch.shape[1:],
                                                                    b_batch.shape[1:])

        self.counter += 1
        return [x_batch, b_batch], y_batch

    def _centroid_scale_box(self, boxes):
        centroid_boxes = to_centroid(boxes)
        norm_boxes = to_normalize(centroid_boxes, self.input_size/self.grid_size)
        return norm_boxes

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
        
    input_size = config["model"]["input_size"]
    grid_size = int(input_size/32)
    batch_size = 8
    max_box_per_image = config["model"]["max_box_per_image"]
    anchors = config["model"]["anchors"]

    train_annotations, _ = parse_annotation(config['train']['train_annot_folder'], 
                                                       config['train']['train_image_folder'], 
                                                       config['model']['labels'])
    return train_annotations, input_size, grid_size, batch_size, max_box_per_image, anchors

@pytest.fixture(scope='function')
def expected():
    x_batch_gt = np.load("x_batch_gt.npy")
    b_batch_gt = np.load("b_batch_gt.npy")
    y_batch_gt = np.load("y_batch_gt.npy")
    return x_batch_gt, b_batch_gt, y_batch_gt

def test_generate_batch(setup, expected):
    train_annotations, input_size, grid_size, batch_size, max_box_per_image, anchors = setup
    x_batch_gt, b_batch_gt, y_batch_gt = expected

    batch_gen = BatchGenerator(train_annotations,
                               input_size,
                               grid_size,
                               batch_size,
                               max_box_per_image,
                               anchors,
                               jitter=False)
    
    # (8, 416, 416, 3) (8, 1, 1, 1, 10, 4) (8, 13, 13, 5, 6)
    (x_batch, b_batch), y_batch = batch_gen[0]
    
    assert np.array_equal(x_batch, x_batch_gt) == True 
    assert np.array_equal(b_batch, b_batch_gt) == True 
    assert np.array_equal(y_batch, y_batch_gt) == True 


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

        
