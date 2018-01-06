
import numpy as np
np.random.seed(1337)
from keras.utils import Sequence
from .utils.augment import ImgAugment
from .utils.box import to_centroid, create_anchor_boxes, find_match_box
        

def create_batch_generator(annotations, 
                           input_size=416,
                           grid_size=13,
                           batch_size=8,
                           anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                           repeat_times=1,
                           jitter=True, 
                           norm=None):
    """
    # Args
        annotations : Annotations instance in utils.annotataion module
    
    # Return 
        worker : BatchGenerator instance
    """

    img_aug = ImgAugment(input_size, input_size, jitter)
    yolo_box = _YoloBox(input_size, grid_size)
    netin_gen = _NetinGen(input_size, norm)
    netout_gen = _NetoutGen(grid_size, annotations.n_classes(), anchors)
    worker = BatchGenerator(netin_gen,
                            netout_gen,
                            yolo_box,
                            img_aug,
                            annotations,
                            batch_size,
                            repeat_times)
    return worker


class BatchGenerator(Sequence):
    def __init__(self,
                 netin_gen,
                 netout_gen,
                 yolo_box,
                 img_aug,
                 annotations,
                 batch_size,
                 repeat_times):
        """
        # Args
            annotations : Annotations instance
        
        """
        self._netin_gen = netin_gen
        self._netout_gen = netout_gen
        self._img_aug = img_aug
        self._yolo_box = yolo_box

        self._batch_size = min(batch_size, len(annotations)*repeat_times)
        self._repeat_times = repeat_times
        self.annotations = annotations
        self.counter = 0

    def __len__(self):
        return int(len(self.annotations) * self._repeat_times /self._batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        """
        x_batch = []
        y_batch= []
        for i in range(self._batch_size):
            # 1. get input file & its annotation
            fname = self.annotations.fname(self._batch_size*idx + i)
            boxes = self.annotations.boxes(self._batch_size*idx + i)
            labels = self.annotations.code_labels(self._batch_size*idx + i)
            
            # 2. read image in fixed size
            img, boxes = self._img_aug.imread(fname, boxes)

            # 3. grid scaling centroid boxes
            norm_boxes = self._yolo_box.trans(boxes)
            
            # 4. generate x_batch
            x_batch.append(self._netin_gen.run(img))
            y_batch.append(self._netout_gen.run(norm_boxes, labels))

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        self.counter += 1
        return x_batch, y_batch

    def on_epoch_end(self):
        self.annotations.shuffle()
        self.counter = 0


class _YoloBox(object):
    
    def __init__(self, input_size, grid_size):
        self._input_size = input_size
        self._grid_size = grid_size

    def trans(self, boxes):
        """
        # Args
            boxes : array, shape of (N, 4)
                (x1, y1, x2, y2)-ordered & input image size scale coordinate
        
        # Returns
            norm_boxes : array, same shape of boxes
                (cx, cy, w, h)-ordered & rescaled to grid-size
        """
        # 1. minimax box -> centroid box
        centroid_boxes = to_centroid(boxes).astype(np.float32)
        # 2. image scale -> grid scale
        norm_boxes = centroid_boxes * (self._grid_size / self._input_size)
        return norm_boxes


class _NetinGen(object):
    def __init__(self, input_size, norm):
        self._input_size = input_size
        self._norm = self._set_norm(norm)
    
    def run(self, image):
        return self._norm(image)
    
    def _set_norm(self, norm):
        if norm is None:
            return lambda x: x
        else:
            return norm


class _NetoutGen(object):
    def __init__(self,
                 grid_size,
                 nb_classes,
                 anchors=[0.57273, 0.677385,
                          1.87446, 2.06253,
                          3.33843, 5.47434,
                          7.88282, 3.52778,
                          9.77052, 9.16828]):
        self._anchors = create_anchor_boxes(anchors)
        self._tensor_shape = self._set_tensor_shape(grid_size, nb_classes)

    def run(self, norm_boxes, labels):
        """
        # Args
            norm_boxes : array, shape of (N, 4)
                scale normalized boxes
            labels : list of integers
            y_shape : tuple (grid_size, grid_size, nb_boxes, 4+1+nb_classes)
        """
        y = np.zeros(self._tensor_shape)
        
        # loop over objects in one image
        for norm_box, label in zip(norm_boxes, labels):
            best_anchor = self._find_anchor_idx(norm_box)

            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            y += self._generate_y(best_anchor, label, norm_box)
        return y

    def _set_tensor_shape(self, grid_size, nb_classes):
        nb_boxes = len(self._anchors)
        return (grid_size, grid_size, nb_boxes, 4+1+nb_classes)

    def _find_anchor_idx(self, norm_box):
        _, _, center_w, center_h = norm_box
        shifted_box = np.array([0, 0, center_w, center_h])
        return find_match_box(shifted_box, self._anchors)
    
    def _generate_y(self, best_anchor, obj_indx, box):
        y = np.zeros(self._tensor_shape)
        grid_x, grid_y, _, _ = box.astype(int)
        y[grid_y, grid_x, best_anchor, 0:4] = box
        y[grid_y, grid_x, best_anchor, 4  ] = 1.
        y[grid_y, grid_x, best_anchor, 5+obj_indx] = 1
        return y
