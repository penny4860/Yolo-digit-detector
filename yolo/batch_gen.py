
import numpy as np
np.random.seed(1337)
import yolo.augment as augment
from keras.utils import Sequence
from yolo.box import to_centroid, to_normalize, create_anchor_boxes, find_match_box


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
        self.batch_size = batch_size
        
        self._input_size = input_size
        self._grid_scaling_factor = float(input_size) / grid_size
        self._netin_gen = _NetinGen(input_size, norm)
        self._netout_gen = _NetoutGen(grid_size, annotations.n_classes(), anchors)
        self._true_box_gen = _TrueBoxGen(max_box_per_image)

        self.jitter  = jitter
        self.counter = 0

    def __len__(self):
        return len(self.annotations._components)

    def __getitem__(self, idx):
        """
        # Args
            idx : int
                batch index
        """
        x_batch = []
        y_batch= []
        true_box_batch = []
        for i in range(self.batch_size):
            # 1. get input file & its annotation
            fname = self.annotations.fname(self.batch_size*idx + i)
            boxes = self.annotations.boxes(self.batch_size*idx + i)
            labels = self.annotations.code_labels(self.batch_size*idx + i)
            
            # 2. read image in fixed size
            img, boxes = augment.imread(fname,
                                        boxes,
                                        self._input_size,
                                        self._input_size,
                                        self.jitter)
            # 3. grid scaling centroid boxes
            norm_boxes = self._centroid_grid_scale_box(boxes)
            
            # 4. generate x_batch
            x_batch.append(self._netin_gen.run(img))
            y_batch.append(self._netout_gen.run(norm_boxes, labels))
            true_box_batch.append(self._true_box_gen.run(norm_boxes))

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        true_box_batch = np.array(true_box_batch)
        self.counter += 1
        return [x_batch, true_box_batch], y_batch

    def _centroid_grid_scale_box(self, boxes):
        """
        # Args
            boxes : array, shape of (N, 4)
                (x1, y1, x2, y2)-ordered & input image size scale coordinate
        
        # Returns
            norm_boxes : array, same shape of boxes
                (cx, cy, w, h)-ordered & rescaled to grid-size
        """
        centroid_boxes = to_centroid(boxes)
        norm_boxes = to_normalize(centroid_boxes, self._grid_scaling_factor)
        return norm_boxes

    def on_epoch_end(self):
        self.annotations.shuffle()
        self.counter = 0


class _NetinGen(object):
    def __init__(self, input_size, norm):
        self._input_size = input_size
        self._norm = self._set_norm(norm)
    
    def run(self, image):
        return self._norm(image)
    
    def get_tensor_shape(self):
        return (self._input_size, self._input_size, 3)

    def _set_norm(self, norm):
        if norm is None:
            return lambda x: x
        else:
            return norm


class _TrueBoxGen(object):
    def __init__(self, max_box_per_image):
        self._max_box_per_image = max_box_per_image

    def get_tensor_shape(self):
        return (1, 1, 1, self._max_box_per_image, 4)

    def run(self, norm_boxes):
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
        true_boxes = np.zeros(self.get_tensor_shape())
        
        # loop over objects in one image
        for norm_box in norm_boxes:
            # assign the true box to b_batch
            true_boxes[:,:,:,true_box_index, :] = norm_box
            true_box_index += 1
            true_box_index = true_box_index % self._max_box_per_image
        return true_boxes


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

    def get_tensor_shape(self):
        return self._tensor_shape

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
