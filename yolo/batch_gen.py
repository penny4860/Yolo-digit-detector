
import numpy as np
np.random.seed(1337)
import yolo.augment as augment
from keras.utils import Sequence
from yolo.box import to_centroid, to_normalize, create_anchor_boxes, find_match_box



class _TrueBoxGen(object):
    def __init__(self):
        pass

    def run(self, norm_boxes, max_box_per_image):
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
        true_boxes = np.zeros((max_box_per_image, 4))
        
        # loop over objects in one image
        for norm_box in norm_boxes:
            # assign the true box to b_batch
            true_boxes[true_box_index, :] = norm_box
            true_box_index += 1
            true_box_index = true_box_index % max_box_per_image
        return true_boxes


class _NetoutGen(object):
    def __init__(self, anchors=[0.57273, 0.677385,
                                1.87446, 2.06253,
                                3.33843, 5.47434,
                                7.88282, 3.52778,
                                9.77052, 9.16828]):
        self.anchors = create_anchor_boxes(anchors)

    def run(self, norm_boxes, labels, y_shape):
        """
        # Args
            labels : list of integers
            
            y_shape : tuple
                (grid_size, grid_size, nb_boxes, 4+1+nb_classes)
            b_shape : tuple
                (1, 1, 1, max_box_per_image, 4)
        """
        y = np.zeros(y_shape)
        
        # loop over objects in one image
        for norm_box, label in zip(norm_boxes, labels):
            best_anchor = self._find_anchor_idx(norm_box)

            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            y += self._generate_y(best_anchor, label, norm_box, y_shape)
        return y

    def _find_anchor_idx(self, norm_box):
        _, _, center_w, center_h = norm_box
        shifted_box = np.array([0, 0, center_w, center_h])
        return find_match_box(shifted_box, self.anchors)
    
    def _generate_y(self, best_anchor, obj_indx, box, y_shape):
        y = np.zeros(y_shape)
        grid_x, grid_y, _, _ = box.astype(int)
        y[grid_y, grid_x, best_anchor, 0:4] = box
        y[grid_y, grid_x, best_anchor, 4  ] = 1.
        y[grid_y, grid_x, best_anchor, 5+obj_indx] = 1
        return y


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
        
        self._true_box_gen = _TrueBoxGen()
        self._netout_gen = _NetoutGen(anchors)
        self._norm = self._set_norm(norm)

        self.jitter  = jitter
        self.counter = 0

    def _set_norm(self, norm):
        if norm is None:
            return lambda x: x
        else:
            return norm

    def __len__(self):
        return len(self.annotations._components)

    def __getitem__(self, idx):
        """
        # Args
            idx : int
                batch index
        """
        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, 3))                         # input images
        true_box_batch = np.zeros((self.batch_size, 1     , 1     , 1    ,  self.max_box_per_image, 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
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
            x_batch[i] = self._norm(img)
            y_batch[i] = self._netout_gen.run(norm_boxes, labels, y_batch.shape[1:])
            true_box_batch[i] = self._true_box_gen.run(norm_boxes, self.max_box_per_image)

        self.counter += 1
        return [x_batch, true_box_batch], y_batch

    def _centroid_scale_box(self, boxes):
        centroid_boxes = to_centroid(boxes)
        norm_boxes = to_normalize(centroid_boxes, self.input_size/self.grid_size)
        return norm_boxes

    def on_epoch_end(self):
        self.annotations.shuffle()
        self.counter = 0
