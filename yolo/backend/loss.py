# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

BOX_IDX_X = 0
BOX_IDX_Y = 1
BOX_IDX_W = 2
BOX_IDX_H = 3
BOX_IDX_CONFIDENCE = 4
BOX_IDX_CLASS_START = 5


class YoloLoss(object):
    
    def __init__(self,
                 grid_size=13,
                 nb_class=1,
                 anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                 coord_scale=1.0,
                 class_scale=1.0,
                 object_scale=5.0,
                 no_object_scale=1.0):
        """
        # Args
            grid_size : int
            batch_size : int
            anchors : list of floats
            nb_box : int
            nb_class : int
            true_boxes : Tensor instance
        """
        self.grid_size = grid_size
        self.anchors = anchors
        self.nb_box = int(len(anchors)/2)
        self.nb_class = nb_class

        self.coord_scale = coord_scale

        # Todo : create method를 따로 만들어서 주입받자.
        self._activator = _Activator(self.anchors)
        self._mask = _Mask(nb_class, coord_scale, class_scale, object_scale, no_object_scale)


    def custom_loss(self, batch_size):
        """
        # Args
            y_true : (N, 13, 13, 5, 6)
            y_pred : (N, 13, 13, 5, 6)
        
        """
        def loss_func(y_true, y_pred):
            # 1. activate prediction & truth tensor
            true_tensor, pred_tensor = self._activator.run(y_true, y_pred)
            true_box_xy, true_box_wh, true_box_conf, true_box_class = true_tensor[..., :2], true_tensor[..., 2:4], true_tensor[..., 4], true_tensor[..., 5]
            true_box_class = tf.cast(true_box_class, tf.int64)

            # 2. mask
            coord_mask = self._mask.create_coord_mask(y_true)
            class_mask = self._mask.create_class_mask(y_true, true_box_class)
            conf_mask = self._mask.create_conf_mask(y_true, pred_tensor, batch_size)
            
            """
            Finalize the loss
            """
            loss = get_loss(coord_mask, conf_mask, class_mask, pred_tensor, true_box_xy, true_box_wh, true_box_conf, true_box_class)
            return loss
        return loss_func
    

def get_loss(coord_mask, conf_mask, class_mask, pred_tensor, true_box_xy, true_box_wh, true_box_conf, true_box_class):
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = pred_tensor[..., :2], pred_tensor[..., 2:4], pred_tensor[..., 4], pred_tensor[..., 5:]
    # true_box_xy, true_box_wh, true_box_conf, true_box_class = true_tensor[..., :2], true_tensor[..., 2:4], true_tensor[..., 4], true_tensor[..., 5]
    true_box_class = tf.cast(true_box_class, tf.int64)
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    loss = loss_xy + loss_wh + loss_conf + loss_class
    return loss


class _Activator(object):
    
    def __init__(self, anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]):
        self._anchor_boxes = np.reshape(anchors, [1,1,1,-1,2])
        
    def run(self, y_true, y_pred):
        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = self._activate_pred_tensor(y_pred)
        true_box_xy, true_box_wh, true_box_conf, true_box_class = self._activate_true_tensor(y_true, pred_box_xy, pred_box_wh)

        # concatenate pred tensor
        pred_box_conf = tf.expand_dims(pred_box_conf, -1)
        y_pred_activated = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class], axis=-1)

        # concatenate true tensor
        true_box_conf = tf.expand_dims(true_box_conf, -1)
        true_box_class = tf.expand_dims(true_box_class, -1)
        true_box_class = tf.cast(true_box_class, true_box_xy.dtype)
        y_true_activated = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_class], axis=-1)
        return y_true_activated, y_pred_activated
    
    def _activate_pred_tensor(self, y_pred):
        """
        # Args
            y_pred : (N, 13, 13, 5, 6)
            cell_grid : (N, 13, 13, 5, 2)
        
        # Returns
            box_xy : (N, 13, 13, 5, 2)
                1) sigmoid activation
                2) grid offset added
            box_wh : (N, 13, 13, 5, 2)
                1) exponential activation
                2) anchor box multiplied
            box_conf : (N, 13, 13, 5, 1)
                1) sigmoid activation
            box_classes : (N, 13, 13, 5, nb_class)
        """
        # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy
        batch_size = tf.shape(y_pred)[0]
        grid_size = tf.shape(y_pred)[1]
        cell_grid = create_cell_grid(grid_size, batch_size)
        
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * self._anchor_boxes
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        pred_box_class = y_pred[..., 5:]
        return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

    def _activate_true_tensor(self, y_true, pred_box_xy, pred_box_wh):
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        
        return true_box_xy, true_box_wh, true_box_conf, true_box_class


def create_cell_grid(grid_size, batch_size):
    x_pos = tf.to_float(tf.range(grid_size))
    y_pos = tf.to_float(tf.range(grid_size))
    xx, yy = tf.meshgrid(x_pos, y_pos)
    xx = tf.expand_dims(xx, -1)
    yy = tf.expand_dims(yy, -1)
    
    grid = tf.concat([xx, yy], axis=-1)         # (7, 7, 2)
    grid = tf.expand_dims(grid, -2)             # (7, 7, 1, 2)
    grid = tf.tile(grid, (1,1,5,1))             # (7, 7, 5, 2)
    grid = tf.expand_dims(grid, 0)              # (1, 7, 7, 1, 2)
    grid = tf.tile(grid, (batch_size,1,1,1,1))  # (N, 7, 7, 1, 2)
    return grid


class _Mask(object):
    
    def __init__(self, nb_class=1, coord_scale=1.0, class_scale=1.0, object_scale=5.0, no_object_scale=1.0):
        self._nb_class = nb_class
        self._coord_scale = coord_scale
        self._class_scale = class_scale
        self._object_scale = object_scale
        self._no_object_scale = no_object_scale
        
    def create_coord_mask(self, y_true):
        """ Simply the position of the ground truth boxes (the predictors)

        # Args
            y_true : Tensor, shape of (None, grid, grid, nb_box, 4+1+n_classes)
        
        # Returns
            mask : Tensor, shape of (None, grid, grid, nb_box, 1)
        """
        #     BOX 별 confidence value 를 mask value 로 사용
        # [1 13 13 5 1]
        mask = tf.expand_dims(y_true[..., BOX_IDX_CONFIDENCE], axis=-1) * self._coord_scale
        return mask
    
    def create_class_mask(self, y_true, true_box_class):
        """ Simply the position of the ground truth boxes (the predictors)

        # Args
            y_true : Tensor, shape of (None, grid, grid, nb_box, 4+1+n_classes)
            true_box_class : Tensor, shape of (None, grid, grid, nb_box)
                indicate class index per boxes
        
        # Returns
            mask : Tensor, shape of (None, grid, grid, nb_box)
        """
        class_wt = np.ones(self._nb_class, dtype='float32')
        mask = y_true[..., 4] * tf.gather(class_wt, true_box_class) * self._class_scale
        return mask
    
    def create_conf_mask(self, y_true, pred_tensor, batch_size):
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        pred_box_xy, pred_box_wh = pred_tensor[..., :2], pred_tensor[..., 2:4]
        
        true_boxes = y_true[..., :4]
        true_boxes = tf.reshape(true_boxes, [batch_size, -1, 4])
        true_boxes = tf.expand_dims(true_boxes, 1)
        true_boxes = tf.expand_dims(true_boxes, 1)
        true_boxes = tf.expand_dims(true_boxes, 1)
        
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        # 1) confidence mask (N, 13, 13, 5)
        conf_mask  = tf.zeros(tf.shape(y_true)[:4])
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self._no_object_scale
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self._object_scale
        return conf_mask
    
