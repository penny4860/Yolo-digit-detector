# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from keras.layers import Input

BOX_IDX_X = 0
BOX_IDX_Y = 1
BOX_IDX_W = 2
BOX_IDX_H = 3
BOX_IDX_CONFIDENCE = 4
BOX_IDX_CLASS_START = 5


class YoloLoss(object):
    
    # Todo : true_boxes 를 model에서 삭제하자.    
    def __init__(self,
                 true_boxes=Input(shape=(1, 1, 1, 10 , 4)),
                 grid_size=13,
                 nb_class=1,
                 anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]):
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
        self.true_boxes = true_boxes
        
        self.coord_scale = 1.0

        # Todo : create method를 따로 만들어서 주입받자.
        self._activator = _Activator(self.anchors)
        self._mask = _Mask(nb_class)


    def custom_loss(self, batch_size, warmup_bs):
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
            conf_mask = self._mask.create_conf_mask(y_true, self.true_boxes, pred_tensor)
            
            """
            Warm-up training
            """
            cell_grid = create_cell_grid(tf.shape(y_pred)[1], batch_size)
            true_box_xy, true_box_wh, coord_mask = warmup(true_box_xy, true_box_wh,
                                                          coord_mask, self.coord_scale,
                                                          cell_grid, warmup_bs, self.nb_box, self.anchors)

            """
            Finalize the loss
            """
            loss = get_loss(coord_mask, conf_mask, class_mask, pred_tensor, true_box_xy, true_box_wh, true_box_conf, true_box_class)
            return loss
        return loss_func

    

def warmup(true_box_xy, true_box_wh, coord_mask, coord_scale, cell_grid, warmup_bs, nb_box, anchors):
    no_boxes_mask = tf.to_float(coord_mask < coord_scale/2.)
    seen = tf.assign_add(tf.Variable(0.), 1.)
    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, warmup_bs), 
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(anchors, [1,1,1,nb_box,2]) * no_boxes_mask, 
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy, 
                                   true_box_wh,
                                   coord_mask])
    return true_box_xy, true_box_wh, coord_mask

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
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_size), [grid_size]), (1, grid_size, grid_size, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 5, 1])
    return cell_grid


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
    
    def create_conf_mask(self, y_true, true_boxes, pred_tensor):
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        pred_box_xy, pred_box_wh = pred_tensor[..., :2], pred_tensor[..., 2:4]
        
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


import pytest
@pytest.fixture(scope='function')
def setup_y_true_tensor(request):
    N_COORD = 4
    n_classes = 4
    y_true = tf.placeholder(tf.float32, [None, 9, 9, 5, N_COORD+1+n_classes], name='y_true')
    y_true_value = np.zeros((1, 9, 9, 5, N_COORD+1+n_classes))
    y_true_value[0,4,3,2,:] = [3.46875, 4.78125, 1, 5.625, 1, 1, 0, 0, 0]           # (cx, cy, w, h, confidence, classes)
    y_true_value[0,4,4,2,:] = [4.484375, 4.875, 1.15625, 5.625, 1, 0, 0, 0, 1]      # (cx, cy, w, h, confidence, classes)
    return y_true, y_true_value

@pytest.fixture(scope='function')
def setup_true_box_tensor(request):
    true_box_class = tf.placeholder(tf.int32, [None, 9, 9, 5], name='y_true')
    true_box_class_value = np.zeros((1, 9, 9, 5))
    true_box_class_value[0,4,3,2] = 0    # class index
    true_box_class_value[0,4,4,2] = 3    # class index
    return true_box_class, true_box_class_value

@pytest.fixture(scope='function')
def setup_y_pred_tensor(request):
    prng = RandomState(1337)
    y_pred_value = prng.randn(1,9,9,5,9) / 4
    y_pred = tf.placeholder(tf.float32, [None, 9, 9, 5, 9], name='y_pred')
    return y_pred, y_pred_value

@pytest.fixture(scope='function')
def setup_true_boxes_tensor(request):
    true_boxes = Input(shape=(1, 1, 1, 10 , 4))
    # true_boxes = tf.placeholder(tf.int32, [None, 1,1,1, 10,4], name='y_true')
    true_boxes_value = np.zeros((1,1,1,1,10,4))
    true_boxes_value[0,0,0,0,0,:] = [3.46875, 4.78125, 1, 5.625]
    true_boxes_value[0,0,0,0,1,:] = [4.484375, 4.875, 1.15625, 5.625]
    return true_boxes, true_boxes_value

def run_op(operation, feed_dict):
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    op_value = sess.run(operation, feed_dict=feed_dict)
    sess.close()
    return op_value

def test_yolo_class_masking(setup_y_true_tensor, setup_true_box_tensor):
    # 1. setup y_true placeholder
    # 2. setup y_true feed value
    y_true, y_true_value = setup_y_true_tensor
    true_box_class, true_box_class_value = setup_true_box_tensor
     
    # 3. create coord_mask operation
    yolo_mask = _Mask(nb_class=4, coord_scale=1.0, class_scale=1.0, object_scale=5.0, no_object_scale=1.0)
    class_mask_op = yolo_mask.create_class_mask(y_true, true_box_class)
 
    # 4. run loss_op in session
    class_mask_value = run_op(class_mask_op, feed_dict={y_true: y_true_value,
                                                        true_box_class : true_box_class_value})
     
    # coordinate mask value : (N, grid, grid, nb_box)
    #     object 가 있는 (grid_x, grid_y, anchor_idx) 에만 1, 나머지는 0
    expected_class_mask = np.zeros((1,9,9,5))
    expected_class_mask[0, 4, 3, 2] = 1.0
    expected_class_mask[0, 4, 4, 2] = 1.0
    assert np.allclose(class_mask_value, expected_class_mask)

def test_yolo_coord_masking(setup_y_true_tensor):
    # 1. setup y_true placeholder
    # 2. setup y_true feed value
    y_true, y_true_value = setup_y_true_tensor
     
    # 3. create coord_mask operation
    yolo_mask = _Mask(nb_class=4, coord_scale=1.0, class_scale=1.0, object_scale=5.0, no_object_scale=1.0)
    coord_mask_op = yolo_mask.create_coord_mask(y_true)
 
    # 4. run loss_op in session
    coord_mask_value = run_op(coord_mask_op, feed_dict={y_true: y_true_value})
     
    # coordinate mask value : (N, grid, grid, nb_box, 1)
    #     object 가 있는 (grid_x, grid_y, anchor_idx) 에만 1, 나머지는 0
    expected_coord_mask = np.zeros((1,9,9,5,1))
    expected_coord_mask[0, 4, 3, 2, :] = 1.0
    expected_coord_mask[0, 4, 4, 2, :] = 1.0
    assert np.allclose(coord_mask_value, expected_coord_mask)

def test_yolo_conf_masking(setup_y_true_tensor, setup_true_boxes_tensor, setup_y_pred_tensor):
    y_true, y_true_value = setup_y_true_tensor
    true_boxes, true_boxes_value = setup_true_boxes_tensor
    y_pred, y_pred_value = setup_y_pred_tensor
     
    yolo_mask = _Mask(nb_class=4, coord_scale=1.0, class_scale=1.0, object_scale=5.0, no_object_scale=1.0)
    conf_mask_op = yolo_mask.create_conf_mask(y_true, true_boxes, y_pred)
    conf_mask_value = run_op(conf_mask_op, feed_dict={y_true:y_true_value,
                                                      true_boxes: true_boxes_value,
                                                      y_pred:y_pred_value})
    # (1, 9, 9, 5)
    print("=================================================")
    print(conf_mask_value.shape)
    print(conf_mask_value)
    print("=================================================")
    

def test_loss_op(setup_y_true_tensor, setup_y_pred_tensor, setup_true_boxes_tensor):
    # 1. build loss function
    batch_size = 1
    warmup_bs = 0
    yolo_loss = YoloLoss(grid_size=9, nb_class=4)
    custom_loss = yolo_loss.custom_loss(batch_size, warmup_bs)

    # 2. placeholder : (y_true, y_pred)
    y_true, y_true_value = setup_y_true_tensor
    y_pred, y_pred_value = setup_y_pred_tensor

    # 3. loss operation
    loss_op = custom_loss(y_true, y_pred)
    
    # 4. setup feed values for each placeholders (true_boxes, y_true, y_pred
    _, true_boxes_value = setup_true_boxes_tensor
    
    # 5. run loss_op in session
    # y_true, y_pred에 실제 value를 insert
    loss_value = run_op(loss_op, feed_dict={yolo_loss.true_boxes: true_boxes_value,
                                            y_true: y_true_value,
                                            y_pred: y_pred_value})

    assert np.allclose(loss_value, 5.47542)

def test_loss_op_for_warmup(setup_y_true_tensor, setup_y_pred_tensor):
    # 1. build loss function
    batch_size = 1
    warmup_bs = 100
    yolo_loss = YoloLoss(grid_size=9, nb_class=4)
    custom_loss = yolo_loss.custom_loss(batch_size, warmup_bs)
  
    # 2. placeholder : (y_true, y_pred)
    y_true, y_true_value = setup_y_true_tensor
    y_pred, y_pred_value = setup_y_pred_tensor
  
    # 3. loss operation
    loss_op = custom_loss(y_true, y_pred)
      
    # 4. setup feed values for each placeholders (true_boxes, y_true, y_pred
    true_boxes_value = np.zeros((1,1,1,1,10,4))
    true_boxes_value[0,0,0,0,0,:] = [3.46875, 4.78125, 1, 5.625]
    true_boxes_value[0,0,0,0,1,:] = [4.484375, 4.875, 1.15625, 5.625]
      
    # 5. run loss_op in session
    # y_true, y_pred에 실제 value를 insert
    loss_value = run_op(loss_op, feed_dict={yolo_loss.true_boxes: true_boxes_value,
                                            y_true: y_true_value,
                                            y_pred: y_pred_value})
    assert np.allclose(loss_value, 3.1930623)

def test_y_tensor_activation(setup_y_true_tensor):
    y_pred = tf.placeholder(tf.float32, [None, 9, 9, 5, 9], name='y_pred')
    y_true, y_true_value = setup_y_true_tensor
    
    activator = _Activator()
    pred_tensor, true_tensor = activator.run(y_true, y_pred)
    

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
    
