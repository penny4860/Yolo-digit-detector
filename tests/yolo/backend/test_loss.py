# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pytest
from numpy.random import RandomState
from yolo.backend.loss import YoloLoss, _Mask, _Activator


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
    true_boxes = tf.placeholder(tf.float32, [None, 1,1,1, 10,4], name='y_true')
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
    #     object �� �ִ� (grid_x, grid_y, anchor_idx) ���� 1, �������� 0
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
    #     object �� �ִ� (grid_x, grid_y, anchor_idx) ���� 1, �������� 0
    expected_coord_mask = np.zeros((1,9,9,5,1))
    expected_coord_mask[0, 4, 3, 2, :] = 1.0
    expected_coord_mask[0, 4, 4, 2, :] = 1.0
    assert np.allclose(coord_mask_value, expected_coord_mask)

def test_yolo_conf_masking(setup_y_true_tensor, setup_true_boxes_tensor, setup_y_pred_tensor):
    y_true, y_true_value = setup_y_true_tensor
    true_boxes, true_boxes_value = setup_true_boxes_tensor
    y_pred, y_pred_value = setup_y_pred_tensor
     
    yolo_mask = _Mask(nb_class=4, coord_scale=1.0, class_scale=1.0, object_scale=5.0, no_object_scale=1.0)
    conf_mask_op = yolo_mask.create_conf_mask(y_true, y_pred, batch_size=1)
    conf_mask_value = run_op(conf_mask_op, feed_dict={y_true:y_true_value,
                                                      true_boxes: true_boxes_value,
                                                      y_pred:y_pred_value})

    expected_conf_mask_value = np.ones((1,9,9,5))
    expected_conf_mask_value[0,4,3,2] = 5
    expected_conf_mask_value[0,4,4,2] = 5
    assert np.allclose(conf_mask_value, expected_conf_mask_value)

def test_loss_op(setup_y_true_tensor, setup_y_pred_tensor, setup_true_boxes_tensor):
    # 1. build loss function
    batch_size = 1
    yolo_loss = YoloLoss(grid_size=9, nb_class=4)
    custom_loss = yolo_loss.custom_loss(batch_size)

    # 2. placeholder : (y_true, y_pred)
    y_true, y_true_value = setup_y_true_tensor
    y_pred, y_pred_value = setup_y_pred_tensor

    # 3. loss operation
    loss_op = custom_loss(y_true, y_pred)
    
    # 4. setup feed values for each placeholders (true_boxes, y_true, y_pred
    _, true_boxes_value = setup_true_boxes_tensor
    
    # 5. run loss_op in session
    # y_true, y_pred�� ���� value�� insert
    loss_value = run_op(loss_op, feed_dict={y_true: y_true_value,
                                            y_pred: y_pred_value})

    assert np.allclose(loss_value, 5.47542)


def test_y_tensor_activation(setup_y_true_tensor):
    y_pred = tf.placeholder(tf.float32, [None, 9, 9, 5, 9], name='y_pred')
    y_true, y_true_value = setup_y_true_tensor
    
    activator = _Activator()
    pred_tensor, true_tensor = activator.run(y_true, y_pred)
    

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
