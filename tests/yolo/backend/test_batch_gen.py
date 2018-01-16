
import numpy as np
import pytest
import os
from yolo.backend.batch_gen import create_batch_generator
import yolo

TEST_SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset")

# @pytest.fixture(scope='function')
# def setup():
#     import json
#     from yolo.backend.utils.annotation import parse_annotation
#     with open(os.path.join(TEST_SAMPLE_DIR, "config.json")) as config_buffer:    
#         config = json.loads(config_buffer.read())
#         
#     input_size = config["model"]["input_size"]
#     grid_size = int(input_size/32)
#     batch_size = 8
#     max_box_per_image = config["model"]["max_box_per_image"]
#     anchors = config["model"]["anchors"]
# 
#     train_annotations = parse_annotation(config['train']['train_annot_folder'], 
#                                                        config['train']['train_image_folder'], 
#                                                        config['model']['labels'])
#     return train_annotations, input_size, grid_size, batch_size, max_box_per_image, anchors
# 
# @pytest.fixture(scope='function')
# def expected():
#     x_batch_gt = np.load(os.path.join(TEST_SAMPLE_DIR, "x_batch_gt.npy"))
#     b_batch_gt = np.load(os.path.join(TEST_SAMPLE_DIR, "b_batch_gt.npy"))
#     y_batch_gt = np.load(os.path.join(TEST_SAMPLE_DIR, "y_batch_gt.npy"))
#     return x_batch_gt, b_batch_gt, y_batch_gt

# def test_generate_batch(setup, expected):
#     train_annotations, input_size, grid_size, batch_size, max_box_per_image, anchors = setup
#     x_batch_gt, b_batch_gt, y_batch_gt = expected
#      
#     batch_gen = create_batch_generator(train_annotations,
#                               input_size,
#                               grid_size,
#                               batch_size,
#                               max_box_per_image,
#                               anchors,
#                               jitter=False)
#  
#     # (8, 416, 416, 3) (8, 1, 1, 1, 10, 4) (8, 13, 13, 5, 6)
#     (x_batch, b_batch), y_batch = batch_gen[0]
#      
#     assert np.array_equal(x_batch, x_batch_gt) == True 
#     assert np.array_equal(b_batch, b_batch_gt) == True 
#     assert np.array_equal(y_batch, y_batch_gt) == True 


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

        
