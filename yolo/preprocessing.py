
import cv2
import copy
import numpy as np
np.random.seed(1337)
from imgaug import augmenters as iaa
from keras.utils import Sequence
from yolo.box import BoundBox, bbox_iou


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

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Todo : class extraction
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

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

    def _is_valid_obj(self, x1, y1, x2, y2, label):
        if x2 > x1 and y2 > y1 and label in self.config['LABELS']:
            return True
        else:
            return False

    def _generate_x(self, img, all_objs):
        # assign input image to x_batch
        if self.norm != None: 
            x = self.norm(img)
        else:
            # plot image and bounding boxes for sanity check
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(img[:,:,::-1], obj['name'], 
                                (obj['xmin']+2, obj['ymin']+12), 
                                0, 1.2e-3 * img.shape[0], 
                                (0,255,0), 2)
                    
            x = img
        return x

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

    def _generate_box(self, obj):
        center_x = .5*(obj['xmin'] + obj['xmax'])
        center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
        center_y = .5*(obj['ymin'] + obj['ymax'])
        center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))
        
        return [center_x, center_y, center_w, center_h], grid_x, grid_y

    def __getitem__(self, idx):
        
        anns = self._get_annotations_batch(idx)
        batch_size = len(anns)

        instance_count = 0

        x_batch = np.zeros((batch_size, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((batch_size, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((batch_size, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))                # desired network output

        # loop over batch
        for annotation in anns:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(annotation, jitter=self.jitter)
            
            # assign input image to x_batch
            x_batch[instance_count] = self._generate_x(img, all_objs)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            # loop over objects in one image
            for obj in all_objs:
                if self._is_valid_obj(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['name']):
                    box, grid_x, grid_y = self._generate_box(obj)

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        best_anchor = self._get_anchor_idx(box)
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1
                        
                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # increase instance counter in current batch
            instance_count += 1  

        self.counter += 1
        #print ' new batch created', self.counter
        
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
        self.counter = 0

    def aug_image(self, train_instance, jitter):
        """
        # Args
            train_instance : dict
            jitter : bool
        
        # Returns
            image : 3d-array
            all_objs : list of dicts
                "name"
                "xmin"
                "xmax"
                "ymin"
                "ymax"
        """
        
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h, w, c = image.shape
        
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            
            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)
                
            image = self.aug_pipe.augment_image(image)            
            
        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
                
        return image, all_objs

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

        
