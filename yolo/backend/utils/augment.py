# -*- coding: utf-8 -*-
from imgaug import augmenters as iaa
import cv2
import numpy as np
np.random.seed(1337)


class ImgAugment(object):
    def __init__(self, w, h, jitter):
        """
        # Args
            desired_w : int
            desired_h : int
            jitter : bool
        """
        self._jitter = jitter
        self._w = w
        self._h = h
        
    def imread(self, img_file, boxes):
        """
        # Args
            img_file : str
            boxes : array, shape of (N, 4)
        
        # Returns
            image : 3d-array, shape of (h, w, 3)
            boxes_ : array, same shape of boxes
                jittered & resized bounding box
        """
        # 1. read image file
        image = cv2.imread(img_file)
    
        # 2. make jitter on image
        boxes_ = np.copy(boxes)
        if self._jitter:
            image, boxes_ = make_jitter_on_image(image, boxes_)
    
        # 3. resize image            
        image, boxes_ = resize_image(image, boxes_, self._w, self._h)
        return image, boxes_


def make_jitter_on_image(image, boxes):
    h, w, _ = image.shape

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
    if flip > 0.5:
        image = cv2.flip(image, 1)
        is_flip = True
    else:
        is_flip = False

    aug_pipe = _create_augment_pipeline()
    image = aug_pipe.augment_image(image)
    
    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * scale - offx)
        x2 = int(x2 * scale - offx)
        
        y1 = int(y1 * scale - offy)
        y2 = int(y2 * scale - offy)

        if is_flip:
            xmin = x1
            x1 = w - x2
            x2 = w - xmin
        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)


def resize_image(image, boxes, desired_w, desired_h):
    h, w, _ = image.shape
    
    # resize the image to standard size
    image = cv2.resize(image, (desired_h, desired_w))
    image = image[:,:,::-1]

    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * float(desired_w) / w)
        x1 = max(min(x1, desired_w), 0)
        x2 = int(x2 * float(desired_w) / w)
        x2 = max(min(x2, desired_w), 0)
        
        y1 = int(y1 * float(desired_h) / h)
        y1 = max(min(y1, desired_h), 0)
        y2 = int(y2 * float(desired_h) / h)
        y2 = max(min(y2, desired_h), 0)

        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)


def _create_augment_pipeline():
    
    ### augmentors by https://github.com/aleju/imgaug
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    aug_pipe = iaa.Sequential(
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
    return aug_pipe


if __name__ == '__main__':
    
    img_file = "C://Users//penny//git//basic-yolo-keras//sample//raccoon_train_imgs//raccoon-1.jpg"
    objects = [{'name': 'raccoon', 'xmin': 81, 'ymin': 88, 'xmax': 522, 'ymax': 408},
               {'name': 'raccoon', 'xmin': 100, 'ymin': 100, 'xmax': 400, 'ymax': 300}]
    boxes = np.array([[81,88,522,408],
                      [100,100,400,300]])
    
    desired_w = 416
    desired_h = 416
    jitter = True
    
    aug = ImgAugment(desired_w, desired_h, jitter)
    img, boxes_ = aug.imread(img_file, boxes)
    img = img.astype(np.uint8)
    
    import matplotlib.pyplot as plt
    for box in boxes_:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
    plt.imshow(img)
    plt.show()

    
    
    



