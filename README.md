[![Build Status](https://travis-ci.org/penny4860/YOLO-detector.svg?branch=master)](https://travis-ci.org/penny4860/YOLO-detector) [![codecov](https://codecov.io/gh/penny4860/YOLO-detector/branch/master/graph/badge.svg)](https://codecov.io/gh/penny4860/YOLO-detector)

# SVHN yolo-v2 digit detector

I have implemented a digit detector that applies yolo-v2 to svhn dataset. This project is not yet complete.

<img src="images/svhn.png" height="300">


## Design Principle and Source Code Structure

A description will be added....

## Usage for python code

### 0. Requirement

* python 3.5
* anaconda 4.4.0
* tensorflow 1.2.1
* keras 2.0.8
* opencv 3.3.0
* imgaug
* Etc.

I recommend that you create and use an anaconda env that is independent of your project. You can create anaconda env for this project by following these simple steps.

```
$ conda create -n yolo python=3.5 anaconda=4.4.0
$ activate yolo # in linux "source activate yolo"
(yolo) $ pip install tensorflow==1.2.1
(yolo) $ pip install keras==2.0.8
(yolo) $ pip install opencv-python
(yolo) $ pip install imgaug
(yolo) $ pip install pytest-cov
(yolo) $ pip install codecov
(yolo) $ pip install -e .
```

### 1. Data preparation
Download the SVHN dataset from http://ufldl.stanford.edu/housenumbers/.

Chnage annotation format to VOC format.
(Todo : format change script added)

Organize the dataset into 4 folders:

* train_image_folder
	* the folder that contains the train images.

* train_annot_folder
	* the folder that contains the train annotations in VOC format.

* valid_image_folder
	* the folder that contains the validation images.

* valid_annot_folder
	* the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.

### 2. Edit the configuration file

A description will be added....

### 3. Start the training process

A description will be added....

## Evaluation of the current implementation:

A description will be added....

## Copyright

* See [LICENSE](LICENSE) for details.
* This project started at [basic-yolo-keras] (https://github.com/experiencor/basic-yolo-keras). I refactored the source code structure of [basic-yolo-keras] (https://github.com/experiencor/basic-yolo-keras) and added the CI test. I also applied the SVHN dataset to implement the digit detector. Thanks to the [Huynh Ngoc Anh](https://github.com/experiencor) for providing a good project as open source.

