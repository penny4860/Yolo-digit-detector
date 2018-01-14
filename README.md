[![Build Status](https://travis-ci.org/penny4860/Yolo-digit-detector.svg?branch=master)](https://travis-ci.org/penny4860/Yolo-digit-detector) [![codecov](https://codecov.io/gh/penny4860/Yolo-digit-detector/branch/master/graph/badge.svg)](https://codecov.io/gh/penny4860/Yolo-digit-detector)

# SVHN yolo-v2 digit detector

I have implemented a digit detector that applies yolo-v2 to svhn dataset. This is an ongoing project.

<img src="images/svhn.png" height="600">


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

I recommend that you create and use an anaconda env that is independent of your project. You can create anaconda env for this project by following these simple steps. This process has been verified on Windows 10 and ubuntu 16.04.

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

### 1. Digit Detection using pretrained weight file

In this project, the pretrained weight file is stored in [mobile_288_weights.h5] (https://github.com/penny4860/Yolo-digit-detector/blob/master/tests/dataset/svhn/mobile_288_weights.h5).

Example code for predicting a digit region in a natural image is described in [detection_example.ipynb] (https://github.com/penny4860/Yolo-digit-detector/blob/master/detection_example.ipynb).

### 2. Training from scratch

#### 1) Data preparation

Download the [SVHN images](http://ufldl.stanford.edu/housenumbers/) and its [annotations in VOC format](https://github.com/penny4860/svhn-voc-annotation-format).

Organize the dataset into 2 folders:

* train_image_folder
	* the folder that contains the [SVHN images](http://ufldl.stanford.edu/housenumbers/).

* train_annot_folder
	* the folder that contains the [annotations in VOC format](https://github.com/penny4860/svhn-voc-annotation-format).

#### 2) Edit the configuration file

A description will be added....

#### 3) Start the training process

A description will be added....

## Evaluation of the current implementation:

A description will be added....

## Copyright

* See [LICENSE](LICENSE) for details.
* This project started at [basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras). I refactored the source code structure of [basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras) and added the CI test. I also applied the SVHN dataset to implement the digit detector. Thanks to the [Huynh Ngoc Anh](https://github.com/experiencor) for providing a good project as open source.

