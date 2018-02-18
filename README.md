
# resnet-yolo detector

## Usage for python code

#### 0. Requirement

* python 3.5
* anaconda 4.4.0
* tensorflow 1.2.1
* keras 2.1.1
* opencv 3.3.0
* imgaug
* Etc.

I recommend that you create and use an anaconda env that is independent of your project. You can create anaconda env for this project by following these simple steps. This process has been verified on Windows 10 and ubuntu 16.04.

```
$ conda create -n yolo python=3.5 anaconda=4.4.0
$ activate yolo # in linux "source activate yolo"
(yolo) $ pip install tensorflow==1.2.1
(yolo) $ pip install keras==2.1.1
(yolo) $ pip install opencv-python
(yolo) $ pip install imgaug
(yolo) $ pip install pytest-cov
(yolo) $ pip install codecov
(yolo) $ pip install -e .
```

#### 1. Detection using pretrained weight file

#### 2. Training from scratch

## Results

#### 1. Raccoon dataset : https://github.com/experiencor/raccoon_dataset

<img src="images/raccoon-12.jpg">

* pretrained weight file is stored at [raccoon](https://drive.google.com/drive/folders/17Co0b5YDNVlWVfuTqygRY_U2qg8FwGmy)
* training set evaluation (160-images)
	* fscore / precision / recall: 0.937, 0.963, 0.913
* test set evaluation (40-images)
	* fscore / precision / recall: 0.631, 0.75, 0.545


## Copyright

* See [LICENSE](LICENSE) for details.
* This project started at [basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras). I refactored the source code structure of [basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras) and added the CI test. I also applied the SVHN dataset to implement the digit detector. Thanks to the [Huynh Ngoc Anh](https://github.com/experiencor) for providing a good project as open source.

