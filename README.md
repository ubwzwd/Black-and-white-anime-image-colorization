# 7398project black and white anime image colorization

## Overview

This is the project for EECE7398 in Northeastern University.
This project is a TensorFlow and Keras implement of Residual Encoder Network (based on [Automatic Colorization](http://tinyclouds.org/colorize/)), the pre-trained VGG16 model from [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) and Pyramid Pooling Module (introduced in [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)).

## Structure

* `code`
  * `config.py`: config variables like batch size, training_iters and so on
  * `image_helper.py`: all functions related to image manipulation
  * `read_input.py`: all functions related to input
  * `residual_encoder.py`: the residual encoder model
  * `common.py`: the common part for training and testing, which is mainly the workflow for this model
  * `train.py`: train the residual encoder model using TensorFlow built-in AdamOptimizer
  * `test.py`: test your own images and save the output images
  * `vgg`: pretrained vgg module
  * `summary`: stored graph (checkpoint), part of trainning results, and test results
  * `sample_output`: samples, every picture is concatenated by 3 images, the gray-scale image, the inference and the original image.
* `docs`: papers we read and the presentation material

## How to use

* First please download pre-trained VGG16 model [vgg16.npy](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) to `vgg` folder
* Please ensure that you have the following packages:
  * tensorflow
  * keras
  * numpy
  * matplotlib
  * opencv
* please install pyramid pooling mudole using the following command: ```pip install pyramid_pooling_module```
* You can download the pre-trained module [here](https://github.com/ubwzwd/7398project/releases/tag/v1.0). Unzip all the files under `summary` folder.
* Put the images you want to train under `train` folder. Put the images you want to test under `test` folder. These images are supposed to be coloful images.
* To test, just run `test.py` in terminal. To train, run `train.py`.
* The output images of test will be put under `summary/test/images/`.

## Examples:

* ![1](/README_utils/1.png)
* ![1](/README_utils/2.png)
* ![1](/README_utils/3.png)
* ![1](/README_utils/4.png)
* ![1](/README_utils/5.png)

From left to right, they are the gray-scale images, inference and the original images.

* PS: The pre-trained module is only trained with about 1k images and 6k epochs due to lack of computation and break down of our graphic cards. So the result may not be good enough. We strongly encourage you to train it with more images and epochs.

## Database

We use [Danbooru2018](https://www.gwern.net/Danbooru2018) to train our module.

## Reference

* [Automatic Colorization](http://tinyclouds.org/colorize/)
* [pavelgonchar/colornet](https://github.com/pavelgonchar/colornet)
* [raghavgupta0296/ColourNet](https://github.com/raghavgupta0296/ColourNet)
* [pretrained VGG16 npy file](https://github.com/machrisaa/tensorflow-vgg)
* [Chong Guo’s blog](https://tinyclouds.org/colorize/)
* [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
* [OBJECT DETECTORS EMERGE IN DEEP SCENE CNNS](https://arxiv.org/abs/1412.6856)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification](https://dl.acm.org/citation.cfm?id=2925974)
* [Deep Koalarization: Image Colorization using CNNs and Inception-ResNet-v2](https://arxiv.org/abs/1712.03400)
* [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)