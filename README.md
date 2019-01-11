# CVLab 2019 Training Course Project

[Google Slides](https://docs.google.com/presentation/d/1-HFXcnl6WXGrZNSE3iMwjJTgC8-JScPVPBywpegCleY/edit?usp=sharing)

## Task

Given 5000 images of car [^1] (260MB), detect the license plate and unwarp it.

Each image is guaranteed with 1 license plate.

The ground truth and metadata are encoded in the name of each image.

To download & uncompress the dataset:

```sh
$ mkdir raw && cd raw/
$ wget https://github.com/amoshyc/cvlab-2019w-project/releases/download/v0.1/ccpd5000.tar.gz
$ tar zxvf ccpd5000.tar.gz
$ ls ccpd5000/**/*.jpg | wc -l # expected 5000
```

[^1]: This is a subset of [CCPD](https://github.com/detectRecog/CCPD) dataset.

## Environment

It is developed on:

1. Python 3.6
2. Pytorch 1.0.0
3. torchvision 0.2.1
4. scikit-image 0.14.1
5. tqdm 4.29.0

Full environment is [here](https://github.com/amoshyc/cvlab-2019w-project/blob/master/environment.yml). Use

```sh
$ wget https://raw.githubusercontent.com/amoshyc/cvlab-2019w-project/master/environment.yml
$ conda env create -f environment.yml
```

to create identical virtual environment `ccpd`. Manually edit first line of `environment.yml` to change the name of virtual environment if you want.

## exp1

Regress the coordinate of 4 corners directly.

To train the model:

```sh
$ cd exp1/
$ python prepare.py
$ CUDA_VISIBLE_DEVICES="0" python train.py
```

## exp2

Predict the coordinates by 4 heatmaps where each heatmap:

1. a 2D Gaussian Distribution (sigma=3) is placed at corresponding corner position
2. other pixels outside of 1. is filled with 0
3. resolution is (H/4, W/4) where (H, W) is the resolution of input image

To train the model:

```sh
$ cd exp2/
$ python prepare.py
$ CUDA_VISIBLE_DEVICES="0" python train.py
```