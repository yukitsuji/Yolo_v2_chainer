#/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import time

import cv2 as cv
import numpy as np
from PIL import Image
import copy

from chainer import datasets
from datasets.coco.coco_detection_dataset import CocoDetectionDataset
from chainer import functions as F


def _transform(inputs, input_scale):
    img, bboxes, labels = inputs
    del inputs

    # Color variance

    # Random crop

    # Multi scale resize.

    # Random flip
    
    img, bboxes, labels = data_augmentation(img, bboxes, labels)
    return img, bboxes, labels


class CocoDetectionTransformed(datasets.TransformDataset):
    def __init__(self, root_dir='./', data_dir='train2014',
                 anno_file='annotations', input_scale=[320, 608]):
        self.d = CocoDetectionDataset(
                     root_dir=root_dir, data_dir=data_dir, anno_file=anno_file)
        t = partial(
            _transform, input_scale=input_scale)
        super().__init__(self.d, t)
