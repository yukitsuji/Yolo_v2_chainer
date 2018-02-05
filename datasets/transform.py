#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

import chainer
from chainer.datasets import TransformDataset
from chainercv import transforms
from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

from config_utils import parse_dict
from utils.cython_util.nms import nms_gt_anchor


class Transform(object):

    def __init__(self, value, dim=None, max_target=30, batchsize=1,
                 anchors=None, n_boxes=5, downscale=32):
        self.value = value
        self.max_target = max_target
        self.anchors = anchors
        self.dim = dim
        self.output_shape = (dim[0], dim[0])
        self.batchsize = batchsize
        self.n_boxes = n_boxes
        self.downscale = downscale
        self.i = 0
        self.count = 0

    def __call__(self, in_data):
        """in_data includes three datas.
        Args:
            img(array): Shape is (3, H, W). range is [0, 255].
            bbox(array): Shape is (N, 4). (y_min, x_min, y_max, x_max).
                         range is [0, max size of boxes].
            label(array): Classes of bounding boxes.

        Returns:
            img(array): Shape is (3, out_H, out_W). range is [0, 1].
                        interpolation value equals to self.value.
        """
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping
        if self.count % 2 == 0 and self.count % self.batchsize == 0 and self.count != 0:
            self.i += 1
            i = self.i % len(self.dim)
            self.output_shape = (self.dim[i], self.dim[i])
        # print(self.count, self.i, self.output_shape)
        self.count += 1
        
        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img, brightness_delta=32,
                             contrast_low=0.5, contrast_high=1.5,
                             saturation_low=0.5, saturation_high=1.5,
                             hue_delta=25)

        # Normalize. range is [0, 1]
        img /= 255.0

        # 2. Random expansion: resize and translation.
        if np.random.randint(2):
            img, param = transforms.random_expand(
                             img, max_ratio=2, fill=self.value,
                             return_param=True)

            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)

        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation # TODO
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, self.output_shape)
        bbox = transforms.resize_bbox(bbox, (H, W), self.output_shape)

        # 5. Random horizontal flipping # OK
        img, params = transforms.random_flip(
                          img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
                   bbox, self.output_shape, x_flip=params['x_flip'])

        # Preparation for Yolov2 network
        bbox[:, ::2] /= self.output_shape[0] # y
        bbox[:, 1::2] /= self.output_shape[1] # x

        num_bbox = len(bbox)
        len_max = max(num_bbox, self.max_target)

        gmap = create_map_anchor_gt(bbox, self.anchors, self.output_shape,
                                    self.downscale, self.n_boxes, len_max)

        out_bbox = np.zeros((len_max, 4), dtype='f')
        out_bbox[:num_bbox] = bbox[:num_bbox]
        out_label = np.zeros((len_max), dtype='i')
        out_label[:num_bbox] = label

        gmap = gmap[:self.max_target]
        out_bbox = out_bbox[:self.max_target]
        out_label = out_label[:self.max_target]

        img = np.clip(img, 0, 1)
        return img, out_bbox, out_label, gmap, np.array([num_bbox], dtype='i')

def create_map_anchor_gt(bbox, anchors, output_shape, downscale, n_boxes,
                         max_target):
    """
    Args:
        bbox(array): Shape is (Boxes, 4). (y, x, y, x)

    Returns:
        gmap(array): Shape is (max_target, 3). (x, y, box)
    """
    shape = (int(output_shape[0] // downscale), int(output_shape[1] // downscale))
    gmap = np.zeros((max_target, 3), dtype='i')
    num_bbox = len(bbox)
    if num_bbox:
        gt_bbox = bbox.copy()
        gt_bbox_w = gt_bbox[:, 3] - gt_bbox[:, 1]
        gt_bbox_h = gt_bbox[:, 2] - gt_bbox[:, 0]
        gt_bbox_center_x = gt_bbox[:, 1] + gt_bbox_w / 2.
        gt_bbox_center_y = gt_bbox[:, 0] + gt_bbox_h / 2.
        gmap[:num_bbox, 0] = (gt_bbox_center_x * shape[1]).astype('i')
        gmap[:num_bbox, 1] = (gt_bbox_center_y * shape[0]).astype('i')
        anchors = anchors.copy()
        anchors[:, 0] /= shape[1] # w
        anchors[:, 1] /= shape[0] # h
        gmap[:num_bbox, 2] = nms_gt_anchor(gt_bbox_w, gt_bbox_h, anchors)
    return gmap
