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
# from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

from config_utils import parse_dict
from utils.cython_util.nms import nms_gt_anchor

import random
import six

from chainercv import utils

def crop_with_bbox_constraints(
        img, bbox, crop_width=None, crop_height=None, constraints=None,
        max_trial=10, return_param=False):
    if constraints is None:
        constraints = (
            (0.1, None),
            (None, 1),
            (0.5, None),
            (0.7, None),
        )

    _, H, W = img.shape

    crop_h = int(crop_height)
    crop_w = int(crop_width)

    diff_h = int((H - crop_h) / 2.)
    diff_w = int((W - crop_w) / 2.)

    params = [{
        'constraint': None, 'y_slice': slice(diff_h, diff_h + crop_h),
        'x_slice': slice(diff_w, diff_w + crop_w)}]

    if len(bbox) == 0:
        constraints = list()

    range_H = H - crop_h
    range_W = W - crop_w

    for min_iou, max_iou in constraints:
        if min_iou is None:
            min_iou = 0
        if max_iou is None:
            max_iou = 1

        for _ in six.moves.range(max_trial):
            crop_t = 0 if range_H == 0 else random.randrange(range_H)
            crop_l = 0 if range_W == 0 else random.randrange(range_W)
            crop_bb = np.array((
                crop_t, crop_l, crop_t + crop_h, crop_l + crop_w))

            iou = utils.bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou < iou.min() and iou.max() <= max_iou:
                params.append({
                    'constraint': (min_iou, max_iou),
                    'y_slice': slice(crop_t, crop_t + crop_h),
                    'x_slice': slice(crop_l, crop_l + crop_w)})
                break

    param = random.choice(params)
    img = img[:, param['y_slice'], param['x_slice']]

    if return_param:
        return img, param
    else:
        return img

def random_crop_with_bbox_constraints(
        img, bbox, min_scale=0.3, max_scale=1,
        max_aspect_ratio=2, constraints=None,
        max_trial=50, return_param=False):
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    _, H, W = img.shape
    params = [{
        'constraint': None, 'y_slice': slice(0, H), 'x_slice': slice(0, W)}]

    if len(bbox) == 0:
        constraints = list()

    for min_iou, max_iou in constraints:
        if min_iou is None:
            min_iou = 0
        if max_iou is None:
            max_iou = 1

        for _ in six.moves.range(max_trial):
            if min_iou == 0 and max_iou == 1:
                # IOUを気にせず、bounding box全体を必ず含むような値を取る。
                scale = random.uniform(0.9, max_scale)
            else:
                scale = random.uniform(min_scale, max_scale)

            # scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(H * scale / np.sqrt(aspect_ratio))
            crop_w = int(W * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(H - crop_h)
            crop_l = random.randrange(W - crop_w)
            crop_bb = np.array((
                crop_t, crop_l, crop_t + crop_h, crop_l + crop_w))

            iou = utils.bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou < iou.min() and iou.max() <= max_iou:
                params.append({
                    'constraint': (min_iou, max_iou),
                    'y_slice': slice(crop_t, crop_t + crop_h),
                    'x_slice': slice(crop_l, crop_l + crop_w)})
                break

    param = random.choice(params)
    img = img[:, param['y_slice'], param['x_slice']]

    if return_param:
        return img, param
    else:
        return img

def expand(img, out_h=None, out_w=None, fill=0, return_param=False):
    """Expand an image."""
    C, H, W = img.shape
    out_H = out_h
    out_W = out_w

    y_offset = random.randint(0, out_H - H)
    x_offset = random.randint(0, out_W - W)

    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape((-1, 1, 1))
    out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

    if return_param:
        param = {'ratio': None, 'y_offset': y_offset, 'x_offset': x_offset}
        return out_img, param
    else:
        return out_img

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
        if self.count % 10 == 0 and self.count % self.batchsize == 0 and self.count != 0:
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

        _, H, W = img.shape
        scale = np.random.uniform(0.25, 2)
        random_expand = np.random.uniform(0.8, 1.2, 2)
        net_h, net_w = self.output_shape
        out_h = net_h * scale # random_expand[0]
        out_w = net_w * scale # random_expand[1]
        if H > W:
            out_w = out_h * (float(W) / H) * np.random.uniform(0.8, 1.2)
        elif H < W:
            out_h = out_w * (float(H) / W) * np.random.uniform(0.8, 1.2)

        out_h = int(out_h)
        out_w = int(out_w)

        img = resize_with_random_interpolation(img, (out_h, out_w))
        bbox = transforms.resize_bbox(bbox, (H, W), (out_h, out_w))

        if out_h < net_h and out_w < net_w:
            img, param = expand(img, out_h=net_h, out_w=net_w,
                                 fill=self.value, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])
        else:
            out_h = net_h if net_h > out_h else int(1.2*out_h)
            out_w = net_w if net_w > out_w else int(1.2*out_w)
            img, param = expand(img, out_h=out_h, out_w=out_w,
                                fill=self.value, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

            img, param = crop_with_bbox_constraints(
                             img, bbox, return_param=True,
                             crop_height=net_h, crop_width=net_w)
            bbox, param = transforms.crop_bbox(
                bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
                allow_outside_center=False, return_param=True)
            label = label[param['index']]


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
        num_array = min(num_bbox, self.max_target)

        img = np.clip(img, 0, 1)
        return img, out_bbox, out_label, gmap, np.array([num_array], dtype='i')

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
