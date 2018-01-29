#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

def load_img_resize_crop(img_path, h, w):
    """Load image and resize and crop to assigned image size."""
    orig_img = cv2.imread(img_path)[:, :, ::-1]
    orig_h, orig_w, _ = orig_img.shape
    if (orig_h / h) < (orig_w / w):
        new_h = h
        new_w = int(orig_w / (orig_h / h))
    else:
        new_w = w
        new_h = int(orig_h / (orig_w / w))

    orig_img = cv2.resize(orig_img, (new_w, new_h))
    orig_shape = orig_img.shape[:2]
    delta_h = int(abs((new_h - h) / 2))
    delta_w = int(abs((new_w - w) / 2))
    img = orig_img[delta_h : delta_h + h, delta_w : delta_w + w]
    img = img.transpose(2, 0, 1)[np.newaxis].astype('f') / 255.0
    return orig_img, img, (delta_h, delta_w)

def load_img_like_darknet(img_path, h, w):
    """Load image like yolov2's test detection function."""
    orig_img = cv2.imread(img_path)[:, :, ::-1]
    orig_h, orig_w, _ = orig_img.shape
    if (orig_h / h) > (orig_w / w):
        new_h = h
        new_w = int(orig_w / (orig_h / h))
    else:
        new_w = w
        new_h = int(orig_h / (orig_w / w))

    orig_img = cv2.resize(orig_img, (new_w, new_h))
    orig_shape = orig_img.shape[:2]
    delta_h = int(abs((new_h - h) / 2))
    delta_w = int(abs((new_w - w) / 2))
    img = np.zeros((h, w, 3), dtype='f') + 127.5
    img[delta_h:delta_h+new_h, delta_w:delta_w+new_w] = orig_img
    img = img.transpose(2, 0, 1)[np.newaxis] / 255.0
    return orig_img, img, (delta_h, delta_w)
