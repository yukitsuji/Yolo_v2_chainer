#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

import chainer
from chainer import cuda, optimizers, serializers
from chainer import training

import cv2
from config_utils import *
from models.yolov2_base import *
import matplotlib.pyplot as plt

chainer.cuda.set_max_workspace_size(chainer.cuda.get_max_workspace_size())
os.environ["CHAINER_TYPE_CHECK"] = "0"
chainer.global_config.autotune = True
chainer.global_config.type_check = False

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))


def load_img_like_darknet(img_path, h, w):
    """Load img like yolov2."""
    img = cv2.imread(img_path)
    orig_h, orig_w, _ = img.shape
    if h / orig_h < w / orig_w:
        new_w = w
        new_h = int(h * (new_w / w))
    else:
        new_h = h
        new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))
    delta_h = new_h - h
    delta_w = new_w - w
    img = img[delta_h : delta_h + h, delta_w : delta_w + w]
    return img.transpose(2, 0, 1)[np.newaxis].astype('f')


def demo_yolov2():
    """Demo yolov2."""
    config, img_path = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    #test_data = load_dataset_test(config["dataset"])
    #test_iter = create_iterator_test(test_data,
    #                                 config['iterator'])
    model.to_gpu(devices['main'])
    h, w = 416, 416
    img = load_img_like_darknet(img_path, h, w)
    # for batch in test_iter:
    #     input_img = batch[0][0].transpose(1, 2, 0)
    #     batch = chainer.dataset.concat_examples(batch, devices['main'])
    # pred_depth, pred_pose, pred_mask = model.inference(*batch)
    for i in range(10):
        imgs = chainer.cuda.to_gpu(np.zeros((1, 3, h, w), dtype='f'), device=devices['main']) + i
        model.inference(imgs)

    start, stop = create_timer()
    imgs = chainer.cuda.to_gpu(img, device=devices['main'])
    model.inference(imgs)
    print_timer(start, stop, sentence="Inference time")


def main():
    demo_yolov2()

if __name__ == '__main__':
    main()
