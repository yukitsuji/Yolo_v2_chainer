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
import matplotlib.pyplot as plt

subprocess.call(['sh', 'setup.sh'])

from config_utils import *
from models.yolov2_base import *
from utils.image_loader import load_img_like_darknet
from utils.postprocess import visualize_with_label, clip_bbox
from utils.postprocess import select_bbox_by_class, select_bbox_by_obj

chainer.cuda.set_max_workspace_size(chainer.cuda.get_max_workspace_size())
os.environ["CHAINER_TYPE_CHECK"] = "0"
chainer.global_config.autotune = True
chainer.global_config.type_check = False

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))


def test_yolov2():
    """test yolov2."""
    config, args = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                    config['iterator'])
    model.to_gpu(devices['main'])

    img_path = args.img_path
    h = args.height
    w = args.width
    thresh = args.thresh
    nms_thresh = args.nms_thresh
    label_names = open(args.name, 'r').read().split('\n')[:-1]

    for batch in test_iter:
        start, stop = create_timer()
        img = batch[0]
        img = chainer.cuda.to_gpu(img, device=devices['main'])
        img_shape = img.shape[2:]
        bbox_pred, conf, prob = model.inference(img, img_shape)
        bbox_pred = chainer.cuda.to_cpu(bbox_pred)
        conf = chainer.cuda.to_cpu(conf)
        prob = chainer.cuda.to_cpu(prob)
        print_timer(start, stop, sentence="Inference time")

        # NMS by each class
        if args.nms == 'class':
            bbox_pred, prob, cls_inds, index = \
                select_bbox_by_class(bbox_pred, conf, prob, thresh, nms_thresh)
        else:
            bbox_pred, prob, cls_inds, index = \
                select_bbox_by_obj(bbox_pred, conf, prob, thresh, nms_thresh)

        # Clip with and height
        bbox_pred[:, 0] -= bbox_pred[:, 2] / 2 # left_x
        bbox_pred[:, 1] -= bbox_pred[:, 3] / 2 # top_y
        bbox_pred[:, 2] += bbox_pred[:, 0] # right_x
        bbox_pred[:, 3] += bbox_pred[:, 1] # bottom_y
        # bbox_pred[:, ::2] -= delta[1] # TODO
        # bbox_pred[:, 1::2] -= delta[0] # TODO
        bbox_pred = clip_bbox(bbox_pred, orig_shape)

        print(bbox_pred)
        print(prob)
        print(cls_inds)
        pred_names = [label_names[i] for i in cls_inds]
        print(pred_names)

        # Visualize
        ax = visualize_with_label(orig_img, bbox_pred, prob, label_names)
        plt.show()


def main():
    test_yolov2()

if __name__ == '__main__':
    main()
