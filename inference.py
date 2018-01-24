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
import matplotlib.pyplot as plt

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))


def demo_yolov2():
    """Demo yolov2."""
    config, img_path = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    model.to_gpu(devices['main'])

    dataset_config = config['dataset']['test']['args']
    index = 0
    # for batch in test_iter:
    #     input_img = batch[0][0].transpose(1, 2, 0)
    #     batch = chainer.dataset.concat_examples(batch, devices['main'])
    # pred_depth, pred_pose, pred_mask = model.inference(*batch)
    imgs = chainer.cuda.to_gpu(np.zeros((1, 3, 416, 416), dtype='f'), device=devices['main'])
    model.inference(imgs)
    # depth = chainer.cuda.to_cpu(pred_depth.data[0, 0])
    # depth = normalize_depth_for_display(depth)
    # cv2.imwrite("input_{}.png".format(index), (input_img + 1) / 2 * 255)
    # cv2.imwrite("output{}.png".format(index), depth * 255 )
    index += 1

def main():
    demo_yolov2()

if __name__ == '__main__':
    main()
