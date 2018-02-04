#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

subprocess.call(['sh', 'setup.sh'])

import chainer
from chainer import cuda, optimizers, serializers
from chainer import training
from chainer.datasets import TransformDataset
from chainercv.links.model.ssd import random_distort

from config_utils import *
from datasets.transform import Transform

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))


def train_yolov2():
    """Training yolov2."""
    config = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    train_data, test_data = load_dataset(config["dataset"])

    train_data = TransformDataset(
        train_data, Transform(0.5, dim=model.dim, max_target=30,
                              anchors=model.anchors, batchsize=config['iterator']['train_batchsize']))

    train_iter, test_iter = create_iterator(train_data, test_data,
                                            config['iterator'], devices,
                                            config['updater']['name'])
    optimizer = create_optimizer(config['optimizer'], model)
    updater = create_updater(train_iter, optimizer, config['updater'], devices)
    trainer = training.Trainer(updater, config['end_trigger'], out=config['results'])
    trainer = create_extension(trainer, test_iter,  model,
                               config['extension'], devices=devices)
    trainer.run()
    chainer.serializers.save_npz(os.path.join(config['results'], 'model.npz'),
                                 model)

def main():
    train_yolov2()

if __name__ == '__main__':
    main()
