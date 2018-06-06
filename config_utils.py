#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import sys
import json
import numpy as np
import os
import subprocess
import shutil
import yaml

try:
    import cupy as cp
except:
    cp = None
    print("Please install cupy if you want to use gpus")

from sklearn.model_selection import train_test_split

import chainer
from chainer import iterators
from chainer.training import extensions

import chainercv
from chainercv.datasets import voc_bbox_label_names
from chainer.datasets import ConcatenatedDataset
from chainer.training import triggers

from utils import lr_utils
from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

SEED = 0

def parse_dict(dic, key, value=None):
    return value if dic is None or not key in dic else dic[key]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='default.yml', type=str, help='configure file')
    parser.add_argument('--img_path', default='./', type=str, help='image path')
    parser.add_argument('--width', default=608, type=int)
    parser.add_argument('--height', default=608, type=int)
    parser.add_argument('--thresh', default=0.25, type=float)
    parser.add_argument('--nms_thresh', default=0.3, type=float)
    parser.add_argument('--nms', default='class', type=str, help='the way to nms')
    parser.add_argument('--name', default='datasets/coco/coco_names.txt',
                         type=str, help='Class names')
    parser.add_argument('--batchsize', default=1, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--save', default='prediction', type=str)
    args = parser.parse_args()
    config = yaml.load(open(args.config))

    SEED = parse_dict(config, "seed", 8964)
    np.random.seed(SEED)
    if cp:
        pass
        # cp.random.seed(SEED)

    if config["mode"] == "Test":
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

        return config, args

    subprocess.check_call(["mkdir", "-p", config["results"]])
    shutil.copy(args.config, os.path.join(config['results'], args.config.split('/')[-1]))
    return config

def parse_trigger(trigger):
    return (int(trigger[0]), trigger[1])

def create_extension(trainer, test_iter, model, config, devices=None):
    """Create extension for training models"""
    for key, ext in config.items():
        if key == "Evaluator":
            cl = get_class(ext['module'])
            Evaluator = getattr(cl, ext['name'])
            trigger = parse_trigger(ext['trigger'])
            args = parse_dict(ext, 'args', {})
            if parse_dict(args, 'label_names', 'voc') == 'voc':
                args['label_names'] = voc_bbox_label_names
            trainer.extend(Evaluator(
                test_iter, model, **args), trigger=trigger)
        elif key == "dump_graph":
            cl = getattr(extensions, key)
            trainer.extend(cl(ext['name']))
        elif key == 'snapshot':
            cl = getattr(extensions, key)
            trigger = parse_trigger(ext['trigger'])
            trainer.extend(cl(), trigger=trigger)
        elif key == 'snapshot_object':
            cl = getattr(extensions, key)
            trigger = parse_trigger(ext['trigger'])
            args = parse_dict(ext, 'args', {})
            if args:
                if args['method'] == 'best':
                    trigger = triggers.MaxValueTrigger(
                                  args['name'], trigger)
            trainer.extend(cl(model, 'yolov2_{.updater.iteration}'),
                           trigger=trigger)
        elif key == 'LogReport':
            cl = getattr(extensions, key)
            trigger = parse_trigger(ext['trigger'])
            trainer.extend(cl(trigger=trigger))
        elif key == "PrintReport":
            cl = getattr(extensions, key)
            report_list = ext['name'].split(' ')
            trigger = parse_trigger(ext['trigger'])
            trainer.extend(cl(report_list), trigger=trigger)
        elif key == "ProgressBar":
            cl = getattr(extensions, key)
            trainer.extend(cl(update_interval=ext['update_interval']))
        elif key == 'observe_lr':
            cl = getattr(extensions, key)
            trigger = parse_trigger(ext['trigger'])
            trainer.extend(cl(), trigger=trigger)
        elif key == "PolynomialShift":
            cl = getattr(lr_utils, key)
            trigger = parse_trigger(ext['trigger'])
            len_dataset = len(trainer.updater.get_iterator('main').dataset)
            batchsize = trainer.updater.get_iterator('main').batch_size
            args = parse_dict(ext, 'args', {})
            args.update({'len_dataset': len_dataset, 'batchsize': batchsize,
                         'stop_trigger': trainer.stop_trigger})
            trainer.extend(cl(**args))
        elif key == "DarknetLRScheduler":
            cl = getattr(lr_utils, key)
            args = parse_dict(ext, 'args', {})
            args['step_trigger'] = [int(num) for num in args['step_trigger']]
            trainer.extend(cl(**args))
        elif key == "ExponentialShift":
            cl = getattr(extensions, key)
            attr = ext['attr']
            rate = ext['rate']
            name = ext['name']
            numbers = [int(num) for num in ext['numbers']]
            trainer.extend(cl(attr, rate),
                           trigger=triggers.ManualScheduleTrigger(numbers, name))

    return trainer

def create_updater(train_iter, optimizer, config, devices):
    if "MultiprocessParallelUpdater" in config['name']:
        Updater = chainer.training.updaters.MultiprocessParallelUpdater
        return Updater(train_iter, optimizer, devices=devices)

    Updater = getattr(chainer.training, config['name'])
    if "Standard" in config['name']:
        device = None if devices is None else devices['main']
        return Updater(train_iter, optimizer, device=device)
    else:
        return Updater(train_iter, optimizer, devices=devices)

def create_optimizer(config, model):
    Optimizer = getattr(chainer.optimizers, config['name'])
    opt = Optimizer(**config['args'])
    opt.setup(model)
    if 'hook' in config.keys():
        for key, value in config['hook'].items():
            hook = getattr(chainer.optimizer, key)
            opt.add_hook(hook(value))
    return opt

def create_iterator_test(test_data, config):
    Iterator = getattr(chainer.iterators, config['name'])
    args = parse_dict(config, 'args', {})
    args['repeat'] = False
    args['shuffle'] = False
    test_iter = Iterator(test_data, config['test_batchsize'], **args)
    return test_iter

def create_iterator(train_data, test_data, config, devices, updater_name):
    Iterator = getattr(chainer.iterators, config['name'])
    args = parse_dict(config, 'args', {})
    if 'MultiprocessParallelUpdater' in updater_name:
        train_iter = [
            Iterator(i, config['train_batchsize'], **args)
            for i in chainer.datasets.split_dataset_n_random(train_data, len(devices))]
    else:
        train_iter = Iterator(train_data, config['train_batchsize'], **args)

    args['repeat'] = False
    test_iter = None
    if test_data is not None:
        test_iter = Iterator(test_data, config['test_batchsize'], **args)
    return train_iter, test_iter

def parse_devices(gpus, updater_name):
    if gpus:
        devices = {'main': gpus[0]}
        if not 'MultiprocessParallelUpdater' in updater_name:
            chainer.cuda.get_device_from_id(gpus[0]).use()
        for gid in gpus[1:]:
            devices['gpu{}'.format(gid)] = gid
        return devices
    return None

def get_class(mod):
    assert len(mod) > 0, (name, mod)
    m = sys.modules[
        mod] if mod in sys.modules else importlib.import_module(mod)
    return m

def load_dataset_test(config):
    test_config = config['test']
    cl = get_class(test_config['module'])
    test_loader = getattr(cl, test_config['name'])
    test_data = test_loader(**test_config['args'])
    return test_data

def load_dataset(config):
    train_config = config['train']
    cl = get_class(train_config['module'])
    train_loader = getattr(cl, train_config['name'])
    train_data = train_loader(**train_config['args'])
    if parse_dict(config, 'train2', None):
        train_config = config['train2']
        cl = get_class(train_config['module'])
        train_loader = getattr(cl, train_config['name'])
        train_data2 = train_loader(**train_config['args'])
        train_data = ConcatenatedDataset(train_data, train_data2)

    test_data = None
    if parse_dict(config, 'valid', None):
        test_config = config['valid']
        cl = get_class(test_config['module'])
        test_loader = getattr(cl, test_config['name'])
        test_data = test_loader(**test_config['args'])
    return train_data, test_data

def get_model(config):
    cl = get_class(config['module'])
    Model = getattr(cl, config['name'])
    pretrained_model = parse_dict(config, 'pretrained_model', None)
    return Model(config["architecture"], pretrained_model=pretrained_model)
