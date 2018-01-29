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
import matplotlib as mpl

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
import matplotlib.pyplot as plt

subprocess.call(['sh', 'setup.sh'])

from config_utils import *
from models.yolov2_base import *
from utils.image_loader import load_img_like_darknet
from utils.cython_util.nms_by_class import nms_by_class, nms_by_obj
# nms_by_obj
from utils.postprocess import visualize_with_label, clip_bbox

chainer.cuda.set_max_workspace_size(chainer.cuda.get_max_workspace_size())
os.environ["CHAINER_TYPE_CHECK"] = "0"
chainer.global_config.autotune = True
chainer.global_config.type_check = False

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))


def demo_yolov2():
    """Demo yolov2."""
    config, args = parse_args()
    model = get_model(config["model"])
    if args.gpu != -1:
        model.to_gpu(args.gpu)

    img_path = args.img_path
    h = args.height
    w = args.width
    thresh = args.thresh
    nms_thresh = args.nms_thresh
    nms_method = nms_by_class if args.nms == 'class' else nms_by_obj
    label_names = open(args.name, 'r').read().split('\n')[:-1]

    orig_img, img, delta = load_img_like_darknet(img_path, h, w)
    orig_shape = orig_img.shape[:2]
    img_shape = img.shape[2:]

    # Dummy: create graph
    dummy = np.zeros((1, 3, h, w), dtype='f') + 1
    if args.gpu != -1:
        dummy = chainer.cuda.to_gpu(dummy, device=args.gpu)
    model.inference(dummy, orig_shape)

    # Inference
    if args.gpu != -1:
        start, stop = create_timer()
        img = chainer.cuda.to_gpu(img, device=args.gpu)
    else:
        start = time.time()
    bbox_pred, conf, prob = model.inference(img, img_shape)
    bbox_pred = chainer.cuda.to_cpu(bbox_pred)
    conf = chainer.cuda.to_cpu(conf)
    prob = chainer.cuda.to_cpu(prob)
    if args.gpu != -1:
        print_timer(start, stop, sentence="Inference time(gpu)")
    else:
        print("Inference time(cpu):", time.time() - start)

    # Post processing
    start = time.time()
    if nms_method == nms_by_class:
        prob_shape = prob.shape
        prob = np.broadcast_to(conf[:, None], prob_shape) * prob
        prob = prob.transpose(1, 0)
        sort_index = np.argsort(prob, axis=1)[:, ::-1].astype(np.int32)
        prob = prob.transpose(1, 0)
        sort_index = sort_index.transpose(1, 0)
        index = nms_by_class(bbox_pred, prob, sort_index, thresh, nms_thresh)
        index = np.asarray(index, dtype='i')
        bbox_pred = bbox_pred[index[:, 0]]
        prob = prob[index[:, 0], index[:, 1]]
        cls_inds = index[:, 1]

    elif nms_method == nms_by_obj:
        cls_inds = np.argmax(prob, axis=1)
        prob = prob[np.arange(prob.shape[0]), cls_inds]
        prob = conf * prob
        is_index = np.where(prob >= thresh)
        bbox_pred = bbox_pred[is_index]
        prob = prob[is_index]
        sort_index = np.argsort(prob)[::-1]
        bbox_pred = bbox_pred[sort_index]
        prob = prob[sort_index]
        cls_inds = cls_inds[is_index][sort_index]
        index = nms_by_obj(bbox_pred, prob, nms_thresh)
        bbox_pred = bbox_pred[index]
        prob = prob[index]
        cls_inds = cls_inds[index]

    bbox_pred[:, 0] -= bbox_pred[:, 2] / 2 # left_x
    bbox_pred[:, 1] -= bbox_pred[:, 3] / 2 # top_y
    bbox_pred[:, 2] += bbox_pred[:, 0] # right_x
    bbox_pred[:, 3] += bbox_pred[:, 1] # bottom_y
    bbox_pred[:, ::2] -= delta[1]
    bbox_pred[:, 1::2] -= delta[0]
    bbox_pred = clip_bbox(bbox_pred, orig_shape)
    pred_names = [label_names[i] for i in cls_inds]
    print("Post Processing: ", time.time() - start)
    print(bbox_pred)
    print(prob)
    print(pred_names)

    # Visualize
    index = index[:, 0] if nms_method == nms_by_class else None
    ax = visualize_with_label(orig_img, bbox_pred, prob, pred_names,
                              index=index)
    if args.save:
        ax.axis('off')
        plt.savefig("{}.png".format(args.save),
                    bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.show()


def main():
    demo_yolov2()

if __name__ == '__main__':
    main()
