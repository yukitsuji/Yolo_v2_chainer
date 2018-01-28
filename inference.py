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
from utils.cython_util.nms_by_class import nms_by_class, nms_by_obj
# from utils.cython_util.nms_by_obj import nms_by_obj

chainer.cuda.set_max_workspace_size(chainer.cuda.get_max_workspace_size())
os.environ["CHAINER_TYPE_CHECK"] = "0"
chainer.global_config.autotune = True
chainer.global_config.type_check = False

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))


def demo_yolov2():
    """Demo yolov2."""
    config, img_path = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    #test_data = load_dataset_test(config["dataset"])
    #test_iter = create_iterator_test(test_data,
    #                                 config['iterator'])
    # model.to_gpu(devices['main'])
    h, w = 608, 608
    thresh = 0.25
    nms_thres = 0.3

    orig_img, img, delta = load_img_like_darknet(img_path, h, w)
    orig_shape = orig_img.shape[:2]
    img_shape = img.shape[2:]
    # for batch in test_iter:
    #     input_img = batch[0][0].transpose(1, 2, 0)
    #     batch = chainer.dataset.concat_examples(batch, devices['main'])
    #     pred_depth, pred_pose, pred_mask = model.inference(*batch)
    # for i in range(1):
    #     # imgs = chainer.cuda.to_gpu(np.zeros((1, 3, h, w), dtype='f'), device=devices['main']) + i
    #     imgs = np.zeros((1, 3, h, w), dtype='f') + 1.
    #     model.inference(imgs, orig_shape)

    # start, stop = create_timer()
    # imgs = chainer.cuda.to_gpu(img, device=devices['main'])
    bbox_pred, conf, prob = model.inference(img, img_shape)
    bbox_pred = chainer.cuda.to_cpu(bbox_pred)
    conf = chainer.cuda.to_cpu(conf)
    prob = chainer.cuda.to_cpu(prob)
    # print_timer(start, stop, sentence="Inference time")

    # NMS by each class
    nms_method = nms_by_obj
    if nms_method == nms_by_class:
        bbox_pred, prob = nms_by_class(bbox_pred, prob, nms_thres)
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
        print(prob)
        index = nms_by_obj(bbox_pred, prob, nms_thres)
        bbox_pred = bbox_pred[index]
        prob = prob[index]
        cls_inds = cls_inds[index]

    # Clip with and height
    bbox_pred[:, 0] -= bbox_pred[:, 2] / 2 # left_x
    bbox_pred[:, 1] -= bbox_pred[:, 3] / 2 # top_y
    bbox_pred[:, 2] += bbox_pred[:, 0] # right_x
    bbox_pred[:, 3] += bbox_pred[:, 1] # bottom_y
    bbox_pred[:, ::2] -= delta[1] # TODO
    bbox_pred[:, 1::2] -= delta[0] # TODO
    bbox_pred = clip_bbox(bbox_pred, orig_shape)

    print(bbox_pred)
    print(prob)
    print(cls_inds)
    coco_names = open('datasets/coco/coco_names.txt', 'r').read().split('\n')[:-1]
    pred_cls = [coco_names[i] for i in cls_inds]
    print(pred_cls)

    # Visualize
    img = visualize(orig_img, bbox_pred, prob)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize(img, bbox_pred, prob):
    for bbox, p in zip(bbox_pred, prob):
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[2], bbox[3])
        cv2.rectangle(
            img,
            left_top, right_bottom,
            (255, 0, 255),
            3
        )
        # text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        # cv2.putText(orig_img, text, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def clip_bbox(bbox_pred, orig_shape):
    """Clip outside of bounding boxes.

    Args:
        bbox_pred(array): Shape is (N, 4). (left_x, top_y, right_x, bottom_y)
        orig_shape: (H, W)
    """
    bbox_pred[:, ::2] = np.clip(bbox_pred[:, ::2], 0, orig_shape[1])
    bbox_pred[:, 1::2] = np.clip(bbox_pred[:, 1::2], 0, orig_shape[0])
    return bbox_pred

def main():
    demo_yolov2()

if __name__ == '__main__':
    main()
