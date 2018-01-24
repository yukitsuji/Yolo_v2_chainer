#/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import numpy as np
import os
import sys
import subprocess
import time
try:
    import matplotlib.pyplot as plt
except:
    pass
import cv2
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from models.yolov2_base import YOLOv2_base

def create_timer():
    start = chainer.cuda.Event()
    stop = chainer.cuda.Event()
    start.synchronize()
    start.record()
    return start, stop

def print_timer(start, stop, sentence="Time"):
    stop.record()
    stop.synchronize()
    elapsed_time = chainer.cuda.cupy.cuda.get_elapsed_time(
                           start, stop) / 1000
    print(sentence, elapsed_time)
    return elapsed_time

def parse_dic(dic, key):
    return None if dic is None or not key in dic else dic[key]


class YOLOv2_update_base(YOLOv2_base):
    """Implementation of update version of YOLOv2(416*416).
    """
    def __init__(self, config, pretrained_model=None):
        super(YOLOv2_update_base, self).__init__(config)
        self.n_boxes = config['n_boxes']
        self.n_classes = config['n_classes']
        self.anchors = parse_dic(config, "anchors")
        self.object_scale = parse_dic(config, "object_scale")
        self.nonobject_scale = parse_dic(config, "nonobject_scale")
        self.coord_scale = parse_dic(config, "coord_scale")
        self.thresh = parse_dic(config, "thresh")

        with self.init_scope():
            delattr(self, 'conv21'); delattr(self, 'bn21')
            self.conv21 = L.Convolution2D(512, 64, ksize=1,
                                          stride=1, pad=0, nobias=True)
            self.bn21   = L.BatchNormalization(64)
            # 1024 + 256 = 1280
            delattr(self, 'conv22')
            self.conv22 = L.Convolution2D(1280, 1024, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn22   = L.BatchNormalization(1024)
            out_ch = self.n_boxes * (5 + self.n_classes)
            self.conv23 = L.Convolution2D(1024, out_ch, ksize=1, stride=1, pad=0)

        if parse_dic(pretrained_model, 'download'):
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if parse_dic(pretrained_model, 'path'):
            chainer.serializers.load_npz(pretrained_model['path'], self)

    def model(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.1)
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.1)
        h = F.leaky_relu(self.bn5(self.conv5(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn6(self.conv6(h)), slope=0.1)
        h = F.leaky_relu(self.bn7(self.conv7(h)), slope=0.1)
        h = F.leaky_relu(self.bn8(self.conv8(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn9(self.conv9(h)), slope=0.1)
        h = F.leaky_relu(self.bn10(self.conv10(h)), slope=0.1)
        h = F.leaky_relu(self.bn11(self.conv11(h)), slope=0.1)
        h = F.leaky_relu(self.bn12(self.conv12(h)), slope=0.1)
        h = F.leaky_relu(self.bn13(self.conv13(h)), slope=0.1)
        high_resolution_feature = F.leaky_relu(self.bn21(self.conv21(h)), slope=0.1)
        high_resolution_feature = reorg(high_resolution_feature)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn14(self.conv14(h)), slope=0.1)
        h = F.leaky_relu(self.bn15(self.conv15(h)), slope=0.1)
        h = F.leaky_relu(self.bn16(self.conv16(h)), slope=0.1)
        h = F.leaky_relu(self.bn17(self.conv17(h)), slope=0.1)
        h = F.leaky_relu(self.bn18(self.conv18(h)), slope=0.1)

        h = F.leaky_relu(self.bn19(self.conv19(h)), slope=0.1)
        h = F.leaky_relu(self.bn20(self.conv20(h)), slope=0.1)

        h = F.concat((high_resolution_feature, h), axis=1)
        h = F.leaky_relu(self.bn22(self.conv22(h)), slope=0.1)
        return self.conv23(h)
