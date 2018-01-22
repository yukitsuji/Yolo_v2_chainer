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


class YOLOv2(Chain):
    """Implementation of YOLOv2(416*416).
    """
    def __init__(self, n_classes=80, n_boxes=5, **kwargs):
        super(YOLOv2, self).__init__()
        self.n_boxes = n_boxes
        self.n_classes = n_classes

        with self.init_scope():
            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1),
            bn1    = L.BatchNormalization(32),
            conv2  = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1),
            bn2    = L.BatchNormalization(64),
            conv3  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1),
            bn3    = L.BatchNormalization(128),
            conv4  = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0),
            bn4    = L.BatchNormalization(64),
            conv5  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1),
            bn5    = L.BatchNormalization(128),
            conv6  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            bn6    = L.BatchNormalization(256),
            conv7  = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0),
            bn7    = L.BatchNormalization(128),
            conv8  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            bn8    = L.BatchNormalization(256),
            conv9  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn9    = L.BatchNormalization(512),
            conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn10   = L.BatchNormalization(256),
            conv11 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn11   = L.BatchNormalization(512),
            conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn12   = L.BatchNormalization(256),
            conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn13   = L.BatchNormalization(512),
            conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn14   = L.BatchNormalization(1024),
            conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0),
            bn15   = L.BatchNormalization(512),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn16   = L.BatchNormalization(1024),
            conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0),
            bn17   = L.BatchNormalization(512),
            conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn18   = L.BatchNormalization(1024),

            conv19 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),
            bn19   = L.BatchNormalization(1024),
            conv20 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),
            bn20   = L.BatchNormalization(1024),
            conv21 = L.Convolution2D(3072, 1024, ksize=3, stride=1, pad=1),
            bn21   = L.BatchNormalization(1024),
            out_ch = n_boxes * (5 + n_classes)
            conv22 = L.Convolution2D(1024, out_ch, ksize=1, stride=1, pad=0)

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
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
        high_resolution_feature = reorg(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn14(self.conv14(h)), slope=0.1)
        h = F.leaky_relu(self.bn15(self.conv15(h)), slope=0.1)
        h = F.leaky_relu(self.bn16(self.conv16(h)), slope=0.1)
        h = F.leaky_relu(self.bn17(self.conv17(h)), slope=0.1)
        h = F.leaky_relu(self.bn18(self.conv18(h)), slope=0.1)

        h = F.leaky_relu(self.bn19(self.conv19(h)), slope=0.1)
        h = F.leaky_relu(self.bn20(self.conv20(h)), slope=0.1)
        h = F.concat((high_resolution_feature, h), axis=1)
        h = F.leaky_relu(self.bn21(self.conv21(h)), slope=0.1)
        return self.conv22(h)

    def __call__(self, imgs, gt_boxes, gt_labels):
        output = self.model(imgs)
        return total_loss

    def inference(self, imgs):
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            start, stop = create_timer()
            output = self.model(imgs)
            print_timer(start, stop, sentence="Inference Time")
            return None
