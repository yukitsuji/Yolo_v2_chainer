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
from models.reorg_layer import reorg

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


class YOLOv2_base(chainer.Chain):
    """Implementation of YOLOv2(416*416).
    """
    def __init__(self, config, pretrained_model=None):
        super(YOLOv2_base, self).__init__()
        self.n_boxes = config['n_boxes']
        self.n_classes = config['n_classes']
        self.anchors = parse_dic(config, "anchors")
        self.object_scale = parse_dic(config, "object_scale")
        self.nonobject_scale = parse_dic(config, "nonobject_scale")
        self.coord_scale = parse_dic(config, "coord_scale")
        self.thresh = parse_dic(config, "thresh")

        if self.anchors:
            self.anchors = np.array(self.anchors, 'f').reshape(-1, 2)

        with self.init_scope():
            self.conv1  = L.Convolution2D(3, 32, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn1    = L.BatchNormalization(32)
            self.conv2  = L.Convolution2D(32, 64, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn2    = L.BatchNormalization(64)
            self.conv3  = L.Convolution2D(64, 128, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn3    = L.BatchNormalization(128)
            self.conv4  = L.Convolution2D(128, 64, ksize=1, stride=1,
                                          pad=0, nobias=True)
            self.bn4    = L.BatchNormalization(64)
            self.conv5  = L.Convolution2D(64, 128, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn5    = L.BatchNormalization(128)
            self.conv6  = L.Convolution2D(128, 256, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn6    = L.BatchNormalization(256)
            self.conv7  = L.Convolution2D(256, 128, ksize=1, stride=1,
                                          pad=0, nobias=True)
            self.bn7    = L.BatchNormalization(128)
            self.conv8  = L.Convolution2D(128, 256, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn8    = L.BatchNormalization(256)
            self.conv9  = L.Convolution2D(256, 512, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn9    = L.BatchNormalization(512)
            self.conv10 = L.Convolution2D(512, 256, ksize=1, stride=1,
                                          pad=0, nobias=True)
            self.bn10   = L.BatchNormalization(256)
            self.conv11 = L.Convolution2D(256, 512, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn11   = L.BatchNormalization(512)
            self.conv12 = L.Convolution2D(512, 256, ksize=1, stride=1,
                                          pad=0, nobias=True)
            self.bn12   = L.BatchNormalization(256)
            self.conv13 = L.Convolution2D(256, 512, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn13   = L.BatchNormalization(512)
            self.conv14 = L.Convolution2D(512, 1024, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn14   = L.BatchNormalization(1024)
            self.conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1,
                                          pad=0, nobias=True)
            self.bn15   = L.BatchNormalization(512)
            self.conv16 = L.Convolution2D(512, 1024, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn16   = L.BatchNormalization(1024)
            self.conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1,
                                          pad=0, nobias=True)
            self.bn17   = L.BatchNormalization(512)
            self.conv18 = L.Convolution2D(512, 1024, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn18   = L.BatchNormalization(1024)

            self.conv19 = L.Convolution2D(1024, 1024, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn19   = L.BatchNormalization(1024)
            self.conv20 = L.Convolution2D(1024, 1024, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn20   = L.BatchNormalization(1024)
            self.conv21 = L.Convolution2D(3072, 1024, ksize=3,
                                          stride=1, pad=1, nobias=True)
            self.bn21   = L.BatchNormalization(1024)
            out_ch = self.n_boxes * (5 + self.n_classes)
            self.conv22 = L.Convolution2D(1024, out_ch, ksize=1, stride=1, pad=0)

        if parse_dic(pretrained_model, 'download'):
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if parse_dic(pretrained_model, 'path'):
            chainer.serializers.load_npz(pretrained_model['path'], self)

    def model(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.1)
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.1)
        h = F.leaky_relu(self.bn5(self.conv5(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.bn6(self.conv6(h)), slope=0.1)
        h = F.leaky_relu(self.bn7(self.conv7(h)), slope=0.1)
        h = F.leaky_relu(self.bn8(self.conv8(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.bn9(self.conv9(h)), slope=0.1)
        h = F.leaky_relu(self.bn10(self.conv10(h)), slope=0.1)
        h = F.leaky_relu(self.bn11(self.conv11(h)), slope=0.1)
        h = F.leaky_relu(self.bn12(self.conv12(h)), slope=0.1)
        h = F.leaky_relu(self.bn13(self.conv13(h)), slope=0.1)
        high_resolution_feature = reorg(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
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

    def __call__(self, imgs, gt_boxes, gt_labels): # TODO
        output = self.model(imgs)
        return total_loss

    def inference(self, imgs, orig_shape):
        """Inference.

        Args:
            imgs(array): Shape is (1, 3, H, W)

        Returns:
            bbox_pred(array): Shape is (1, box * out_h * out_w, 4)
            conf(array): Shape is (1, box * out_h * out_w)
            prob(array): Shape is (1, box * out_h * out_w, n_class)
        """
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            # start, stop = create_timer()
            output = self.model(imgs).data
            # print_timer(start, stop, sentence="Model inference time")
            # start, stop = create_timer()
            N, input_channel, input_h, input_w = imgs.shape
            N, _, out_h, out_w = output.shape
            shape = (N, self.n_boxes, self.n_classes+5, out_h, out_w)
            xy, wh, conf, prob = self.xp.split(self.xp.reshape(output, shape), (2, 4, 5,), axis=2)
            xy = F.sigmoid(xy).data # shape is (N, n_boxes, 2, out_h, out_w)
            wh = F.exp(wh).data # shape is (N, n_boxes, 2, out_h, out_w)
            shape = (N, self.n_boxes, out_h, out_w)
            x_shift = self.xp.broadcast_to(self.xp.arange(out_w, dtype='f').reshape(1, 1, 1, out_w), shape)
            y_shift = self.xp.broadcast_to(self.xp.arange(out_h, dtype='f').reshape(1, 1, out_h, 1), shape)
            if self.anchors.ndim != 4:
                n_device = chainer.cuda.get_device(output)
                self.anchors = chainer.cuda.to_gpu(self.anchors, device=n_device)
                self.anchors = self.xp.reshape(self.anchors, (1, self.n_boxes, 2, 1))
            w_anchor = self.xp.broadcast_to(self.anchors[:, :, :1, :], shape)
            h_anchor = self.xp.broadcast_to(self.anchors[:, :, 1:, :], shape)
            bbox_pred = self.xp.zeros((N, self.n_boxes, out_h, out_w, 4), 'f')
            bbox_pred[:, :, :, :, 0] = (xy[:, :, 0] + x_shift) / out_w * orig_shape[1]
            bbox_pred[:, :, :, :, 1] = (xy[:, :, 1] + y_shift) / out_h * orig_shape[0]
            bbox_pred[:, :, :, :, 2] = wh[:, :, 0] * w_anchor / out_w * orig_shape[1]
            bbox_pred[:, :, :, :, 3] = wh[:, :, 1] * h_anchor / out_h * orig_shape[0]
            conf = F.sigmoid(conf[:, :, 0]).data # shape is (N, n_boxes, out_h, out_w)
            prob = prob.transpose(0, 1, 3, 4, 2)
            prob = F.softmax(prob, axis=4).data # shape is (N, n_boxes, out_h, out_w, n_classes)
            # print_timer(start, stop, sentence="Inference time")
            # print_timer(start, stop, sentence="Post processing time")
            bbox_pred = bbox_pred.reshape(-1, 4)
            conf = conf.reshape(-1)
            prob = prob.reshape(-1, self.n_classes)
            return bbox_pred, conf, prob
