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
from utils.postprocess import select_bbox_by_class, select_bbox_by_obj
from utils.postprocess import clip_bbox, xywh_to_xyxy, xyxy_to_yxyx
from utils.timer import create_timer, print_timer


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
        self.nms_thresh = parse_dic(config, "nms_thresh")
        self.width = parse_dic(config, "width")
        self.height = parse_dic(config, "height")
        self.widthes = parse_dic(config, "widthes")
        self.heightes = parse_dic(config, "heightes")
        self.nms = parse_dic(config, 'nms')

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


    def get_region_box(self, xy, wh, x_shift, y_shift, out_h, out_w):
        x = (xy[:, :, 0] + x_shift) / out_w
        y = (xy[:, :, 1] + y_shift) / out_h
        w = wh[:, :, 0] * 0.5 / out_w
        h = wh[:, :, 1] * 0.5 / out_h
        return x, y, w, h

    def calc_best_iou(pred_x, pred_y, pred_w, pred_h, gt_boxes):
        """
        Args:
            pred_x(array): Shape is (B, n_boxes, out_h, out_w)
        """
        left_x = pred_x - pred_w / 2.
        right_x = pred_x + pred_w / 2.
        left_x = self.xp.maximum(pred_x - pred_w / 2., gt_boxes[:, ])


    def __call__(self, imgs, gt_boxes, gt_labels):
        """
        Args:
            imgs(array): Shape is (B, 3, H, W)
            gt_boxes(array): Shape is (B, Max target, 4)
            gt_labels(array): Shape is (B, Max target)
        """
        output = self.model(imgs).data
        N, input_channel, input_h, input_w = imgs.shape
        N, _, out_h, out_w = output.shape
        shape = (N, self.n_boxes, self.n_classes+5, out_h, out_w)
        pred_xy, pred_wh, pred_conf, pred_prob = \
            F.split_axis(F.reshape(output, shape), (2, 4, 5,), axis=2)
        pred_xy = F.sigmoid(xy) # shape is (N, n_boxes, 2, out_h, out_w)
        pred_wh = F.exp(wh) # shape is (N, n_boxes, 2, out_h, out_w)
        pred_conf = F.sigmoid(pred_conf[:, :, 0])
        pred_prob = pred_prob.transpose(0, 1, 3, 4, 2)
        pred_prob = F.softmax(pred_prob, axis=4)

        shape = (N, self.n_boxes, out_h, out_w)
        x_shift = self.xp.broadcast_to(self.xp.arange(out_w, dtype='f').reshape(1, 1, 1, out_w), shape)
        y_shift = self.xp.broadcast_to(self.xp.arange(out_h, dtype='f').reshape(1, 1, out_h, 1), shape)
        if self.anchors.ndim != 4:
            n_device = chainer.cuda.get_device_from_array(output)
            if n_device.id != -1:
                self.anchors = chainer.cuda.to_gpu(self.anchors, device=n_device)
            self.anchors = self.xp.reshape(self.anchors, (1, self.n_boxes, 2, 1))
        w_anchor = self.xp.broadcast_to(self.anchors[:, :, :1, :], shape)
        h_anchor = self.xp.broadcast_to(self.anchors[:, :, 1:, :], shape)

        # Compare bounding boxes between predictions and groundtruthes
        # if the best match iou between them is over best_iou's threshold,
        # loss is 0
        # And if number of iteration is under assigned value,
        # regularization of bounding box regression would be applyed.
        pred_x, pred_y, pred_w, pred_h = \
            self.get_region_box(self, pred_xy, pred_wh,
                                x_shift, y_shift, w_anchor, h_anchor)

        best_iou, best_iou_index = calc_best_iou(pred_x.data, pred_y.data,
                                                 pred_w.data, pred_h.data,
                                                 gt_boxes)
        self.nonobject_scale * (0 - pred_conf)
        if best_iou > self.best_iou_thresh:
            pass
        # Compare each groundtruth boxes

        # もう１つの方は、anchorとground truthを事前に比較しておき、各場所(i, j)毎に、
        # anchor boxとgroundtruth比較して、最もマッチしたものをGroundTruthとして計算する。
        #


        bbox_pred = self.xp.zeros((N, self.n_boxes, out_h, out_w, 4), 'f')
        # bbox_pred[:, :, :, :, 0] = (xy[:, :, 0] + x_shift) / out_w * img_shape[1]
        # bbox_pred[:, :, :, :, 1] = (xy[:, :, 1] + y_shift) / out_h * img_shape[0]
        # bbox_pred[:, :, :, :, 2] = wh[:, :, 0] * w_anchor / out_w * img_shape[1]
        # bbox_pred[:, :, :, :, 3] = wh[:, :, 1] * h_anchor / out_h * img_shape[0]
        return total_loss

    def prepare(self, imgs):
        batchsize = len(imgs)
        input_imgs = np.zeros((batchsize, 3, self.height, self.width), 'f')
        input_imgs += 0.5
        orig_sizes = np.zeros((batchsize, 2), dtype='f')
        delta_sizes = np.zeros((batchsize, 2), dtype='f')
        for b in range(batchsize):
            img = imgs[b]
            _, orig_h, orig_w = img.shape
            if (orig_h / self.height) > (orig_w / self.width):
                new_h = self.height
                new_w = int((orig_w * self.height) / orig_h)
            else:
                new_w = self.width
                new_h = int((orig_h * self.width) / orig_w)

            img = F.resize_images(img[np.newaxis, :], (new_h, new_w)).data
            delta_h = int(abs((new_h - self.height) / 2))
            delta_w = int(abs((new_w - self.width) / 2))
            img /= 255.
            input_imgs[b, :, delta_h:new_h+delta_h, delta_w:new_w+delta_w] = img
            orig_sizes[b] = [orig_h, orig_w] # TODO
            delta_sizes[b] = [delta_h, delta_w] # TODO

        input_imgs = self.xp.array(input_imgs, dtype='f')
        return input_imgs, orig_sizes, delta_sizes

    def evaluation(self, imgs):
        """Inference.

        Args:
            imgs(array): Shape is (N, 3, H, W)
            img_shape: (H, W)

        Returns:
            bbox_pred(array): Shape is (1, box * out_h * out_w, 4)
            conf(array): Shape is (1, box * out_h * out_w)
            prob(array): Shape is (1, box * out_h * out_w, n_class)
        """
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():

            # Prepare images for model.
            input_imgs, orig_sizes, delta_sizes = self.prepare(imgs)
            bbox_pred, conf, prob = self.inference(input_imgs,
                                                   (self.height, self.width))
            batchsize = len(input_imgs)
            bbox_pred = bbox_pred.reshape(batchsize, -1, 4)
            conf = conf.reshape(batchsize, -1)
            prob = prob.reshape(batchsize, -1, self.n_classes)
            bbox_preds = chainer.cuda.to_cpu(bbox_pred)
            confs = chainer.cuda.to_cpu(conf)
            probs = chainer.cuda.to_cpu(prob)

            bboxes, labels, scores = [], [], []
            for bbox_pred, conf, prob, orig_size, delta_size in \
                zip(bbox_preds, confs, probs, orig_sizes, delta_sizes):
                # Post processing
                if self.nms == 'class':
                    bbox_pred, prob, cls_inds, index = \
                        select_bbox_by_class(bbox_pred, conf, prob,
                                             self.thresh, self.nms_thresh)
                else:
                    bbox_pred, prob, cls_inds, index = \
                        select_bbox_by_obj(bbox_pred, conf, prob,
                                           self.thresh, self.nms_thresh)
                if len(bbox_pred):
                    bbox_pred = xywh_to_xyxy(bbox_pred)
                    bbox_pred[:, ::2] -= delta_size[1]
                    bbox_pred[:, 1::2] -= delta_size[0]
                    # expand to original size
                    if orig_size[0] < orig_size[1]:
                        expand = orig_size[1] / self.width
                    else:
                        expand = orig_size[0] / self.height
                    bbox_pred *= expand
                    # Clip
                    bbox_pred = clip_bbox(bbox_pred, orig_size)
                    # convert (x, y) to (y, x)
                    bbox_pred_yx = xyxy_to_yxyx(bbox_pred)
                else:
                    bbox_pred_yx = [[]]
                    labels = []
                    prob = []
                bboxes.append(bbox_pred_yx)
                labels.append(cls_inds)
                scores.append(prob)
            return bboxes, labels, scores

    def inference(self, imgs, img_shape):
        """Inference.

        Args:
            imgs(array): Shape is (1, 3, H, W)
            img_shape: (H, W)

        Returns:
            bbox_pred(array): Shape is (1, box * out_h * out_w, 4)
            conf(array): Shape is (1, box * out_h * out_w)
            prob(array): Shape is (1, box * out_h * out_w, n_class)
        """
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            output = self.model(imgs).data
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
                n_device = chainer.cuda.get_device_from_array(output)
                if n_device.id != -1:
                    self.anchors = chainer.cuda.to_gpu(self.anchors, device=n_device)
                self.anchors = self.xp.reshape(self.anchors, (1, self.n_boxes, 2, 1))
            w_anchor = self.xp.broadcast_to(self.anchors[:, :, :1, :], shape)
            h_anchor = self.xp.broadcast_to(self.anchors[:, :, 1:, :], shape)
            bbox_pred = self.xp.zeros((N, self.n_boxes, out_h, out_w, 4), 'f')
            bbox_pred[:, :, :, :, 0] = (xy[:, :, 0] + x_shift) / out_w * img_shape[1]
            bbox_pred[:, :, :, :, 1] = (xy[:, :, 1] + y_shift) / out_h * img_shape[0]
            bbox_pred[:, :, :, :, 2] = wh[:, :, 0] * w_anchor / out_w * img_shape[1]
            bbox_pred[:, :, :, :, 3] = wh[:, :, 1] * h_anchor / out_h * img_shape[0]
            conf = F.sigmoid(conf[:, :, 0]).data
            prob = prob.transpose(0, 1, 3, 4, 2)
            prob = F.softmax(prob, axis=4).data
            return bbox_pred, conf, prob
