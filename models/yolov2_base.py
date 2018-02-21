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

def parse_dict(dic, key, value=None):
    return value if dic is None or not key in dic else dic[key]


class YOLOv2_base(chainer.Chain):
    """Implementation of YOLOv2(416*416).
    """
    def __init__(self, config, pretrained_model=None):
        super(YOLOv2_base, self).__init__()
        self.n_boxes = config['n_boxes']
        self.n_classes = config['n_classes']
        self.anchors = parse_dict(config, "anchors")
        self.object_scale = parse_dict(config, "object_scale")
        self.noobject_scale = parse_dict(config, "noobject_scale")
        self.coord_scale = parse_dict(config, "coord_scale")
        self.class_scale = parse_dict(config, 'class_scale')
        self.best_iou_thresh = parse_dict(config, 'best_iou_thresh')

        self.thresh = parse_dict(config, "thresh")
        self.nms_thresh = parse_dict(config, "nms_thresh")
        self.dim = parse_dict(config, 'dim')
        self.width = parse_dict(config, "width")
        self.height = parse_dict(config, "height")
        self.nms = parse_dict(config, 'nms')
        self.regularize_box = parse_dict(config, 'regularize_box')
        self.regularize_bn = parse_dict(config, 'regularize_bn')
        self.seen_thresh = parse_dict(config, 'seen_thresh')
        self.seen = 0

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

        if parse_dict(pretrained_model, 'download'):
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if parse_dict(pretrained_model, 'path'):
            chainer.serializers.load_npz(pretrained_model['path'], self)

        if self.regularize_bn:
            layers = list(self._children)
            self.layer_bn_list = [layer for layer in layers if "bn" in layer]

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

    def calc_iou_anchor_gt(self, bbox_pred_x, bbox_pred_y, bbox_pred_w, bbox_pred_h,
                           gt_boxes, gt_labels, gmap,
                           num_labels,
                           x_shift, y_shift, w_anchor, h_anchor,
                           out_h, out_w, tx, ty, tw, th, tconf, tprob,
                           coord_scale_array, conf_scale_array):
        """
        Args:
            pred_x(array): Shape is (B, n_boxes, out_h, out_w, 4)
                           Shape is (n_boxes * out_h * out_w, 4)
            gt_boxes(array): Shape is (B, target, 4)

        Returns:
            Shape is (B * target)
        """
        batchsize = int(gmap.shape[0])
        num_labels = chainer.cuda.to_cpu(num_labels)
        batch_index = self.xp.array([b for b in range(batchsize) for n in range(num_labels[b, 0])], dtype='i')
        target_index = self.xp.array([n for b in range(batchsize) for n in range(num_labels[b, 0])], dtype='i')
        label_index = self.xp.array([n for b in range(batchsize) for n in gt_labels[b, :num_labels[b, 0]]], dtype='i')
        num_positive = len(label_index)
        each_indexes = gmap[batch_index, target_index]
        x_index = each_indexes[:, 0]
        y_index = each_indexes[:, 1]
        bbox_index = each_indexes[:, 2]

        gt_boxes = gt_boxes[batch_index, target_index]
        gt_boxes_w = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_boxes_h = gt_boxes[:, 2] - gt_boxes[:, 0]
        bp_x = bbox_pred_x[batch_index, bbox_index, y_index, x_index]
        bp_y = bbox_pred_y[batch_index, bbox_index, y_index, x_index]
        bp_w = bbox_pred_w[batch_index, bbox_index, y_index, x_index]
        bp_h = bbox_pred_h[batch_index, bbox_index, y_index, x_index]

        left_x = self.xp.maximum(bp_x - bp_w / 2., gt_boxes[:, 1])
        right_x = self.xp.minimum(bp_x + bp_w / 2., gt_boxes[:, 3])
        top_y = self.xp.maximum(bp_y - bp_h / 2., gt_boxes[:, 0])
        bottom_y = self.xp.minimum(bp_y + bp_h / 2., gt_boxes[:, 2])

        intersect = self.xp.maximum(0, right_x - left_x) * self.xp.maximum(0, bottom_y - top_y)
        union = bp_w * bp_h + gt_boxes_w * gt_boxes_h
        iou = intersect / (union - intersect + 1e-3)

        tx[batch_index, bbox_index, y_index, x_index] = (gt_boxes[:, 1] + gt_boxes_w / 2.) * out_w - x_shift[batch_index, bbox_index, y_index, x_index]
        ty[batch_index, bbox_index, y_index, x_index] = (gt_boxes[:, 0] + gt_boxes_h / 2.) * out_h - y_shift[batch_index, bbox_index, y_index, x_index]
        tw[batch_index, bbox_index, y_index, x_index] = self.xp.log(gt_boxes_w * out_w / w_anchor[batch_index, bbox_index, y_index, x_index])
        th[batch_index, bbox_index, y_index, x_index] = self.xp.log(gt_boxes_h * out_h / h_anchor[batch_index, bbox_index, y_index, x_index])
        coord_scale_array[batch_index, bbox_index, y_index, x_index] = self.coord_scale * (2 - gt_boxes_h * gt_boxes_w)

        tconf[batch_index, bbox_index, y_index, x_index] = iou
        conf_scale_array[batch_index, bbox_index, y_index, x_index] = self.object_scale
        tprob[batch_index, bbox_index, y_index, x_index] = 0
        tprob[batch_index, bbox_index, y_index, x_index, label_index] = 1
        return tx, ty, tw, th, tconf, tprob, coord_scale_array, conf_scale_array, num_positive

    def calc_best_iou(self, bbox_pred_x, bbox_pred_y, bbox_pred_w, bbox_pred_h,
                      gt_boxes, conf_scale_array):
        """
        Args:
            pred_x(array): Shape is (B, n_boxes, out_h, out_w)
                                    B * n_boxes * out_h * out_w, target
            gt_boxes(array): Shape is (B, target, 4)
                                    B * n_boxes * out_h * out_w, target, 4
        """
        B, n_boxes, out_h, out_w = bbox_pred_x.shape
        num_target = gt_boxes.shape[1]
        gt_boxes = self.xp.broadcast_to(gt_boxes[:, None, None, None], (B, n_boxes, out_h, out_w, num_target, 4))

        bbox_pred_x = self.xp.broadcast_to(bbox_pred_x[:, :, :, :, None], (B, n_boxes, out_h, out_w, num_target))
        bbox_pred_y = self.xp.broadcast_to(bbox_pred_y[:, :, :, :, None], (B, n_boxes, out_h, out_w, num_target))
        bbox_pred_w = self.xp.broadcast_to(bbox_pred_w[:, :, :, :, None], (B, n_boxes, out_h, out_w, num_target))
        bbox_pred_h = self.xp.broadcast_to(bbox_pred_h[:, :, :, :, None], (B, n_boxes, out_h, out_w, num_target))

        pred_left_x = bbox_pred_x - bbox_pred_w / 2.
        pred_right_x = bbox_pred_x + bbox_pred_w / 2.
        pred_top_y = bbox_pred_y - bbox_pred_h / 2.
        pred_bottom_y = bbox_pred_y + bbox_pred_h / 2.

        left_x = self.xp.maximum(pred_left_x, gt_boxes[:, :, :, :, :, 1])
        right_x = self.xp.minimum(pred_right_x, gt_boxes[:, :, :, :, :, 3])
        top_y = self.xp.maximum(pred_top_y, gt_boxes[:, :, :, :, :, 0])
        bottom_y = self.xp.minimum(pred_bottom_y, gt_boxes[:, :, :, :, :, 2])

        intersect = self.xp.maximum(right_x - left_x, 0) * \
                        self.xp.maximum(bottom_y - top_y, 0)
        union = bbox_pred_h * bbox_pred_w + \
                    (gt_boxes[:, :, :, :, :, 2] - gt_boxes[:, :, :, :, :, 0]) * (gt_boxes[:, :, :, :, :, 3] - gt_boxes[:, :, :, :, :, 1])
        iou = intersect / (union - intersect +  1e-3)

        best_iou = self.xp.max(iou, axis=4)
        over_best_iou = (best_iou > self.best_iou_thresh)
        conf_scale_array[:] = self.noobject_scale
        conf_scale_array[over_best_iou] = 0
        return conf_scale_array

    def __call__(self, imgs, gt_boxes, gt_labels, gmap, num_labels):
        """
        Args:
            imgs(array): Shape is (B, 3, H, W)
            gt_boxes(array): Shape is (B, Max target, 4)
            gt_labels(array): Shape is (B, Max target)
        """
        output = self.model(imgs)
        N, input_channel, input_h, input_w = imgs.shape
        N, _, out_h, out_w = output.shape
        shape = (N, self.n_boxes, self.n_classes+5, out_h, out_w)
        pred_xy, pred_wh, pred_conf, pred_prob = \
            F.split_axis(F.reshape(output, shape), (2, 4, 5,), axis=2)
        pred_xy = F.sigmoid(pred_xy) # shape is (N, n_boxes, 2, out_h, out_w)
        pred_wh_exp = F.exp(pred_wh) # shape is (N, n_boxes, 2, out_h, out_w)
        pred_conf = F.sigmoid(pred_conf[:, :, 0]) # (N, n_boxes, out_h, out_w)
        pred_prob = pred_prob.transpose(0, 1, 3, 4, 2)
        pred_prob = F.softmax(pred_prob, axis=4)

        with self.xp.cuda.Device(chainer.cuda.get_device_from_array(pred_xy.data)):
            shape = (N, self.n_boxes, out_h, out_w)
            x_shift = self.xp.broadcast_to(self.xp.arange(out_w, dtype='f').reshape(1, 1, 1, out_w), shape)
            y_shift = self.xp.broadcast_to(self.xp.arange(out_h, dtype='f').reshape(1, 1, out_h, 1), shape)
            if self.anchors.ndim != 4:
                self.anchors = self.xp.array(self.anchors, dtype='f')
                self.anchors = self.xp.reshape(self.anchors, (1, self.n_boxes, 2, 1))
            w_anchor = self.xp.broadcast_to(self.anchors[:, :, :1, :], shape)
            h_anchor = self.xp.broadcast_to(self.anchors[:, :, 1:, :], shape)

            bbox_pred_x = (pred_xy[:, :, 0].data + x_shift) / out_w
            bbox_pred_y = (pred_xy[:, :, 1].data + y_shift) / out_h
            bbox_pred_w = pred_wh_exp[:, :, 0].data * w_anchor / out_w
            bbox_pred_h = pred_wh_exp[:, :, 1].data * h_anchor / out_h

            tx = self.xp.zeros((N, self.n_boxes, out_h, out_w), dtype='f')
            ty = self.xp.zeros((N, self.n_boxes, out_h, out_w), dtype='f')
            tw = self.xp.zeros((N, self.n_boxes, out_h, out_w), dtype='f')
            th = self.xp.zeros((N, self.n_boxes, out_h, out_w), dtype='f')
            tconf = self.xp.zeros((N, self.n_boxes, out_h, out_w), dtype='f')
            tprob = pred_prob.data.copy()

            coord_scale_array = self.xp.zeros((N, self.n_boxes, out_h, out_w), dtype='f')
            conf_scale_array = self.xp.zeros((N, self.n_boxes, out_h, out_w), dtype='f')

            self.seen += N
            if self.regularize_box and self.seen < self.seen_thresh:
                tx[:] = 0.5
                ty[:] = 0.5
                coord_scale_array[:] = 0.01

            conf_scale_array = \
                self.calc_best_iou(bbox_pred_x, bbox_pred_y, bbox_pred_w, bbox_pred_h,
                                   gt_boxes, conf_scale_array)

            tx, ty, tw, th, tconf, tprob, coord_scale_array, conf_scale_array, num_positive = \
                self.calc_iou_anchor_gt(bbox_pred_x, bbox_pred_y, bbox_pred_w,
                                        bbox_pred_h,
                                        gt_boxes, gt_labels, gmap, num_labels,
                                        x_shift, y_shift, w_anchor, h_anchor,
                                        out_h, out_w, tx, ty, tw, th, tconf, tprob,
                                        coord_scale_array, conf_scale_array)

            gamma_loss = 0
            if self.regularize_bn: # new feature
                for layer in self.layer_bn_list:
                    layer = getattr(self, layer)
                    gamma = layer.gamma
                    gamma_loss += F.sum(F.absolute(gamma))
                gamma_loss *= self.regularize_bn

            x_loss = F.sum(((tx - pred_xy[:, :, 0]) ** 2) * coord_scale_array) / 2.
            y_loss = F.sum(((ty - pred_xy[:, :, 1]) ** 2) * coord_scale_array) / 2.
            w_loss = F.sum(((tw - pred_wh[:, :, 0]) ** 2) * coord_scale_array) / 2.
            h_loss = F.sum(((th - pred_wh[:, :, 1]) ** 2) * coord_scale_array) / 2.
            conf_loss = F.sum(((tconf - pred_conf) ** 2) * conf_scale_array)/ 2
            prob_loss = F.sum(((tprob - pred_prob) ** 2) * self.class_scale)/ 2
            total_loss = x_loss + y_loss + w_loss + h_loss + \
                             conf_loss + prob_loss
            num_positive = max(1, num_positive)
            total_loss /= num_positive

            if not isinstance(gamma_loss, int):
                total_loss += gamma_loss
                chainer.report({'gamma': gamma}, self)
            chainer.report({'total_loss': total_loss}, self)
            chainer.report({'xy_loss': x_loss + y_loss}, self)
            chainer.report({'wh_loss': w_loss + h_loss}, self)
            chainer.report({'conf_loss': conf_loss}, self)
            chainer.report({'prob_loss': prob_loss}, self)
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
            orig_sizes[b] = [orig_h, orig_w]
            delta_sizes[b] = [delta_h, delta_w]

        input_imgs = self.xp.array(input_imgs, dtype='f')
        return input_imgs, orig_sizes, delta_sizes

    def predict(self, imgs):
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
