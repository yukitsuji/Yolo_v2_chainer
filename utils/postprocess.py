#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.cython_util.nms_by_class import nms_by_class, nms_by_obj

def select_bbox_by_class(bbox_pred, conf, prob, thresh, nms_thresh):
    prob_shape = prob.shape
    prob = np.broadcast_to(conf[:, None], prob_shape) * prob
    prob = prob.transpose(1, 0)
    sort_index = np.argsort(prob, axis=1)[:, ::-1].astype(np.int32)
    prob = prob.transpose(1, 0)
    sort_index = sort_index.transpose(1, 0)
    index = nms_by_class(bbox_pred, prob, sort_index, thresh, nms_thresh)
    index = np.asarray(index, dtype='i')
    if len(index):
        bbox_pred = bbox_pred[index[:, 0]]
        prob = prob[index[:, 0], index[:, 1]]
        cls_inds = index[:, 1]
        return bbox_pred, prob, cls_inds, index
    else:
        return [], [], [], []

def select_bbox_by_obj(bbox_pred, conf, prob, thresh, nms_thrsh):
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
    if len(index):
        bbox_pred = bbox_pred[index]
        prob = prob[index]
        cls_inds = cls_inds[index]
        return bbox_pred, prob, cls_inds, None
    else:
        return [], [], [], []

def visualize_with_label(img, bbox_pred, prob, names, ax=None, index=None):
    """Visualize image with labels."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img.astype(np.uint8))
    duplicate = np.array([])
    for i, (bbox, p, name) in enumerate(zip(bbox_pred, prob, names)):
        if i in duplicate:
            continue
        xy = (bbox[0], bbox[1])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=3))
        caption = []
        caption.append(name)
        caption.append('{:.2f}'.format(p))
        if index is not None:
            same_place = index[i+1:] == index[i]
            if same_place.any():
                same_place = np.arange(i+1, index.shape[0])[same_place]
                duplicate = np.union1d(duplicate, same_place)
                for j in same_place:
                    caption.append(names[j])
                    caption.append('{:.2f}'.format(prob[j]))

        y = xy[1] + 34 if xy[1] - 34 < 0 else xy[1]
        ax.text(xy[0], y, ': '.join(caption), style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax


def clip_bbox(bbox_pred, orig_shape):
    """Clip outside of bounding boxes.

    Args:
        bbox_pred(array): Shape is (N, 4). (left_x, top_y, right_x, bottom_y)
        orig_shape: (H, W)
    """
    bbox_pred[:, ::2] = np.clip(bbox_pred[:, ::2], 0, orig_shape[1])
    bbox_pred[:, 1::2] = np.clip(bbox_pred[:, 1::2], 0, orig_shape[0])
    return bbox_pred
