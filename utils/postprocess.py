#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def nms_by_class(bbox_pred, prob):
    """NMS by each class.

    Args:
        bbox_pred(array): Shape is (N, 4)
        prob(array): Shape is (N, class, 4)
    Returns:

    """
    pass

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
