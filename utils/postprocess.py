#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from utils.cython_util import nms

def nms_by_class(bbox_pred, prob):
    """NMS by each class.

    Args:
        bbox_pred(array): Shape is (N, 4)
        prob(array): Shape is (N, class, 4)
    Returns:

    """
    pass
