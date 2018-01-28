# distutils: language=c++
# -*- coding: utf-8 -*-

cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport log
from libc.math cimport sin, cos
from libc.math cimport abs as c_abs

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t

cdef inline DTYPE_t max_float(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline DTYPE_t min_float(np.float32_t a, np.float32_t b):
    return a if a <= b else b

cdef inline int max_int(int a, int b):
    return a if a >= b else b

cdef inline int min_int(int a, int b):
    return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def nms_by_class(np.ndarray[DTYPE_t, ndim=2] bbox_pred,
                 np.ndarray[DTYPE_t, ndim=2] prob,
                 float threshold):
    """
    Args:
        bbox_pred(array): Shape is (W*H, 4)
        prob(array)     : Shape is (W*H, 2)
    """
    cdef int num_candidate = prob.shape[0]
    cdef np.ndarray[DTYPE_int_t, ndim=1] suppressed = np.zeros((num_candidate), dtype=DTYPE_int)
    cdef int i, j

    cdef float center_x, center_y, height, width
    cdef float xx1, xx2, yy1, yy2

    cdef float com_center_x, com_center_y, com_height, com_width
    cdef float com_xx1, com_xx2, com_yy1, com_yy2
    cdef left_x, right_x, top_y, bottom_y

    result_index = []
    for i in range(num_candidate):
      if suppressed[i] == 1:
        continue
      result_index.append(i)
      center_x = bbox_pred[i, 0]
      center_y = bbox_pred[i, 1]
      width = bbox_pred[i, 2]
      height = bbox_pred[i, 3]

      xx1 = center_x - width / 2
      xx2 = center_x + width / 2
      yy1 = center_y - height / 2
      yy2 = center_y + height / 2

      for j in range(i + 1, num_candidate):
        if suppressed[j] == 1:
          continue
        com_center_x = bbox_pred[j, 0]
        com_center_y = bbox_pred[j, 1]
        com_width = bbox_pred[j, 2]
        com_height = bbox_pred[j, 3]

        com_xx1 = com_center_x - com_width / 2
        com_xx2 = com_center_x + com_width / 2
        com_yy1 = com_center_y - com_height / 2
        com_yy2 = com_center_y + com_height / 2

        left_x = max_float(xx1, com_xx1)
        right_x = min_float(xx2, com_xx2)
        top_y = max_float(yy1, com_yy1)
        bottom_y = min_float(yy2, com_yy2)

        intersection = max_float(0, right_x - left_x) * max_float(0, bottom_y - top_y)
        union = height * width + com_height * com_width

        if intersection / (union - intersection) > 0:
          suppressed[j] = 1
    return result_index


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def nms_by_obj(np.ndarray[DTYPE_t, ndim=2] bbox_pred,
                 np.ndarray[DTYPE_t, ndim=1] prob,
                 float threshold):
    """
    Args:
        bbox_pred(array): Shape is (W*H, 4)
        prob(array)     : Shape is (W*H, 2)
    """
    cdef int num_candidate = prob.shape[0]
    cdef np.ndarray[DTYPE_int_t, ndim=1] suppressed = np.zeros((num_candidate), dtype=DTYPE_int)
    cdef int i, j

    cdef float center_x, center_y, height, width
    cdef float xx1, xx2, yy1, yy2

    cdef float com_center_x, com_center_y, com_height, com_width
    cdef float com_xx1, com_xx2, com_yy1, com_yy2
    cdef left_x, right_x, top_y, bottom_y

    result_index = []
    for i in range(num_candidate):
      if suppressed[i] == 1:
        continue
      result_index.append(i)
      center_x = bbox_pred[i, 0]
      center_y = bbox_pred[i, 1]
      width = bbox_pred[i, 2]
      height = bbox_pred[i, 3]

      xx1 = center_x - width / 2
      xx2 = center_x + width / 2
      yy1 = center_y - height / 2
      yy2 = center_y + height / 2

      for j in range(i + 1, num_candidate):
        if suppressed[j] == 1:
          continue
        com_center_x = bbox_pred[j, 0]
        com_center_y = bbox_pred[j, 1]
        com_width = bbox_pred[j, 2]
        com_height = bbox_pred[j, 3]

        com_xx1 = com_center_x - com_width / 2
        com_xx2 = com_center_x + com_width / 2
        com_yy1 = com_center_y - com_height / 2
        com_yy2 = com_center_y + com_height / 2

        left_x = max_float(xx1, com_xx1)
        right_x = min_float(xx2, com_xx2)
        top_y = max_float(yy1, com_yy1)
        bottom_y = min_float(yy2, com_yy2)

        intersection = max_float(0, right_x - left_x) * max_float(0, bottom_y - top_y)
        union = height * width + com_height * com_width

        if intersection / (union - intersection) > threshold:
          suppressed[j] = 1
    return result_index
