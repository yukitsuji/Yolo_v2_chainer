import numpy

import chainer
from chainer import function
from chainer.utils import argument
from chainer.utils import type_check
from chainer import cuda

class Reorg(function.Function):

    def check_type_forward(self, in_types):
        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.char == 'f',
            x_type.ndim == 4,
        )

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        B, C, H, W = x.shape
        out_h = int(H // 2)
        out_w = int(W // 2)
        x = x.reshape(B, C, out_h, 2, out_w, 2)
        x = x.transpose(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, C, out_h, out_w, -1)
        x = x.transpose(0, 4, 1, 2, 3)
        return x.reshape(B, -1, out_h, out_w),

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        xp = cuda.get_array_module(gy)
        B, C, H, W = gy.shape
        out_h = H * 2
        out_w = W * 2
        gy = gy.reshape(B, 4, -1, H, W)
        gy = gy.transpose(0, 2, 3, 4, 1)
        gy = gy.reshape(B, -1, out_h, out_w, 2, 2)
        gy = gy.transpose(0, 1, 2, 4, 3, 5)
        return gy.reshape(B, -1, out_h, out_w),


def reorg(x, **kwargs):
    return Reorg()(x,)
