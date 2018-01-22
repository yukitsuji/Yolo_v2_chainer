import argparse
import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F
import chainer.links as L

from models.yolov2_base import YOLOv2

args = parser.parse_args()

def parse_arg():
    parser = argparse.ArgumentParser(
                 description="Converter from darknet to Chainer")
    parser.add_argument('--model', default='normal', help='[normal, high, tiny]')
    parser.add_argument('--orig', help="File name of darknet's model parameter")
    parser.add_argument('--name', help="File name of chainer's model parameter")
    parser.add_argument('--n_class', default=80, type=int, help="Number of class")
    parser.add_argument('--n_box', default=5, type=int, help="Number of boxes")
    return parser.parse_args()

def to_chainer_converter():
    """model converter from darknet to chainer."""
    args = parse_args()
    if args.model == 'normal':
        Model = Yolov2
    elif args.model == 'high':
        raise("Not Implemented Error: High resolution Yolo")
    else:
        raise("Not Implemented Error: Tiny Yolo")

    model = Model(**args)
    with open(args.orig, 'rb') as f:
        orig_data = np.fromfile(f, dtype='f')[4:] # skip header
    chainer.serializers.save_npz(os.path.join(config['results'], 'model.npz'),
                                 model)

def main():
    to_chainer_converter()

if __name__ == '__main__':
    main()

#
# layers=[
#     [3, 32, 3],
#     [32, 64, 3],
#     [64, 128, 3],
#     [128, 64, 1],
#     [64, 128, 3],
#     [128, 256, 3],
#     [256, 128, 1],
#     [128, 256, 3],
#     [256, 512, 3],
#     [512, 256, 1],
#     [256, 512, 3],
#     [512, 256, 1],
#     [256, 512, 3],
#     [512, 1024, 3],
#     [1024, 512, 1],
#     [512, 1024, 3],
#     [1024, 512, 1],
#     [512, 1024, 3],
#     [1024, 1024, 3],
#     [1024, 1024, 3],
#     [3072, 1024, 3],
# ]
#
# offset=0
# for i, l in enumerate(layers):
#     in_ch = l[0]
#     out_ch = l[1]
#     ksize = l[2]
#
#     # load bias(Bias.bはout_chと同じサイズ)
#     txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
#     offset += out_ch
#     exec(txt)
#
#     # load bn(BatchNormalization.gammaはout_chと同じサイズ)
#     txt = "yolov2.bn%d.gamma.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
#     offset += out_ch
#     exec(txt)
#
#     # (BatchNormalization.avg_meanはout_chと同じサイズ)
#     txt = "yolov2.bn%d.avg_mean = dat[%d:%d]" % (i+1, offset, offset+out_ch)
#     offset += out_ch
#     exec(txt)
#
#     # (BatchNormalization.avg_varはout_chと同じサイズ)
#     txt = "yolov2.bn%d.avg_var = dat[%d:%d]" % (i+1, offset, offset+out_ch)
#     offset += out_ch
#     exec(txt)
#
#     # load convolution weight(Convolution2D.Wは、outch * in_ch * フィルタサイズ。
#     # これを(out_ch, in_ch, 3, 3)にreshapeする)
#     txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % \
#               (i+1, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
#     offset += (out_ch*in_ch*ksize*ksize)
#     exec(txt)
#     print(i+1, offset)
#
# # load last convolution weight(BiasとConvolution2Dのみロードする)
# in_ch = 1024
# out_ch = last_out
# ksize = 1
#
# txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i+2, offset, offset+out_ch)
# offset += out_ch
# exec(txt)
#
# txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (i+2, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
# offset += out_ch*in_ch*ksize*ksize
# exec(txt)
# print(i+2, offset)
#
# print("save weights file to yolov2_darknet.model")
# serializers.save_hdf5("yolov2_darknet.model", yolov2)
