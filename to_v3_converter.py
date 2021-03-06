import argparse
import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F
import chainer.links as L

from models.yolov3_base import YOLOv3_base
from models.yolov3_variants import YOLOv3_update_base

def parse_args():
    parser = argparse.ArgumentParser(
                 description="Converter from darknet to Chainer")
    parser.add_argument('--model', default='pre', help='[pre, normal]')
    parser.add_argument('--orig', help="File name of darknet's model parameter")
    parser.add_argument('--name', help="File name of chainer's model parameter")
    parser.add_argument('--n_class', default=80, type=int, help="Number of class")
    parser.add_argument('--n_box', default=5, type=int, help="Number of boxes")
    parser.add_argument('--darknet', default=0, type=int)
    return parser.parse_args()


def to_chainer_converter():
    """model converter from darknet to chainer."""
    args = parse_args()
    if args.model == 'pre':
        Model = YOLOv3_base
    elif args.model == 'normal':
        Model = YOLOv3_update_base
    elif args.model == 'high':
        raise("Not Implemented Error: High resolution Yolo")
    elif args.model == 'tiny':
        raise("Not Implemented Error: Tiny Yolo")
    else:
        raise("Not Implemented Error")

    config = {'n_classes':args.n_class, 'n_boxes':args.n_box}
    pretrained_model = {'download': None, 'path': None}
    model = Model(config, pretrained_model=pretrained_model)

    with open(args.orig, 'rb') as f:
        orig_data = np.fromfile(f, dtype='f')[5:] # skip header

    i = 1
    offset = 0
    while True:
        try:
            if not i in [59, 67, 75]:
                bn = getattr(model, 'bn{}'.format(i))
                out_ch = bn.gamma.shape[0]
                bn.beta.data = orig_data[offset : offset + out_ch]
                offset += out_ch
                bn.gamma.data = orig_data[offset: offset + out_ch]
                offset += out_ch
                bn.avg_mean = orig_data[offset: offset + out_ch]
                offset += out_ch
                bn.avg_var = orig_data[offset: offset + out_ch]
                offset += out_ch

            conv = getattr(model, 'conv{}'.format(i))
            out_ch, in_ch, h, w = conv.W.shape
            if i in [59, 67, 75]:
                conv.b.data = orig_data[offset: offset+out_ch]
                offset += out_ch
            print(i, offset, offset + out_ch * in_ch * h * w, orig_data.shape)
            conv.W.data = orig_data[offset: offset + out_ch * in_ch * h * w].reshape(out_ch, in_ch, h, w)
            offset += out_ch * in_ch * h * w
            i += 1
            if args.darknet and i == 53:
                break
        except:
            print("Load last convolutional layer", offset, i)
            break


    save_name = "{}.npz".format(args.name)
    chainer.serializers.save_npz(save_name, model)
    print("Complete")

def main():
    to_chainer_converter()

if __name__ == '__main__':
    main()
