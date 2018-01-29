from __future__ import division

import argparse
import subprocess
import sys
import time
import numpy as np

import chainer
from chainer import iterators

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.evaluations import eval_detection_voc
from chainercv.links import SSD300
from chainercv.utils import apply_prediction_to_iterator

subprocess.call(['sh', 'setup.sh'])

from config_utils import *
from models.yolov2_base import *
from utils.image_loader import load_img_like_darknet
from utils.postprocess import visualize_with_label, clip_bbox
from utils.postprocess import select_bbox_by_class, select_bbox_by_obj

chainer.cuda.set_max_workspace_size(chainer.cuda.get_max_workspace_size())
os.environ["CHAINER_TYPE_CHECK"] = "0"
chainer.global_config.autotune = True
chainer.global_config.type_check = False

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))


class ProgressHook(object):

    def __init__(self, n_total):
        self.n_total = n_total
        self.start = time.time()
        self.n_processed = 0

    def __call__(self, imgs, pred_values, gt_values):
        self.n_processed += len(imgs)
        fps = self.n_processed / (time.time() - self.start)
        sys.stdout.write(
            '\r{:d} of {:d} images, {:.2f} FPS'.format(
                self.n_processed, self.n_total, fps))
        sys.stdout.flush()


def main():
    config, args = parse_args()
    model = get_model(config["model"])
    if args.gpu != -1:
        model.to_gpu(args.gpu)

    dataset = VOCBboxDataset(
        data_dir="../dataset/VOC_test/VOC2007_test",
        year='2007', split='test', use_difficult=True, return_difficult=True)
    iterator = iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.evaluation, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterator explicitly
    del imgs

    pred_bboxes, pred_labels, pred_scores = pred_values
    gt_bboxes, gt_labels, gt_difficults = gt_values

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    print()
    print('mAP: {:f}'.format(result['map']))
    for l, name in enumerate(voc_bbox_label_names):
        if result['ap'][l]:
            print('{:s}: {:f}'.format(name, result['ap'][l]))
        else:
            print('{:s}: -'.format(name))


if __name__ == '__main__':
    main()
