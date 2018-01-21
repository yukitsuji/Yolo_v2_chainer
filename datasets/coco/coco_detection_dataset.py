#/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import random
from scipy.misc import imread

from chainer import dataset

from collections import defaultdict
import os
from datasets.coco.utils import *


class CocoDetectionDataset(dataset.DatasetMixin):

    """Dataset class for a task on `Coco Detection Dataset`_.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least three directories, :obj:`training`, `testing`
            and `ImageSets`.
    """
    def __init__(self, root_dir='./', data_dir='train2014',
                 anno_file='annotations'):
        coco = COCO(os.path.join(root_dir, anno_file))
        anns = coco.loadAnns(coco.getAnnIds())

        self.coco = coco
        self.anns = anns
        self.coco_root = root_dir
        self.coco_data = data_dir

    def __len__(self):
        return len(self.anns)

    def get_example(self, i):
        """Called by the iterator to fetch a data sample.

        A data sample from MSCOCO consists of an image and its corresponding
        caption.

        The returned image has the shape (channel, height, width).
        """
        ann = self.anns[i]

        # Load the image
        img_id = ann['image_id']
        img_file_name = self.coco.loadImgs([img_id])[0]['file_name']
        img = Image.open(
            os.path.join(self.coco_root, self.coco_data, img_file_name))
        if img.mode == 'RGB':
            img = np.asarray(img, np.float32).transpose(2, 0, 1)
        elif img.mode == 'L':
            img = np.asarray(img, np.float32)
            img = np.broadcast_to(img, (3,) + img.shape)
        else:
            raise ValueError('Invalid image mode {}'.format(img.mode))

        # Load the caption, i.e. sequence of tokens
        tokens = [self.vocab.get(w, _unk) for w in
                  ['<bos>'] + split(ann['caption']) + ['<eos>']]
        tokens = np.array(tokens, np.int32)

        return img, tokens
