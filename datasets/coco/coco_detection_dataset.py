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
        category_ids = coco.getCategoryIds()
        cat_label_dic = {cat:(label+1) for label, cat in enumerate(category_ids)}
        all_img_ids = coco.getImgIds()
        all_annotation_ids = coco.getAnnotationIds(img_ids=all_img_ids)
        all_annotations = coco.loadAnnotations(all_annotation_ids)
        img_base_dir = os.path.join(root_dir, data_dir)
        all_img_info = coco.loadImgs(all_img_ids)
        self.imgs_path = [os.path.join(img_base_dir, img_info['file_name']) for img_info in all_img_info]
        self.bboxes = [[ann['bbox'] for ann in anns] for anns in all_annotations]
        self.labels = [[cat_label_dic[ann['category_id']] for ann in anns] for anns in all_annotations]
        self.coco_root = root_dir
        self.coco_data = data_dir

        # Convert all datasets to batch
        print(anns[i])

    def __len__(self):
        return len(self.imgs_path)

    def get_example(self, i):
        """Called by the iterator to fetch a data sample.

        A data sample from MSCOCO consists of an image and its corresponding
        caption.

        The returned image has the shape (channel, height, width).
        """
        # Load the image
        img = Image.open(self.imgs_path[i])
        if img.mode == 'RGB':
            img = np.asarray(img, np.float32).transpose(2, 0, 1)
        else:
            raise ValueError('Invalid image mode {}'.format(img.mode))
        bboxes = self.bboxes[i]
        labels = self.labels[i]

        return img, bboxes, labels
