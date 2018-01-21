__author__ = 'yukitsuji'

# Original code is here (https://github.com/cocodataset/cocoapi/)

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys
from urllib.request import urlretrieve


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnotationIds(self, img_ids=None, category_ids=None, iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param img_ids  (int array)     : get anns for given imgs
               category_Ids  (int array): get anns for given cats
               iscrowd (boolean)        : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        if isinstance(img_ids, list):
            lists = [self.imgToAnns[img_id] for img_id in img_ids if img_id in self.imgToAnns]
            anns = lists
            # anns = list(itertools.chain.from_iterable(lists))
        else:
            anns = self.dataset['annotations']
        
        if isinstance(category_ids, list):
            anns = [[an for an in ann if ann['category_id'] in category_ids] for ann in anns]

        if iscrowd is not None:
            ids = [[an['id'] for an in anns if an['iscrowd'] == iscrowd] for ann in anns]
        else:
            ids = [[an['id'] for an in ann] for ann in anns]
        return ids

    def getCategoryIds(self, category_names=None, category_ids=None):
        """
        filtering parameters. default skips that filter.
        :param category_names (str array)  : get categories for given names
        :param category_ids (int array)  : get categories for given ids
        :return: ids (int array)   : integer array of cat ids
        """
        cats = self.dataset['categories']
        if category_names:
            cats =  [cat for cat in cats if cat['name'] if category_names]

        if category_ids:
            cats = [cat for cat in cats if cat['id'] in category_ids]

        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, img_ids=None, category_ids=None):
        '''
        Get img ids that satisfy given filter conditions.
        :param img_ids (int array) : get imgs for given ids
        :param category_ids (int array) : get imgs with all given categories
        :return: img_ids (int array)  : integer array of img ids
        '''
        img_ids = set(img_ids) if img_ids else set([])
        if isinstance(category_ids, list):
            for category_id in category_ids:
                if len(img_ids):
                    img_ids &= set(self.catToImgs[category_id])
                else:
                    img_ids = set(self.catToImgs[category_id])
        else:
            img_ids = self.imgs.keys()
        return list(img_ids)

    def loadAnnotations(self, annotation_ids=None):
        """
        Load annotations with the specified ids.
        :param annotation_ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if isinstance(annotation_ids, list):
            return [[self.anns[ann_id] for ann_id in ann_ids] for ann_ids in annotation_ids]
        elif isinstance(annotation_ids, int):
            return [self.anns[annotation_ids]]
        raise("Please specify some annotation ids by 'int' or 'list'")

    def loadCategories(self, ids=None):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if isinstance(ids, list):
            return [self.cats[id] for id in ids]
        elif isinstance(ids, int):
            return [self.cats[ids]]

    def loadImgs(self, img_ids=None):
        """
        Load anns with the specified ids.
        :param img_ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if isinstance(img_ids, list):
            return [self.imgs[img_id] for img_id in img_ids]
        elif isinstance(img_ids, int):
            return [self.imgs[img_ids]]

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(self, tarDir = None, imgIds = [] ):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is None:
            print('Please specify target directory')
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urlretrieve(img['coco_url'], fname)
            print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time()- tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann
