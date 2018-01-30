# Yolo_v2_chainer
Yolo v2 implementation by Chainer

YOLO9000: Better, Faster, Stronger [link](https://pjreddie.com/media/files/papers/YOLO9000.pdf)  
See the [project webpage](https://pjreddie.com/darknet/yolo/) for more details.

original code: https://github.com/pjreddie/darknet

<img src="./imgs/chainer/dog.png"/>  

## Download datasets
```bash
・COCO
http://cocodataset.org/#download

・VOC
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
```

## Converter from darknet to chainer
```bash
python to_chainer_converter.py --orig /path/to/original_model --name yolo_v2_chainer
```

## demo
```bash
python demo.py experiments/yolov2_update_608_test.yml --img_path ./data/dog.jpg --thresh 0.20 --nms_thresh 0.3 --save dog
nms: by class
nms_thresh = 0.3
img_thresh = ?
```

## Evaluation
```bash
python evaluate.py experiments/yolov2_update_416_eval.yml --batchsize 32 --gpu 0
# fps of evaluation includes the time of loading images.
nms: by class
nms_thresh = 0.5
img_thresh = 0.001
```

## Model Comparison from Darknet
| Model | Dataset | darknet map | chainer map | cfg | darknet weight | Chainer | orig fps | chainer fps |
|:--------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| **YOLOv2** | **VOC2007+20012** | **76.8** | **75.2** |  **[link](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg)** | **[link](https://pjreddie.com/media/files/yolo-voc.weights)** | | **** | **** |
| **YOLOv2 544×544** | **VOC2007+20012** | **78.6** | **77.9** | **[link](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg)** | **[link](https://pjreddie.com/media/files/yolo-voc.weights)** | | | |
| **Tiny YOLO** | **VOC2007+20012** | | | **[link](https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg)** | **[link](https://pjreddie.com/media/files/tiny-yolo-voc.weights)** | | | |
| **YOLOv2 608×608** | **COCO** | | |  **[link](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg)** | **[link](https://pjreddie.com/media/files/yolo.weights)** | **[link](https://www.dropbox.com/s/j9ehggm8f82h0kb/yolov2_update_coco_608.npz)** | 58.8 | 66.6 |
| **Tiny YOLO** | **COCO** | | | **[link](https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo.cfg)** | **[link](https://pjreddie.com/media/files/tiny-yolo.weights)** | | | |
| **prior version(YOLOv2)** | **COCO** | | | **[link](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.2.0.cfg)** | **[link](https://drive.google.com/open?id=0B4kMaWAXZNSWcUJCVW1aOHV0MkU)** | **[link](https://www.dropbox.com/s/vff05c4gb6dojft/yolov2_prior_coco_608.npz)**|  |  | |

<!-- ## Darknetを読んで
- GroundTruthは、一枚ごとに30個以内（インスタンス）と仮定している。
- loss関数の計算方法
- まず、各pixel, anchor毎に、GroundTruthとIOU Matchingを行う。 もしmax iouが閾値を超えている場合、
  その領域の誤差を0とする。もし閾値を超えていなければ、noobject_scale * (0 - l.output[index])
  また、少ないbatch数(12800まで)のときには、すべての領域に関して、x, y, w, hの誤差を計算する。scaleは0.01
- GroundTruthの(x, y, w, h)の値は、(x / img_w, y / img_h, exp(w * (anchor_w / img_w)))
- Anchorの値の意味： -->


# TODO
- Data loader for imagenet and coco dataset.
- Training codes for darknet 224×224, 448×448.
- Training codes for Yolo v2.
- Evaluation scripts for Imagenet[2012] and COCO[2014] dataset.
- Inference scripts for Yolo v2.
- Model converter from darknet to Chainer.
