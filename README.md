# Yolo_v2_chainer
Yolo v2 implementation by Chainer

YOLO9000: Better, Faster, Stronger [link](https://pjreddie.com/media/files/papers/YOLO9000.pdf)  
See the [project webpage](https://pjreddie.com/darknet/yolo/) for more details.

original code: https://github.com/pjreddie/darknet

## Converter from darknet to chainer
```bash
python to_chainer_converter.py --orig /path/to/original_model --name yolo_v2_chainer
```

## Inference
```bash
python inference.py --img /path/to/img.png --width 1024 --height 512
```


# TODO
- Data loader for imagenet and coco dataset.
- Training codes for darknet 224×224, 448×448.
- Training codes for Yolo v2.
- Evaluation scripts for Imagenet[2012] and COCO[2014] dataset.
- Inference scripts for Yolo v2.
- Model converter from darknet to Chainer.
