#!/bin/bash

# Thiết lập biến môi trường PYTHONPATH
export PYTHONPATH=$(python -c 'import os, sys; print(os.path.abspath("..") + ":" + os.environ.get("PYTHONPATH", ""))')

# Chạy lệnh python với các tham số
python vis_tools/demo_cam_with_mmdet.py \
    vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth \
    configs/pct_base_classifier.py \
    work_dirs/pct_base_classifier/epoch_130.pth
