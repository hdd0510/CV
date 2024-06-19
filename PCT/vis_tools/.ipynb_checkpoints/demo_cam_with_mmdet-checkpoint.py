import time
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo
from models import build_posenet

from mmdet.apis import inference_detector, init_detector
has_mmdet = True


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))


color2 = [
    # Right Leg
    (0, 0, 255), (0, 0, 255), (0, 0, 255),
    # Left Leg
    (0, 255, 0), (0, 255, 0), (0, 255, 0),
    # Spine
    (255, 255, 0), (255, 255, 0), (255, 255, 0),
    # Right Arm
    (255, 0, 0), (255, 0, 0), (255, 0, 0),
    # Left Arm
    (255, 165, 0), (255, 165, 0), (255, 165, 0),
]

link_pairs2 = [
    # Right leg
    [0, 1], [1, 2], [2, 6],  # r ankle -> r knee -> r hip -> pelvis
    # Left leg
    [5, 4], [4, 3], [3, 6],  # l ankle -> l knee -> l hip -> pelvis
    # Spine
    [6, 7], [7, 8], [8, 9],  # pelvis -> thorax -> upper neck -> head top
    # Right arm
    [10, 11], [11, 12], [12, 7],  # r wrist -> r elbow -> r shoulder -> thorax
    # Left arm
    [15, 14], [14, 13], [13, 7],  # l wrist -> l elbow -> l shoulder -> thorax
]

point_color2 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (252, 176, 243), (0, 176, 240), (252, 176, 243),
                (0, 176, 240), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def vis_pose_result(image, pose_results, thickness, scale=2):
    data_numpy = image

    h = data_numpy.shape[0]
    w = data_numpy.shape[1]

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(data_numpy[:, :, ::-1])
    bk.set_zorder(-1)

    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(16, -1)
        joints_dict = map_joint_dict(dt_joints)

        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11, 16):
                lw = thickness
            else:
                lw = thickness * 2

            line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                          joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                          joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2], )
            line.set_zorder(0)
            ax.add_line(line)

        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness

            circle = mpatches.Circle(tuple(dt_joints[k, :2]),
                                     radius=radius,
                                     ec='black',
                                     fc=chunhua_style.ring_color[k],
                                     alpha=1,
                                     linewidth=1)
            circle.set_zorder(1)
            ax.add_patch(circle)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    # Phóng to ảnh
    image_from_plot = cv2.resize(image_from_plot, (0, 0), fx=scale, fy=scale)

    return image_from_plot


def process_frame(det_model, pose_model, image, det_cat_id, bbox_thr, dataset, dataset_info):
    mmdet_results = inference_detector(det_model, image)
    person_results = process_mmdet_results(mmdet_results, det_cat_id)
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image,
        person_results,
        bbox_thr=bbox_thr,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None)
    return pose_results


def init_pose_model(config, checkpoint=None, device='cuda'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cuda')
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def main():
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    video_capture = cv2.VideoCapture(0)  # Mở camera laptop
    if not video_capture.isOpened():
        raise ValueError("Camera laptop không thể mở được. Hãy kiểm tra lại.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Chất lượng JPEG thấp hơn để giảm dữ liệu
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        
        pose_results = process_frame(det_model, pose_model, frame, args.det_cat_id, args.bbox_thr, dataset, dataset_info)
        vis_frame = vis_pose_result(frame, pose_results, args.thickness)

        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Pose Estimation', vis_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Thời gian chạy: {elapsed_time} giây")


if __name__ == '__main__':
    main()
