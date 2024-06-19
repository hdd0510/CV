from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

import sys
sys.path.append("../lib")
import time

import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

# CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CTX = torch.device('cpu')
keypoint_info = {
    0: 'r ankle',
    1: 'r knee',
    2: 'r hip',
    3: 'l hip',
    4: 'l knee',
    5: 'l ankle',
    6: 'pelvis',
    7: 'thorax',
    8: 'upper neck',
    9: 'head top',
    10: 'r wrist',
    11: 'r elbow',
    12: 'r shoulder',
    13: 'l shoulder',
    14: 'l elbow',
    15: 'l wrist'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

NUM_KPTS = 16   
SKELETON_MPII = [
    [0, 1], [1, 2], [2, 6], [3, 4], [4, 5], [3, 6],  # Legs
    [6, 7], [7, 8], [8, 9],                         # Spine
    [7, 12], [12, 11], [11, 10],                    # Right arm
    [7, 13], [13, 14], [14, 15]                     # Left arm
]


MPIIColors = [
    [0, 128, 128], [0, 255, 255], [128, 128, 0], [255, 255, 0], [128, 0, 128], [255, 0, 255], 
    [128, 0, 0], [255, 0, 0], [0, 128, 0], [0, 255, 0], [0, 0, 128], [0, 0, 255], 
    [64, 128, 128], [64, 255, 255], [128, 64, 128], [255, 64, 255], [64, 128, 0], [64, 255, 0]
]

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    keypoints = keypoints[0]
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON_MPII)):
        kpt_a, kpt_b = SKELETON_MPII[i][0], SKELETON_MPII[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, MPIIColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, MPIIColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), MPIIColors[i], 2)

def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Define PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)  # .unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--imgDir', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/u01/dunghd/project/output')
    parser.add_argument('--outputImagesDir', type=str, default='/u01/dunghd/project/output', help='Output directory for inferred images')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)
    
    # Create directory for output images if it doesn't exist
    if not os.path.exists(args.outputImagesDir):
        os.makedirs(args.outputImagesDir, exist_ok=True)
        
    csv_output_rows = []

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    box_model.to(CTX)
    box_model.eval()
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # List all image files in the directory
    # image_files = [os.path.join(args.imgDir, f) for f in os.listdir(args.imgDir) if os.path.isfile(os.path.join(args.imgDir, f))]
    img_files = [args.imgDir]
    for count, image_file in enumerate(img_files):
        total_now = time.time()
        
        image_bgr = cv2.imread(image_file)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 images for person detection and pose estimation
        if cfg.DATASET.COLOR_RGB:
            image_per = image_rgb.copy()
            image_pose = image_rgb.copy()
        else:
            image_per = image_bgr.copy()
            image_pose = image_bgr.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        # object detection box
        now = time.time()
        pred_boxes = get_person_detection_boxes(box_model, image_per, threshold=0.82)
        then = time.time()
        print(f"Find person bbox in {image_file}: {then - now} sec")

        # Can not find people. Move to next image
        if not pred_boxes:
            continue

        if args.writeBoxFrames:
            for box in pred_boxes:
                box = [list(map(int, x)) for x in box]
                cv2.rectangle(image_debug, box[0], box[1], color=(0, 255, 0), thickness=3)  # Draw Rectangle with the coordinates

        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        now = time.time()
        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)

        draw_pose(pose_preds,image_debug)
        then = time.time()
        print(f"Find person pose in {image_file}: {then - now} sec")

        new_csv_row = []
        for coords in pose_preds:
            # Draw each point on image
            for coord in coords:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                new_csv_row.extend([x_coord, y_coord])

        total_then = time.time()

        text = "{:03.2f} sec".format(total_then - total_now)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # cv2.imshow("pos", image_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        csv_output_rows.append(new_csv_row)
        
        # Save inferred image to outputImagesDir
        img_file = os.path.join(args.outputImagesDir, f'pose_{count:08d}.jpg')
        cv2.imwrite(img_file, image_debug)

    # write csv
    csv_headers = ['frame']
    for keypoint in keypoint_info.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerow(csv_output_rows)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
