#import albumentations as A
import cv2

import torch

#from albumentations.pytorch import ToTensorV2
#from utils import seed_everything

hyp = {
    "lr0": 0.00334,
    "lrf": 0.15135,
    "momentum": 0.74832,
    "weight_decay": 0.00025,
    "warmup_epochs": 3.3835,
    "warmup_momentum": 0.59462,
    "warmup_bias_lr": 0.18657,
    "box": 0.02,
    "cls": 0.21638,
    "cls_pw": 0.5,
    "obj": 0.51728,
    "obj_pw": 0.67198,
    "iou_t": 0.2,
    "anchor_t": 3.3744,
    "fl_gamma": 0.0,
    "hsv_h": 0.01041,
    "hsv_s": 0.54703,
    "hsv_v": 0.27739,
    "degrees": 0.0,
    "translate": 0.04591,
    "scale": 0.75544,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.85834,
    "mixup": 0.04266,
    "copy_paste": 0.0,
    "anchors": 3.412,
}

DATASET = 'PASCAL_VOC'
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 16
IMAGE_SIZE = 416
NUM_CLASSES = 20
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 200
CONF_THRESHOLD = 0.25
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
IMG_DIR = "../VOCdevkit/VOC2012/JPEGImages"
LABEL_DIR = "../VOCdevkit/VOC2012/Annotations"

# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ]  # Note these have been rescaled to be between [0, 1]

ANCHORS = [
    [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]],
    [[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]],
    [[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]
]
# ANCHORS = [
#     [(116, 90), (156, 198), (373, 326)],
#     [(30, 61), (62, 45), (59, 119)],
#     [(10, 13), (16, 30), (33, 23)]
# ]

scale = 1.1
# train_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
#         A.PadIfNeeded(
#             min_height=int(IMAGE_SIZE * scale),
#             min_width=int(IMAGE_SIZE * scale),
#             border_mode=cv2.BORDER_CONSTANT,
#         ),
#         A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
#         A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
#         A.OneOf(
#             [
#                 A.ShiftScaleRotate(
#                     rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
#                 ),
#                 A.IAAAffine(shear=15, p=0.5, mode="constant"),
#             ],
#             p=1.0,
#         ),
#         A.HorizontalFlip(p=0.5),
#         A.Blur(p=0.1),
#         A.CLAHE(p=0.1),
#         A.Posterize(p=0.1),
#         A.ToGray(p=0.1),
#         A.ChannelShuffle(p=0.05),
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
#         ToTensorV2(),
#     ],
#     bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
# )
# test_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=IMAGE_SIZE),
#         A.PadIfNeeded(
#             min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
#         ),
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
#         ToTensorV2(),
#     ],
#     bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
# )

PASCAL_CLASSES = [
    "aeroplane",#0
    "bicycle",  #1
    "bird",     #2
    "boat",     #3
    "bottle",   #4
    "bus",      #5
    "car",      #6
    "cat",          #7
    "chair",        #8
    "cow",          #9
    "diningtable",  #10
    "dog",          #11
    "horse",        #12
    "motorbike",    #13
    "person",       #14
    "pottedplant",  #15
    "sheep",        #16
    "sofa",         #17
    "train",        #18
    "tvmonitor"     #19
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]
