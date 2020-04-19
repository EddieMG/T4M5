import os
from glob import glob
import json
import numpy as np
import cv2
import pycocotools.mask as mask_utils
from matplotlib import pyplot as plt
from pycocotools import coco
from itertools import groupby
import random
import copy
import torch
import detectron2.utils.comm as comm
import logging
from time import sleep
from detectron2.engine import HookBase
from detectron2.engine import SimpleTrainer, hooks
from detectron2.engine import DefaultTrainer
from math import ceil
from PIL import Image
from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer


#KITTIMOTS_TRAIN = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
#KITTIMOTS_VAL = [2, 6, 7, 8, 10, 13, 14, 16, 18]


#This is removing sequences whole
KITTIMOTS_TRAIN = [1,2,6,18,20] #1

KITTIMOTS_VAL = [0,3,10,12,14]
KITTIMOTS_TEST = [4,5,7,8,9,11,15]

TRAIN = [1,2,6,18,20] #for virtual
#VAL = [0,3,10,12,14]
#TEST = [4,5,7,8,9,11,15]

# KITTIMOTS_TRAIN = 12
KITTIMOTS_DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN_IMG = KITTIMOTS_DATA_DIR + 'training/image_02'
KITTIMOTS_TRAIN_LABEL = KITTIMOTS_DATA_DIR + 'instances_txt'
KITTIMOTS_TRAIN_MASK = KITTIMOTS_DATA_DIR + 'instances'
KITTI_CATEGORIES = {
    'Pedestrian': 1,
    'Whatever': 0,  # We need 3 classes to not get NANs when evaluating, for some reason, duh
    'Car': 2
}
COCO_CATEGORIES_KITTI = {
    1: 2,
    2: 0
}

MOTSCHALLENGE_DATA_DIR = '/home/mcv/datasets/MOTSChallenge/train/'
MOTSCHALLENGE_TRAIN_IMG = MOTSCHALLENGE_DATA_DIR + 'images'
MOTSCHALLENGE_TRAIN_LABEL = MOTSCHALLENGE_DATA_DIR + 'instances_txt'
MOTSCHALLENGE_TRAIN_MASK = MOTSCHALLENGE_DATA_DIR + 'instances'

MOTS_CATEGORIES = {
    'Pedestrian': 2,
    'Whatever': 0,  # We need 3 classes to not get NANs when evaluating, for some reason, duh
    'Car': 1
}
COCO_CATEGORIES_MOTS = {
    2: 0
}

VIRTUAL_KITTI_DATA_DIR = '/home/mcv/datasets/vKITTI/'
INTERMIDIATE_MASK = '/clone/frames/instanceSegmentation/Camera_0'
INTERMIDIATE_IMG = '/clone/frames/rgb/Camera_0'

class MOTS_KITTI_Dataloader():

    def __init__(self):
        self.train_img_dir = MOTSCHALLENGE_TRAIN_IMG
        self.train_label_dir = MOTSCHALLENGE_TRAIN_LABEL
        self.train_mask_dir = MOTSCHALLENGE_TRAIN_MASK

        label_paths_kitti = sorted(glob(os.path.join(KITTIMOTS_TRAIN_LABEL, '*.txt')))
        label_indices_kitti = ['{0:04d}'.format(l) for l in range(len(label_paths_kitti))]
        self.train_sequences_kitti = [f"{i:04d}" for i in KITTIMOTS_TRAIN]
        # self.train_sequences = label_indices[:KITTIMOTS_TRAIN]
        # self.val_sequences = label_indices[KITTIMOTS_TRAIN:]
        self.val_sequences_kitti = [f"{i:04d}" for i in KITTIMOTS_VAL]

        label_paths = sorted(glob(os.path.join(self.train_label_dir, '*.txt')))
        label_indices = [item.split('/')[-1][:-4] for item in label_paths]
        self.train_sequences = label_indices
        self.val_sequences = label_indices

        print(f'Train Sequences: {self.train_sequences}')
        print(f'Validation Sequences: {self.val_sequences}')
        print(f'Train Sequences: {self.train_sequences_kitti}')
        print(f'Validation Sequences: {self.val_sequences_kitti}')

    def get_dicts(self, train_flag=False):
        sequences = self.train_sequences if train_flag is True else self.val_sequences
        sequences_kitti = self.train_sequences_kitti if train_flag is True else self.val_sequences_kitti
        dataset_dicts = []
        for seq in sequences:
            seq_dicts = self.get_seq_dicts(seq)
            dataset_dicts += seq_dicts

        for seq in sequences_kitti:
            seq_dicts = self.get_seq_dicts_kitti(seq)
            dataset_dicts += seq_dicts

        return dataset_dicts

    def get_seq_dicts(self, seq):
        image_paths = sorted(glob(os.path.join(self.train_img_dir, seq, '*.png')))
        if not image_paths:
            self.extension_flag = False
            image_paths = sorted(glob(os.path.join(self.train_img_dir, seq, '*.jpg')))
        else:
            self.extension_flag = True
        mask_paths = sorted(glob(os.path.join(self.train_mask_dir, seq, '*.png')))

        label_path = os.path.join(self.train_label_dir, seq + '.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
            lines = [l.split(' ') for l in lines]

        seq_dicts = []
        for k in range(len(image_paths)):
            frame_lines = [l for l in lines if int(l[0]) == k]
            if frame_lines:
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                frame_annotations = self.get_frame_annotations(frame_lines, h, w)
                img_dict = self.get_img_dict(seq, k, h, w, frame_annotations)
                seq_dicts.append(img_dict)

        return seq_dicts

    def get_seq_dicts_kitti(self, seq):
        image_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_IMG, seq, '*.png')))
        mask_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_MASK, seq, '*.png')))

        label_path = os.path.join(KITTIMOTS_TRAIN_LABEL, seq + '.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
            lines = [l.split(' ') for l in lines]

        seq_dicts = []
        for k in range(len(image_paths)):
            frame_lines = [l for l in lines if int(l[0]) == k]
            if frame_lines:
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                frame_annotations = self.get_frame_annotations_kitti(frame_lines, h, w)
                img_dict = self.get_img_dict_kitti(seq, k, h, w, frame_annotations)
                seq_dicts.append(img_dict)

        return seq_dicts

    def get_frame_annotations(self, frame_lines, h, w):
        frame_annotations = []
        for detection in frame_lines:
            category_id = int(detection[2])
            if category_id not in MOTS_CATEGORIES.values():
                continue

            rle = {
                'counts': detection[-1].strip(),
                'size': [h, w]
            }
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]

            mask = coco.maskUtils.decode(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue

            annotation = {
                'category_id': COCO_CATEGORIES_MOTS[category_id],
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg,
            }
            frame_annotations.append(annotation)

        return frame_annotations

    def get_frame_annotations_kitti(self, frame_lines, h, w):
        frame_annotations = []
        for detection in frame_lines:
            category_id = int(detection[2])
            if category_id not in KITTI_CATEGORIES.values():
                continue

            rle = {
                'counts': detection[-1].strip(),
                'size': [h, w]
            }
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]

            mask = coco.maskUtils.decode(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue

            annotation = {
                'category_id': COCO_CATEGORIES_KITTI[category_id],
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg
            }
            frame_annotations.append(annotation)

        return frame_annotations

    def get_img_dict(self, seq, k, h, w, frame_annotations):
        if self.extension_flag:
            filename = '{0:06d}.png'.format(k)
        else:
            filename = '{0:06d}.jpg'.format(k)
        img_path = os.path.join(self.train_img_dir, seq, filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }

        return img_dict

    def get_img_dict_kitti(self, seq, k, h, w, frame_annotations):
        filename = '{0:06d}.png'.format(k)
        img_path = os.path.join(KITTIMOTS_TRAIN_IMG, seq, filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }

        return img_dict

class Virtual_Real_KITTI():

    def __init__(self):
        ##Virtual
        self.sequences = ['Scene{0:02d}'.format(l) for l in TRAIN]
        print(f'Train Sequences: {self.sequences}')

        ###REAL
        #label_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_LABEL, '*.txt')))
        #label_indices = ['{0:04d}'.format(l) for l in range(len(label_paths))]
        self.train_sequences = [f"{i:04d}" for i in KITTIMOTS_TRAIN]

        # validations 0002 0006  0007   0008   0010   0013   0014   0016   0018

        print(f'Train Sequences: {self.train_sequences}')



    def get_dicts(self, percentage):

        sequences = self.train_sequences
        dataset_dicts = []

        #real
        #for seq in sequences[0:ceil(percentage)]: #no random
        for seq in sequences: #random
            print(seq)
            seq_dicts = self.get_seq_dicts(seq,percentage)
            dataset_dicts += seq_dicts

        dataset_dicts = random.sample(dataset_dicts, ceil(len(dataset_dicts)*(percentage*2/10)))

        #virtual
        for seq in self.sequences:
            seq_dicts = self.get_seq_dicts_virtual(seq)
            dataset_dicts += seq_dicts

        return dataset_dicts

    def get_seq_dicts_virtual(self, seq):
        image_paths = sorted(glob(VIRTUAL_KITTI_DATA_DIR + seq + INTERMIDIATE_IMG + os.sep + '*.jpg'))
        mask_paths = sorted(glob(VIRTUAL_KITTI_DATA_DIR + seq + INTERMIDIATE_MASK + os.sep + '*.png'))
        seq_dicts = []
        for k, (m_path, i_path) in enumerate(zip(mask_paths, image_paths)):
            img = np.array(Image.open(m_path)).astype(np.uint8)
            frame_annotations = self.get_frame_annotations_virtual(img)
            img_dict = self.get_img_dict_virtual(seq, k, i_path, img, frame_annotations)
            seq_dicts.append(img_dict)
        return seq_dicts

    def get_frame_annotations_virtual(self, img):
        h, w = img.shape
        frame_annotations = []
        instances = np.unique(img)
        for ins in instances[1:]:
            mask = np.copy(img)
            mask[(mask == ins)] = 1
            mask[(mask != 1)] = 0
            rle = mask_utils.frPyObjects(mask_to_rle(mask), w, h)
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue
            annotation = {
                'category_id': 2,
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg,
            }
            frame_annotations.append(annotation)
        return frame_annotations

    def get_img_dict_virtual(self, seq, k, filename, img, frame_annotations):
        h, w = img.shape
        img_dict = {
            'file_name': filename,
            'image_id': k + (int(seq[-2:]) * 1e6),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }
        return img_dict



    def get_seq_dicts(self, seq,percentage):
        image_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_IMG, seq, '*.png')))
        mask_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_MASK, seq, '*.png')))

        label_path = os.path.join(KITTIMOTS_TRAIN_LABEL, seq + '.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
            lines = [l.split(' ') for l in lines]

        seq_dicts = []

        if percentage == 0.5:
            for k in range(ceil(len(image_paths)*percentage)): #5
                frame_lines = [l for l in lines if int(l[0]) == k]
                if frame_lines:
                    h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                    frame_annotations = self.get_frame_annotations(frame_lines, h, w)
                    img_dict = self.get_img_dict(seq, k, h, w, frame_annotations)
                    seq_dicts.append(img_dict)

            return seq_dicts

        else:
            for k in range(len(image_paths)):
                frame_lines = [l for l in lines if int(l[0]) == k]
                if frame_lines:
                    h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                    frame_annotations = self.get_frame_annotations(frame_lines, h, w)
                    img_dict = self.get_img_dict(seq, k, h, w, frame_annotations)
                    seq_dicts.append(img_dict)

            return seq_dicts

    def get_frame_annotations(self, frame_lines, h, w):
        frame_annotations = []
        for detection in frame_lines:
            category_id = int(detection[2])
            if category_id not in KITTI_CATEGORIES.values():
                continue

            rle = {
                'counts': detection[-1].strip(),
                'size': [h, w]
            }
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]

            mask = coco.maskUtils.decode(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue

            annotation = {
                'category_id': COCO_CATEGORIES_KITTI[category_id],
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg
            }
            frame_annotations.append(annotation)

        return frame_annotations

    def get_img_dict(self, seq, k, h, w, frame_annotations):
        filename = '{0:06d}.png'.format(k)
        img_path = os.path.join(KITTIMOTS_TRAIN_IMG, seq, filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }

        return img_dict

class MOTS_Dataloader():

    def __init__(self):
        self.train_img_dir = MOTSCHALLENGE_TRAIN_IMG
        self.train_label_dir = MOTSCHALLENGE_TRAIN_LABEL
        self.train_mask_dir = MOTSCHALLENGE_TRAIN_MASK

        label_paths = sorted(glob(os.path.join(self.train_label_dir, '*.txt')))
        label_indices = [item.split('/')[-1][:-4] for item in label_paths]
        self.train_sequences = label_indices
        self.val_sequences = label_indices

        print(f'Train Sequences: {self.train_sequences}')
        print(f'Validation Sequences: {self.val_sequences}')

    def get_dicts(self, train_flag=False):
        sequences = self.train_sequences if train_flag is True else self.val_sequences

        dataset_dicts = []
        for seq in sequences:
            seq_dicts = self.get_seq_dicts(seq)
            dataset_dicts += seq_dicts

        return dataset_dicts

    def get_seq_dicts(self, seq):
        image_paths = sorted(glob(os.path.join(self.train_img_dir, seq, '*.png')))
        if not image_paths:
            self.extension_flag = False
            image_paths = sorted(glob(os.path.join(self.train_img_dir, seq, '*.jpg')))
        else:
            self.extension_flag = True
        mask_paths = sorted(glob(os.path.join(self.train_mask_dir, seq, '*.png')))

        label_path = os.path.join(self.train_label_dir, seq + '.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
            lines = [l.split(' ') for l in lines]

        seq_dicts = []
        for k in range(len(image_paths)):
            frame_lines = [l for l in lines if int(l[0]) == k]
            if frame_lines:
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                frame_annotations = self.get_frame_annotations(frame_lines, h, w)
                img_dict = self.get_img_dict(seq, k, h, w, frame_annotations)
                seq_dicts.append(img_dict)

        return seq_dicts

    def get_frame_annotations(self, frame_lines, h, w):
        frame_annotations = []
        for detection in frame_lines:
            category_id = int(detection[2])
            if category_id not in MOTS_CATEGORIES.values():
                continue

            rle = {
                'counts': detection[-1].strip(),
                'size': [h, w]
            }
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]

            mask = coco.maskUtils.decode(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue

            annotation = {
                'category_id': COCO_CATEGORIES_MOTS[category_id],
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg,
            }
            frame_annotations.append(annotation)

        return frame_annotations

    def get_img_dict(self, seq, k, h, w, frame_annotations):
        if self.extension_flag:
            filename = '{0:06d}.png'.format(k)
        else:
            filename = '{0:06d}.jpg'.format(k)
        img_path = os.path.join(self.train_img_dir, seq, filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }

        return img_dict

class KITTIMOTS_Dataloader():

    def __init__(self):
        label_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_LABEL, '*.txt')))
        label_indices = ['{0:04d}'.format(l) for l in range(len(label_paths))]
        self.train_sequences = [f"{i:04d}" for i in KITTIMOTS_TRAIN]
        self.val_sequences = [f"{i:04d}" for i in KITTIMOTS_VAL]
        self.test_sequences = [f"{i:04d}" for i in KITTIMOTS_TEST]
        # validations 0002 0006  0007   0008   0010   0013   0014   0016   0018

        print(f'Train Sequences: {self.train_sequences}')
        print(f'Validation Sequences: {self.val_sequences}')

    def get_dicts(self, ds_type="train"):
        if ds_type == "train":
            sequences = self.train_sequences
        elif ds_type == "validation":
            sequences = self.val_sequences
        elif ds_type == "test":
            sequences = self.test_sequences

        dataset_dicts = []
        for seq in sequences:
            seq_dicts = self.get_seq_dicts(seq)
            dataset_dicts += seq_dicts

        return dataset_dicts

    def get_seq_dicts(self, seq):
        image_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_IMG, seq, '*.png')))
        mask_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_MASK, seq, '*.png')))

        label_path = os.path.join(KITTIMOTS_TRAIN_LABEL, seq + '.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
            lines = [l.split(' ') for l in lines]

        seq_dicts = []
        for k in range(len(image_paths)):
            frame_lines = [l for l in lines if int(l[0]) == k]
            if frame_lines:
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                frame_annotations = self.get_frame_annotations(frame_lines, h, w)
                img_dict = self.get_img_dict(seq, k, h, w, frame_annotations)
                seq_dicts.append(img_dict)

        return seq_dicts

    def get_frame_annotations(self, frame_lines, h, w):
        frame_annotations = []
        for detection in frame_lines:
            category_id = int(detection[2])
            if category_id not in KITTI_CATEGORIES.values():
                continue

            rle = {
                'counts': detection[-1].strip(),
                'size': [h, w]
            }
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]

            mask = coco.maskUtils.decode(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue

            annotation = {
                'category_id': COCO_CATEGORIES_KITTI[category_id],
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg
            }
            frame_annotations.append(annotation)

        return frame_annotations

    def get_img_dict(self, seq, k, h, w, frame_annotations):
        filename = '{0:06d}.png'.format(k)
        img_path = os.path.join(KITTIMOTS_TRAIN_IMG, seq, filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }

        return img_dict

class VirtualKITTI_Dataloader():
    def __init__(self):
        self.sequences = ['Scene{0:02d}'.format(l) for l in TRAIN]
        print(f'Train Sequences: {self.sequences}')

    def get_dicts(self):
        dataset_dicts = []
        for seq in self.sequences:
            seq_dicts = self.get_seq_dicts(seq)
            dataset_dicts += seq_dicts
        return dataset_dicts

    def get_seq_dicts(self, seq):
        image_paths = sorted(glob(VIRTUAL_KITTI_DATA_DIR + seq + INTERMIDIATE_IMG + os.sep + '*.jpg'))
        mask_paths = sorted(glob(VIRTUAL_KITTI_DATA_DIR + seq + INTERMIDIATE_MASK + os.sep + '*.png'))
        seq_dicts = []
        for k, (m_path, i_path) in enumerate(zip(mask_paths, image_paths)):
            img = np.array(Image.open(m_path)).astype(np.uint8)
            frame_annotations = self.get_frame_annotations(img)
            img_dict = self.get_img_dict(seq, k, i_path, img, frame_annotations)
            seq_dicts.append(img_dict)
        return seq_dicts

    def get_frame_annotations(self, img):
        h, w = img.shape
        frame_annotations = []
        instances = np.unique(img)
        for ins in instances[1:]:
            mask = np.copy(img)
            mask[(mask == ins)] = 1
            mask[(mask != 1)] = 0
            rle = mask_utils.frPyObjects(draw_loss(mask), w, h)
            bbox = coco.maskUtils.toBbox(rle).tolist()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(item) for item in bbox]
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            seg = [[int(i) for i in c.flatten()] for c in contours]
            seg = [s for s in seg if len(s) >= 6]
            if not seg:
                continue
            annotation = {
                'category_id': 2,
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': bbox,
                'segmentation': seg,
            }
            frame_annotations.append(annotation)
        return frame_annotations

    def get_img_dict(self, seq, k, filename, img, frame_annotations):
        h, w = img.shape
        img_dict = {
            'file_name': filename,
            'image_id': k + (int(seq[-2:]) * 1e6),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }
        return img_dict

class OurMapper(DatasetMapper):
    """
    A customized version of `detectron2.data.DatasetMapper`
    """
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = deepcopy(cfg)
        self.tfm_gens = []

    def __call__(self, dataset_dict):

        self.tfm_gens = []

        dataset_dict = deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            # Crop
            '''print("Augmentation: ", "T.RandomCrop('relative', [0.8, 0.4])")
            crop_gen = T.RandomCrop('relative', [0.8, 0.4])
            self.tfm_gens.append(crop_gen)'''
            # Horizontal flip
            print("Augmentation: ", "T.RandomFlip(prob=0.5, horizontal=True, vertical=False)")
            flip_gen = T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
            self.tfm_gens.append(flip_gen)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

class OurTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

def mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle