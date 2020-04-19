import os
from glob import glob
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pycocotools import coco
import copy
import torch

from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode


KITTIMOTS_VAL = [2, 6, 7, 8, 10, 13, 14, 16, 18]
#KITTIMOTS_TRAIN = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
#KITTIMOTS_VAL = [2, 6, 7, 8, 10, 13, 14, 16, 18]
KITTIMOTS_TRAIN = [2]
#KITTIMOTS_TRAIN = 12
KITTIMOTS_DATA_DIR = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN_IMG = KITTIMOTS_DATA_DIR+'training/image_02'
KITTIMOTS_TRAIN_LABEL = KITTIMOTS_DATA_DIR+'instances_txt'
KITTIMOTS_TRAIN_MASK = KITTIMOTS_DATA_DIR+'instances'
KITTI_CATEGORIES = {
    'Pedestrian': 1,
    'Whatever': 0, # We need 3 classes to not get NANs when evaluating, for some reason, duh
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
class MOTS_KITTI_Dataloader():

    def __init__(self):
        self.train_img_dir = MOTSCHALLENGE_TRAIN_IMG
        self.train_label_dir = MOTSCHALLENGE_TRAIN_LABEL
        self.train_mask_dir = MOTSCHALLENGE_TRAIN_MASK

        label_paths_kitti = sorted(glob(os.path.join(KITTIMOTS_TRAIN_LABEL, '*.txt')))
        label_indices_kitti = ['{0:04d}'.format(l) for l in range(len(label_paths_kitti))]
        self.train_sequences_kitti = [f"{i:04d}" for i in KITTIMOTS_TRAIN]
        #self.train_sequences = label_indices[:KITTIMOTS_TRAIN]
        #self.val_sequences = label_indices[KITTIMOTS_TRAIN:]
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

        label_path = os.path.join(KITTIMOTS_TRAIN_LABEL, seq+'.txt')
        with open(label_path,'r') as file:
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
        img_path = os.path.join(KITTIMOTS_TRAIN_IMG,seq,filename)
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

        if not os.path.isdir(self.train_img_dir):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(self.train_label_dir):
            raise Exception('The labels directory is not correct.')
        if not os.path.isdir(self.train_mask_dir):
            raise Exception('The masks directory is not correct')

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
        if not os.path.isdir(KITTIMOTS_TRAIN_IMG):
            raise Exception('The image directory is not correct.')
        if not os.path.isdir(KITTIMOTS_TRAIN_LABEL):
            raise Exception('The labels directory is not correct.')
        if not os.path.isdir(KITTIMOTS_TRAIN_MASK):
            raise Exception('The masks directory is not correct')

        label_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_LABEL, '*.txt')))
        label_indices = ['{0:04d}'.format(l) for l in range(len(label_paths))]
        self.train_sequences = [f"{i:04d}" for i in KITTIMOTS_TRAIN]
        #self.train_sequences = label_indices[:KITTIMOTS_TRAIN]
        #self.val_sequences = label_indices[KITTIMOTS_TRAIN:]
        self.val_sequences = [f"{i:04d}" for i in KITTIMOTS_VAL]

        # validations 0002 0006  0007   0008   0010   0013   0014   0016   0018

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
        image_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_IMG, seq, '*.png')))
        mask_paths = sorted(glob(os.path.join(KITTIMOTS_TRAIN_MASK, seq, '*.png')))

        label_path = os.path.join(KITTIMOTS_TRAIN_LABEL, seq+'.txt')
        with open(label_path,'r') as file:
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
        img_path = os.path.join(KITTIMOTS_TRAIN_IMG,seq,filename)
        img_dict = {
            'file_name': img_path,
            'image_id': k + (int(seq) * 1e3),
            'height': h,
            'width': w,
            'annotations': frame_annotations
        }

        return img_dict
