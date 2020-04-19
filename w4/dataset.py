# import some common libraries
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm.auto import tqdm
import pickle

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


class get_kitti_dicts():

    def __init__(self, img_dir, annot_dir):

        self.img_dir = img_dir
        self.annot_dir = annot_dir

        self.dataset_dicts = []
        self.dataset_train = []
        self.dataset_val = []

        self.thing_classes = ['Car', 'Pedestrian']

        for idx, img_name in tqdm(enumerate(os.listdir(img_dir)), desc='Getting kitti dicts'):
            if 'det' not in img_name:
                record = {}
                filename = os.path.join(img_dir, img_name)
                height, width = Image.open(filename).size

                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width

                f = open(os.path.join(annot_dir, img_name.split('.')[0] + ".txt"), "r")
                annos = f.readline().strip()

                img = np.asarray(Image.open(filename))

                objs = []
                while annos != "":
                    if annos.split(' ')[0] not in self.thing_classes:
                        annos = f.readline().strip()
                        continue
                    bbox = [float(coord) for coord in annos.split(' ')[4:8]]
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": self.thing_classes.index(annos.split(' ')[0]),
                        "iscrowd": 0
                    }
                    objs.append(obj)
                    annos = f.readline().strip()

                record["annotations"] = objs
                self.dataset_dicts.append(record)

        self.dataset_train = self.dataset_dicts

    def __len__(self):
        return len(self.dataset_dicts)

    def train_val_split(self, split=0):
        val_samples = random.choices(np.arange(0, len(self), 1), k=int(len(self) * split))
        self.dataset_train = [dic for dic in self.dataset_dicts if not dic['image_id'] in val_samples]
        self.dataset_val = [dic for dic in self.dataset_dicts if dic['image_id'] in val_samples]

    def get_dicts(self, data_split):
        if data_split is 'train':
            return self.dataset_train
        else:
            return self.dataset_val


class get_kitti_mots_dicts():

    def __init__(self, img_dir, annot_dir, thing_classes, map_classes=None, dict_dir=[]):

        self.img_dir = img_dir
        self.annot_dir = annot_dir

        self.dataset_dicts = []
        self.dataset_train = []
        self.dataset_val = []

        self.thing_classes = thing_classes

        if dict_dir:
            with (open(dict_dir, "rb")) as openfile:
                self.dataset_dicts = pickle.load(openfile)

            self.dataset_train = self.dataset_dicts
            return

        for i, folder_name in tqdm(enumerate(os.listdir(img_dir)), desc='Getting kitti dicts'):

            for j, img_name in tqdm(enumerate(os.listdir(os.path.join(img_dir, folder_name)))):
                if 'png' not in img_name and 'jpg' not in img_name:
                    continue

                record = {}
                filename = os.path.join(img_dir, folder_name, img_name)
                annot_filename = os.path.join(annot_dir, folder_name, img_name.split('.')[0] + '.png')

                annot = np.asarray(Image.open(annot_filename))

                height, width = annot.shape[:]

                record["file_name"] = filename
                record["image_id"] = i + j
                record["height"] = height
                record["width"] = width

                # Identify different patterns indexes
                patterns = list(np.unique(annot))[1:-1]

                objs = []
                for pattern in patterns:

                    # Coordinates of pattern pixels
                    coords = np.argwhere(annot == pattern)

                    # Bounding box of pattern
                    x0, y0 = coords.min(axis=0)
                    x1, y1 = coords.max(axis=0)

                    bbox = [y0, x0, y1, x1]

                    copy = annot.copy()
                    copy[annot == pattern] = 255
                    copy[annot != pattern] = 0
                    copy = np.asarray(copy, np.uint8)

                    contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    # image = cv2.imread(filename)
                    # cv2.drawContours(image, contours, -1, (0,255,0), 1)

                    # plt.imshow(image)
                    # plt.pause(10)

                    contour = [np.reshape(contour, (contour.shape[0], 2)) for contour in contours]
                    contour = np.asarray([item for tree in contour for item in tree])
                    px = contour[:, 0]
                    py = contour[:, 1]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    if len(poly) < 6:
                        continue

                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": map_classes[int(np.floor(annot[coords[0][0]][coords[0][1]] / 1e3))],
                        "iscrowd": 0
                    }

                    objs.append(obj)

                record["annotations"] = objs
                self.dataset_dicts.append(record)

        self.dataset_train = self.dataset_dicts

    def __len__(self):
        return len(self.dataset_dicts)

    def train_val_split(self, split=0):
        random.seed(len(self))
        val_samples = random.choices(np.arange(0, len(self), 1), k=int(len(self) * split))
        self.dataset_train = [dic for dic in self.dataset_dicts if not dic['image_id'] in val_samples]
        self.dataset_val = [dic for dic in self.dataset_dicts if dic['image_id'] in val_samples]

    def get_dicts(self, data_split):
        if data_split is 'train':
            return self.dataset_train
        else:
            return self.dataset_val