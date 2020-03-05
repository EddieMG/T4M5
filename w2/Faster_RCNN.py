import torch, torchvision
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

testset_path = '/home/mcv/datasets/MIT_split/test/'
for directory in os.listdir(testset_path):
    for i in range(4):
        directory = os.path.join(testset_path, directory)
        filename = random.choice(os.listdir(directory))
        path = os.path.join(directory, filename)
        print("random file: ", path)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))