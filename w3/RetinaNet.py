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

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Select a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


# TO DO INFERENCE ON RANDOM IMAGES
'''testset_path = '/home/mcv/datasets/MIT_split/test/'
for directory in os.listdir(testset_path):
    for i in range(4):
        directory = os.path.join(testset_path, directory)
        filename = random.choice(os.listdir(directory))
        path = os.path.join(directory, filename)
        print("random file: ", path)
        im = cv2.imread(path)
        outputs = predictor(im)
        #outputs["instances"].pred_classes
        #outputs["instances"].pred_boxes
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('predictions_STT_0.5/'+filename, v.get_image()[:, :, ::-1])'''

# TO DO INFERENCE ON A LIST OF IMAGES
images_tested_on_FasterRCNN = [
    '/home/mcv/datasets/KITTI-MOTS/training/image_02/0019/000685.png',
    '/home/mcv/datasets/KITTI-MOTS/training/image_02/0009/000191.png',
    '/home/mcv/datasets/KITTI-MOTS/training/image_02/0000/000069.png'
]

for path in images_tested_on_FasterRCNN:
    print("image: ", path)
    filename = path.split('/')[-1]
    im = cv2.imread(path)
    outputs = predictor(im)
    # outputs["instances"].pred_classes
    # outputs["instances"].pred_boxes
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('images_tested_on_RetinaNet/' + filename, v.get_image()[:, :, ::-1])


