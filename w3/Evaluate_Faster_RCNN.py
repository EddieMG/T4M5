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
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Select a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

folder_images_path = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
folder_label_path = '/home/mcv/datasets/KITTI-MOTS/instances/'

for f, folder in tqdm(enumerate(listdir(folder_images_path))):
    print("--------------------------")
    images_path = join(folder_images_path, folder)
    onlyimages = [ima for ima in listdir(images_path) if isfile(join(images_path, ima))]
    for i, image in enumerate(onlyimages):
        image_path = folder_images_path + folder + "/" + image
        label_path = folder_label_path + folder + "/" + image

        #GET PREDICTION
        im = cv2.imread(image_path)
        outputs = predictor(im)
        print(image_path)
        print(label_path)
        print(outputs)

        #GET GT
        label = np.asarray(Image.open(label_path))
        height, width = label.shape[:]

        patterns = list(np.unique(label))[1:-1]
        for pattern in patterns:
            category_id = int(str(pattern)[0]) - 1
            mask_coord = np.argwhere(label==pattern)
            x0, y0 = mask_coord.min(axis=0)
            x1, y1 = mask_coord.max(axis=0)

        break
    break

for directory in listdir(testset_path):
    for i in range(3):
        directory = join(testset_path, directory)
        filename = random.choice(listdir(directory))
        path = join(directory, filename)
        print("random file: ", path)
        im = cv2.imread(path)
        outputs = predictor(im)
        #print("----------------------------------------")
        #print(outputs)
        #outputs["instances"].pred_classes
        #outputs["instances"].pred_boxes
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('./FRCNN_inference_threshold_05/'+filename+'.png', v.get_image()[:, :, ::-1])
