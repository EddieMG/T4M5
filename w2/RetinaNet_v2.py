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
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Select a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
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
 '/home/mcv/datasets/MIT_split/test/forest/nat324.jpg',
 '/home/mcv/datasets/MIT_split/test/forest/for60.jpg',
 '/home/mcv/datasets/MIT_split/test/forest/land222.jpg',
 '/home/mcv/datasets/MIT_split/test/forest/for3.jpg',
 '/home/mcv/datasets/MIT_split/test/tallbuilding/urban1059.jpg',
 '/home/mcv/datasets/MIT_split/test/tallbuilding/art1728.jpg',
 '/home/mcv/datasets/MIT_split/test/tallbuilding/a462093.jpg',
 '/home/mcv/datasets/MIT_split/test/tallbuilding/urb777.jpg',
 '/home/mcv/datasets/MIT_split/test/inside_city/urb781.jpg',
 '/home/mcv/datasets/MIT_split/test/inside_city/par40.jpg',
 '/home/mcv/datasets/MIT_split/test/inside_city/art649.jpg',
 '/home/mcv/datasets/MIT_split/test/inside_city/urb376.jpg',
 '/home/mcv/datasets/MIT_split/test/coast/nat472.jpg',
 '/home/mcv/datasets/MIT_split/test/coast/nat850.jpg',
 '/home/mcv/datasets/MIT_split/test/coast/n708050.jpg',
 '/home/mcv/datasets/MIT_split/test/coast/cdmc951.jpg',
 '/home/mcv/datasets/MIT_split/test/mountain/nat41.jpg',
 '/home/mcv/datasets/MIT_split/test/mountain/land30.jpg',
 '/home/mcv/datasets/MIT_split/test/mountain/n371077.jpg',
 '/home/mcv/datasets/MIT_split/test/mountain/natu668.jpg',
 '/home/mcv/datasets/MIT_split/test/street/boston32.jpg',
 '/home/mcv/datasets/MIT_split/test/street/par20.jpg',
 '/home/mcv/datasets/MIT_split/test/street/boston68.jpg',
 '/home/mcv/datasets/MIT_split/test/street/urb19.jpg',
 '/home/mcv/datasets/MIT_split/test/Opencountry/land703.jpg',
 '/home/mcv/datasets/MIT_split/test/Opencountry/natu540.jpg',
 '/home/mcv/datasets/MIT_split/test/Opencountry/nat965.jpg',
 '/home/mcv/datasets/MIT_split/test/Opencountry/land687.jpg',
 '/home/mcv/datasets/MIT_split/test/highway/gre36.jpg',
 '/home/mcv/datasets/MIT_split/test/highway/gre475.jpg',
 '/home/mcv/datasets/MIT_split/test/highway/bost172.jpg',
 '/home/mcv/datasets/MIT_split/test/highway/bost297.jpg']

for path in images_tested_on_FasterRCNN:
    print("image: ", path)
    filename = path.split('/')[-1]
    im = cv2.imread(path)
    outputs = predictor(im)
    # outputs["instances"].pred_classes
    # outputs["instances"].pred_boxes
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im_path = 'images_tested_on_RetinaNet_v2/' + filename
    print("writing " + im_path+ "...")
    cv2.imwrite(im_path, v.get_image()[:, :, ::-1])


