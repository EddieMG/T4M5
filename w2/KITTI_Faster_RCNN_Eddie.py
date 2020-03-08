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
import json
from os.path import isfile, join

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode


train_images_path = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
train_labels_path = '/home/mcv/datasets/KITTI/training/label_2'

eval_images_path = '/home/mcv/datasets/KITTI/data_object_image_2/mini_train/image_2'
eval_labels_path = '/home/mcv/datasets/KITTI/training/label_2'

test_images_path = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'
test_labels_path = '/home/mcv/datasets/KITTI/training/label_2'

thing_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']


#CREATE TRAIN LIST: [[image_path, label_path], ...]
train_onlyimages = [f for f in listdir(train_images_path) if isfile(join(train_images_path, f))]
print('len(onlyimages): ', len(train_onlyimages))
# group together image path and label path
train = []
for idx, filename in enumerate(train_onlyimages):
    image_path = train_images_path + '/' + filename
    label_path = train_labels_path + '/' + filename.split('.')[0] + '.txt'
    train.append([image_path, label_path])

#CREATE TEST LIST: [[image_path, label_path], ...]
test_onlyimages = [f for f in listdir(test_images_path) if isfile(join(test_images_path, f))]
print('len(onlyimages): ', len(test_onlyimages))
# group together image path and label path
test = []
for idx, filename in enumerate(test_onlyimages):
    image_path = test_images_path + '/' + filename
    label_path = test_labels_path + '/' + filename.split('.')[0] + '.txt'
    test.append([image_path, label_path])

def get_KITTI_dicts(data_list):

    dataset_dicts = []

    for i, path in enumerate(data_list):
        print("-----------------------------------------------------")
        print(str(i) + '/' + str(len(data_list)))
        filename_image = path[0]
        filename_label = path[1]
        # print("------------------------------------")
        print('filename_image: ', filename_image)
        print('filename_label: ', filename_label)

        height, width = cv2.imread(filename_image).shape[:2]
        record = {}
        record['file_name'] = filename_image
        record['image_id'] = i
        record['height'] = height
        record['width'] = width

        objs = []
        try:
            with open(filename_label) as t:
                lines = t.readlines()
                for line in lines:
                    if (line[-1] == "\n"):
                        line_splitted = line[:-1].split(' ')
                    else:
                        line_splitted = line.split(' ')
                    category = line_splitted[0]
                    category_id = thing_classes.index(category)
                    # KITTI BBs are 0-based index: left, top, right, bottom
                    x0 = line_splitted[4]
                    y0 = line_splitted[5]
                    x1 = line_splitted[6]
                    y1 = line_splitted[7]
                    box = list(map(float, [x0, y0, x1, y1]))
                    print(category, category_id)
                    obj = {
                        "bbox": box,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        # "segmentation": [poly], To draw a line, along to ballon
                        # you will need this for mask RCNN
                        "category_id": category_id,
                        "iscrowd": 0
                        # Is Crowd specifies whether the segmentation is for a single object or for a group/cluster of objects.
                    }
                    objs.append(obj)
                record["annotations"] = objs
            dataset_dicts.append(record)
        except:
            print("IMAGE: ", filename_image, "HAS NO LABEL!")

    return dataset_dicts

#DatasetCatalog.register("my_dataset", get_dicts)
#out_path = '/home/grupo04/T4M5/kitti.json'
#json.dump(get_KITTI_dicts(train), open(out_path, 'w'))
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "/home/grupo04/T4M5/kitti.json", train_images_path)

#for d,x in [("train",train), ("test",test)]:
  #  DatasetCatalog.register("KITTI_" + d, lambda x=x: get_PCB_dict(x))
 #   MetadataCatalog.get("KITTI_" + d).set(thing_classes)
    #I set the colors, but it's no use. Retry after training.
#PCB_metadata = MetadataCatalog.get("PCB_train")
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()