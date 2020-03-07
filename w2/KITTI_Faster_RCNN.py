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
from detectron2.data import DatasetCatalog

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

def get_dicts():

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

DatasetCatalog.register("my_dataset", get_dicts)

for d,x in [("train",train), ("test",test)]:
    DatasetCatalog.register("KITTI_" + d, lambda x=x: get_PCB_dict(x))
    MetadataCatalog.get("KITTI_" + d).set(thing_classes)
    #I set the colors, but it's no use. Retry after training.
PCB_metadata = MetadataCatalog.get("PCB_train")
