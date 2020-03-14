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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader

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
for idx, filename in enumerate(train_onlyimages[0:6000]):
    image_path = train_images_path + '/' + filename
    label_path = train_labels_path + '/' + filename.split('.')[0] + '.txt'
    train.append([image_path, label_path])
val = []
for idx, filename in enumerate(train_onlyimages[6001:]):
    image_path = train_images_path + '/' + filename
    label_path = train_labels_path + '/' + filename.split('.')[0] + '.txt'
    val.append([image_path, label_path])

#CREATE TEST LIST: [[image_path, label_path], ...]
test_onlyimages = [f for f in listdir(test_images_path) if isfile(join(test_images_path, f))]
print('len(onlyimages): ', len(test_onlyimages))
# group together image path and label path
test = []
for idx, filename in enumerate(test_onlyimages):
    image_path = test_images_path + '/' + filename
    label_path = test_labels_path + '/' + filename.split('.')[0] + '.txt'
    test.append([image_path, label_path])

def get_KITTI_MOTS_dicts(data_list):

    dataset_dicts = []

    for i, path in enumerate(data_list):
        #print("-----------------------------------------------------")
        print(str(i) + '/' + str(len(data_list)))
        filename_image = path[0]
        filename_label = path[1]
        # print("------------------------------------")
        #print('filename_image: ', filename_image)
        #print('filename_label: ', filename_label)

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
                    #print(category, category_id)
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

#DatasetCatalog.register("my_dataset", get_KITTI_MOTS_dicts)

for d,x in [("train",train), ("test",test),("val",val)]:
    DatasetCatalog.register("KITTI_MOTS_" + d, lambda x=x: get_KITTI_MOTS_dicts(x))
    #I set the colors, but it's no use. Retry after training.
    #thing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 100, 100), (100, 0, 255), (100, 100, 100)]
    MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'])
KITTI_MOTS_metadata = MetadataCatalog.get("KITTI_MOTS_train")
KITTI_MOTS_metadata = MetadataCatalog.get("KITTI_MOTS_test")


train_dataset_dicts = get_KITTI_MOTS_dicts(train)
idx_train = 0
for d in random.sample(train_dataset_dicts, 2):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=KITTI_MOTS_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("train_random_sample_"+str(idx_train)+".png",vis.get_image()[:, :, ::-1])
    idx_train = idx_train + 1


#TODO: A for loop to iterate on the model and extract evaluation metrics
#It would be needed a for from here til the end, to train on steps and validate the data, so we can track how good it is performing.

#We train the model with the weight initialized.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("KITTI_MOTS_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 9000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


#EVALUATION
evaluator = COCOEvaluator("KITTI_MOTS_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test
