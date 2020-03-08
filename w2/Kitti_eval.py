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

train_images_path = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
train_labels_path = '/home/mcv/datasets/KITTI/training/label_2'

thing_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']


#CREATE TRAIN LIST: [[image_path, label_path], ...]
train_onlyimages = [f for f in listdir(train_images_path) if isfile(join(train_images_path, f))]
print('len(onlyimages): ', len(train_onlyimages))
# group together image path and label path
train = []
for idx, filename in enumerate(train_onlyimages[0:7459]):
    image_path = train_images_path + '/' + filename
    label_path = train_labels_path + '/' + filename.split('.')[0] + '.txt'
    train.append([image_path, label_path])
    
val = []
for idx, filename in enumerate(train_onlyimages[7460:7480]):
    val_image_path = train_images_path + '/' + filename
    val_label_path = train_labels_path + '/' + filename.split('.')[0] + '.txt'
    val.append([val_image_path, val_label_path])

def get_KITTI_dicts(data_list):

    dataset_dicts = []

    for i, path in enumerate(data_list):
        print("-----------------------------------------------------")
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

#DatasetCatalog.register("my_dataset", get_KITTI_dicts)

for d,x in [("train",train),("val",val)]:
    DatasetCatalog.register("KITTI_" + d, lambda x=x: get_KITTI_dicts(x))
    #I set the colors, but it's no use. Retry after training.
    #thing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 100, 100), (100, 0, 255), (100, 100, 100)]
    MetadataCatalog.get("KITTI_" + d).set(thing_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'])
KITTI_metadata = MetadataCatalog.get("KITTI_train")



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
#We set the model with the latest weights of the training and we set the predictor
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") #Treu instancies
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_9000.pth") #no treu instancies
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.DATASETS.TEST = ("KITTI_val", )
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  
#predictor = DefaultPredictor(cfg)
cfg.DATASETS.TRAIN = ("KITTI_train",)
trainer = DefaultTrainer(cfg) 


#here we extract from the validation data a couple of images to check how the training is going, this part should be skipped on the for loop of the training.
#from detectron2.utils.visualizer import ColorMode
#dataset_dicts = get_KITTI_dicts(val)
#idx_train = 0

#for d in dataset_dicts:
    #im = cv2.imread(d["file_name"])
    #outputs = predictor(im)
    #print(outputs)
   # visualizer = Visualizer(im[:, :, ::-1], metadata=KITTI_metadata, scale=0.5)
  #  visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
 #   cv2.imwrite("predictions_sample_" + str(idx_train) + ".png", visualizer.get_image()[:, :, ::-1])
#    idx_train = idx_train + 1


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("KITTI_val", cfg, False, output_dir="./output2/")
val_loader = build_detection_test_loader(cfg, "KITTI_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
'''for d in dataset_dicts: 
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)     '''
    

#We should save here the prediction in json format to then evaluate with the cpp. This should be done at N iterations.
#out_path = '/home/grupo04/T4M5/predictions.json'
#json.dump(outputs, open(out_path, 'w'))