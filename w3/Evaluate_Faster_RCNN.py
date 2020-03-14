import torch, torchvision
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from sklearn.metrics import precision_score
# import some common libraries
import numpy as np
import cv2
from PIL import Image
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

coco_classes = MetadataCatalog.get("coco_2017_val").thing_classes
print("coco_classes: ", coco_classes)
folder_images_path = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
folder_label_path = '/home/mcv/datasets/KITTI-MOTS/instances/'

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

for f, folder in tqdm(enumerate(listdir(folder_images_path))):
    print(str(f)+"/"+str(len(listdir(folder_images_path))))
    images_path = join(folder_images_path, folder)
    onlyimages = [ima for ima in listdir(images_path) if isfile(join(images_path, ima))]
    for i, image in enumerate(onlyimages):
        image_path = folder_images_path + folder + "/" + image
        label_path = folder_label_path + folder + "/" + image




        #Use this to create the .txt files and then use those files to do the get dict function.
        #GET GT from the labels .png
        label = np.asarray(Image.open(label_path))
        height, width = label.shape[:]
        print(height,width)
        patterns = list(np.unique(label))[1:-1]
        labels_pedestrian = []
        labels_car = []
        #tensor(torch.Tensor): float matrix of Nx4.Each row is (x0, y0, x1, y1) for boxes object
        for pattern in patterns:
            category_id = int(str(pattern)[0]) - 1
            mask_coord = np.argwhere(label==pattern)
            x0, y0 = mask_coord.min(axis=0)
            x1, y1 = mask_coord.max(axis=0)
            box = np.array([y0, x0, y1, x1])
            if category_id == 1:
                labels_pedestrian.append(box)
            else:
                labels_car.append(box)
            #print(box)
        #print(labels_pedestrian)
        #print(labels_car)



        #IGNORE FROM HERE TO THE BOTTOM

        # GET PREDICTION
        im = cv2.imread(image_path)
        outputs = predictor(im)
        print(image_path)
        print(label_path)
        print(outputs['instances'])
        print("len(outputs['instances']): ", len(outputs['instances']))

        for p in range(len(outputs['instances'])):
            coco_class_id = outputs['instances'].pred_classes[p].item()
            print(coco_class_id)
            #We only want person/pedestrian (class 0) and car (class 2)
            if coco_class_id == 0 :
                bbox = outputs['instances'].pred_boxes[p].tensor.cpu().numpy()
                #prediction["Bbox"] = outputs['instances'].pred_boxes[p].tensor
                trybox= np.array(bbox[0])
                for array in labels_pedestrian:
                    print(type(array))
                    print(type(trybox))
                    ap_pedestrian = bb_intersection_over_union(array, trybox) #just trying to get the evaluation manually
                    print("IOU= ", ap_pedestrian)
            elif coco_class_id == 2:
                bbox = outputs['instances'].pred_boxes[p].tensor.cpu().numpy()
                #prediction["Bbox"] = outputs['instances'].pred_boxes[p].tensor
                trybox= np.array(bbox[0])
                for array in labels_car:
                    print(type(array))
                    print(type(trybox))
                    print(array)
                    print(trybox)
                    ap_car = bb_intersection_over_union(trybox,array) #just trying to get the evaluation manually
                    print("IOU= ",ap_car)

        #print(prediction['Bbox'])
        #bbox= prediction['Bbox'].cpu().numpy()
        #print(bbox)



        break
    break

'''for directory in listdir(testset_path):
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
        cv2.imwrite('./FRCNN_inference_threshold_05/'+filename+'.png', v.get_image()[:, :, ::-1])'''
