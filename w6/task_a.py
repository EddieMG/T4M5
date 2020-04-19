import os
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from dataset import KITTIMOTS_Dataloader
from dataset import KITTI_CATEGORIES
from loss import ValidationLoss, draw_loss
from dataset import OurTrainer

def task_a(model_name, model_file, augmentation=False):
    model_name = model_name + '_inference'
    print('Running task A for model', model_name)

    SAVE_PATH = os.path.join('./results_week_6_task_a', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('KITTIMOTS_train',)
    cfg.DATASETS.TEST = ('KITTIMOTS_val',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.SCORE_THRESH = 0.5
    print(cfg)
    # Training

    print('Training')
    if augmentation:
        print("data augmentation")
        '''resize_factor = 1
        crop_size = [300,300]
        print("resize_factor: ", resize_factor)
        print("crop_size: ", crop_size)'''
        trainer = OurTrainer(cfg)
    else:
        print("NO data augmentation")
        trainer = DefaultTrainer(cfg)

    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    print('Evaluating')
    evaluator = COCOEvaluator('KITTIMOTS_val', cfg, False, output_dir='./output')
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    print('Plotting losses')
    draw_loss(cfg, cfg.SOLVER.MAX_ITER, model_name, SAVE_PATH)

    # Qualitative results: visualize some results
    print('Getting qualitative results')
    predictor = DefaultPredictor(cfg)
    predictor.model.load_state_dict(trainer.model.state_dict())
    inputs = kitti_val()
    inputs = inputs[:20] + inputs[-20:]
    for i, input in enumerate(inputs):
        file_name = input['file_name']
        print('Prediction on image ' + file_name)
        img = cv2.imread(file_name)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=0.8,
            instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(SAVE_PATH, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])



if __name__ == '__main__':
    # Loading data
    print('Loading data')
    dataloader_train = KITTIMOTS_Dataloader()
    dataloader_val = KITTIMOTS_Dataloader()
    def kitti_train(): return dataloader_train.get_dicts(train_flag=True)
    def kitti_val(): return dataloader_val.get_dicts(train_flag=False)
    DatasetCatalog.register('KITTIMOTS_train', kitti_train)
    MetadataCatalog.get('KITTIMOTS_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    task_a("MaskRCNN_R_50_FPN_Cityscapes_Aug3", "Cityscapes/mask_rcnn_R_50_FPN.yaml", augmentation=True)

