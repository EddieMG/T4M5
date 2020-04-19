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
from dataset import MOTS_Dataloader
from dataset import MOTS_KITTI_Dataloader
from dataset import MOTS_CATEGORIES
from loss import ValidationLoss, draw_loss


def task_a_no_KITTI_training(model_name, model_file, evaluate=True, visualize=True):
    print('Running task A for model', model_name)

    SAVE_PATH = os.path.join('./results_week_5_task_a', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    model_training_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) # Store current model training metadata
    cfg.DATASETS.TRAIN = ('KITTIMOTS_train', )
    cfg.DATASETS.TEST = ('MOTS_train', )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)

    if evaluate:
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        # Evaluation
        print('Evaluating')
        evaluator = COCOEvaluator('MOTS_train', cfg, False, output_dir=SAVE_PATH)
        trainer = DefaultTrainer(cfg)
        trainer.test(cfg, model, evaluators=[evaluator])

    if visualize:
        # Qualitative results: visualize some results
        print('Getting qualitative results')
        predictor = DefaultPredictor(cfg)
        inputs = mots_train()
        inputs = inputs[:20] + inputs[-20:]
        for i, input in enumerate(inputs):
            img = cv2.imread(input['file_name'])
            outputs = predictor(img)
            v = Visualizer(
                img[:, :, ::-1],
                metadata=model_training_metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            cv2.imwrite(os.path.join(SAVE_PATH, 'Inference_' + model_name + '_inf_' + str(i) + '.png'), v.get_image()[:, :, ::-1])

def task_a_KITTI_training(model_name, model_file):
    #model_name = model_name + '_inference'
    print('Running task A for model', model_name)

    SAVE_PATH = os.path.join('./results_week_5_task_a', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('KITTIMOTS_train',)
    cfg.DATASETS.TEST = ('MOTS_train',)
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

    # Training
    print('Training')
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    print('Evaluating')
    evaluator = COCOEvaluator('MOTS_train', cfg, False, output_dir=SAVE_PATH)
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    print('Plotting losses')
    draw_loss(cfg, cfg.SOLVER.MAX_ITER, model_name, SAVE_PATH)

    # Qualitative results: visualize some results
    print('Getting qualitative results')
    predictor = DefaultPredictor(cfg)
    predictor.model.load_state_dict(trainer.model.state_dict())
    inputs = mots_train()
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

def task_b_MOTS_training(model_name, model_file):
    #model_name = model_name + '_inference'
    print('Running task B for model', model_name)

    SAVE_PATH = os.path.join('./results_week_5_task_b', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('MOTS_train',)
    cfg.DATASETS.TEST = ('KITTIMOTS_val',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.SCORE_THRESH = 0.5

    # Training
    print('Training')
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    print('Evaluating')
    evaluator = COCOEvaluator('KITTIMOTS_val', cfg, False, output_dir=SAVE_PATH)
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    print('Plotting losses')
    draw_loss(cfg, cfg.SOLVER.MAX_ITER, model_name, SAVE_PATH)

    # Qualitative results: visualize some results
    print('Getting qualitative results')
    predictor = DefaultPredictor(cfg)
    predictor.model.load_state_dict(trainer.model.state_dict())
    inputs = kitti_val()
    #inputs = inputs[:20] + inputs[-20:]
    inputs = inputs[220:233] + inputs[1995:2100]
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

def task_b_MOTS_and_KITTI_training(model_name, model_file):
    # model_name = model_name + '_inference'
    print('Running task B for model', model_name)

    SAVE_PATH = os.path.join('./results_week_5_task_c', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('MOTS_KITTI_train',)
    cfg.DATASETS.TEST = ('KITTIMOTS_val',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025

    ###################################################################################################################
    #                                               hyperparameters
    ###################################################################################################################
    #cfg.SOLVER.LR_POLICY = 'steps_with_decay'
    #cfg.SOLVER.STEPS = [0, 1000, 2000]
    #cfg.SOLVER.GAMMA = 0.1
    #cfg.DATASETS.TRAIN.USE_FLIPPED = True #Eeste no va
    #cfg.MODEL.RPN.IOU_THRESHOLDS = [0.1, 0.9] #defatults 0.3 and 0.7
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[256, 512]]#default: [[32, 64, 128, 256, 512]]
    #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    ###################################################################################################################
    #                                        End of hyperparameters playing
    ###################################################################################################################
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.SCORE_THRESH = 0.5
    print(cfg)
    # Training
    print('Training')
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    print('Evaluating')
    evaluator = COCOEvaluator('KITTIMOTS_val', cfg, False, output_dir=SAVE_PATH)
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    print('Plotting losses')
    draw_loss(cfg, cfg.SOLVER.MAX_ITER, model_name, SAVE_PATH)

    # Qualitative results: visualize some results
    print('Getting qualitative results')
    predictor = DefaultPredictor(cfg)
    predictor.model.load_state_dict(trainer.model.state_dict())
    inputs = kitti_val()
    # inputs = inputs[:20] + inputs[-20:]
    inputs = inputs[220:233] + inputs[1995:2100]
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
        cv2.imwrite(os.path.join(SAVE_PATH, 'Inference_' + model_name + '_inf_' + str(i) + '.png'),
                    v.get_image()[:, :, ::-1])

if __name__ == '__main__':

    # Loading data KITTI-MOTS
    print('Loading data')
    dataloader_kitti = KITTIMOTS_Dataloader()
    def kitti_train(): return dataloader_kitti.get_dicts(train_flag=True)
    def kitti_val(): return dataloader_kitti.get_dicts(train_flag=False)
    DatasetCatalog.register('KITTIMOTS_train', kitti_train)
    MetadataCatalog.get('KITTIMOTS_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    # Loading data MOTSChallenge
    print('Loading MOTSChallenge data')
    dataloader_mots = MOTS_Dataloader()
    def mots_train(): return dataloader_mots.get_dicts(train_flag=True)
    DatasetCatalog.register('MOTS_train', mots_train)
    MetadataCatalog.get('MOTS_train').set(thing_classes=list(MOTS_CATEGORIES.keys()))

    # Loading data MOTSChallenge and KITTI-MOTS
    print('Loading MOTSChallenge and KITTI data')
    dataloader_mots_kitti = MOTS_KITTI_Dataloader()
    def mots_kitti_train(): return dataloader_mots_kitti.get_dicts(train_flag=True)
    DatasetCatalog.register('MOTS_KITTI_train', mots_kitti_train)
    MetadataCatalog.get('MOTS_KITTI_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    task_a_no_KITTI_training("MaskRCNN_R_50_FPN_COCO", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", evaluate=True, visualize=True)
    task_a_no_KITTI_training("MaskRCNN_R_50_FPN_COCO_Cityscape", "Cityscapes/mask_rcnn_R_50_FPN.yaml", evaluate=True, visualize=True)
    task_a_KITTI_training("MaskRCNN_R_50_FPN_COCO_KITTIMOTS", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    task_a_KITTI_training("MaskRCNN_R_50_FPN_COCO_Cityscapes_KITTI", "Cityscapes/mask_rcnn_R_50_FPN.yaml")

    task_b_MOTS_and_KITTI_training("MaskRCNN_R_50_FPN_COCO_MOTS", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    task_b_MOTS_and_KITTI_training("MaskRCNN_R_50_FPN_COCO_Cityscape_MOTS", "Cityscapes/mask_rcnn_R_50_FPN.yaml")
    task_b_MOTS_and_KITTI_training("MaskRCNN_R_50_FPN_COCO_MOTS_KITTI", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    task_b_MOTS_and_KITTI_training("MaskRCNN_R_50_FPN_COCO_Cityscapes_MOTS_KITTI", "Cityscapes/mask_rcnn_R_50_FPN.yaml")

