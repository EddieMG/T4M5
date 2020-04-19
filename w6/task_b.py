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
from time import sleep
from dataset import KITTIMOTS_Dataloader,Virtual_Real_KITTI
from dataset import VirtualKITTI_Dataloader
from dataset import KITTI_CATEGORIES
from loss import ValidationLoss, draw_loss
from dataset import OurTrainer

def task_b(model_name, model_file, percentage, augmentation=False):

    try:
        dataloader_train_v_r = Virtual_Real_KITTI()
        def virtual_real_kitti(): return dataloader_train_v_r.get_dicts(percentage)
        DatasetCatalog.register('VirtualReal', virtual_real_kitti)
        MetadataCatalog.get('VirtualReal').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    except:
        print("VirtualReal already defined!")

    model_name = model_name + '_inference'
    print('Running task B for model', model_name)

    SAVE_PATH = os.path.join('./results_week_6_task_b', model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load model and configuration
    print('Loading Model')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ('VirtualReal',)
    cfg.DATASETS.TEST = ('KITTIMOTS_test',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH

    #load saved model
    '''checkpoint = '/home/grupo04/jobs_w6/results_week_6_task_b/MaskRCNN_R_50_FPN_Cityscapes_2_inference/model_final.pth'
    last_checkpoint = torch.load(checkpoint)
    new_path = checkpoint.split('.')[0] + '_modified.pth'
    last_checkpoint['iteration'] = -1
    torch.save(last_checkpoint, new_path)
    cfg.MODEL.WEIGHTS = new_path'''

    #load a model form detectron2 model zoo
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
    evaluator = COCOEvaluator('KITTIMOTS_test', cfg, False, output_dir='./output')
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
    vKITTI_dataloader = VirtualKITTI_Dataloader()
    dataloader_train = KITTIMOTS_Dataloader()


    dataloader_val = KITTIMOTS_Dataloader()
    dataloader_test = KITTIMOTS_Dataloader()

    def virtual_kitti(): return vKITTI_dataloader.get_dicts()
    def kitti_train(): return dataloader_train.get_dicts(ds_type="train")
    def kitti_val(): return dataloader_val.get_dicts(ds_type="validation")
    def kitti_test(): return dataloader_test.get_dicts(ds_type="test")


    DatasetCatalog.register('VirtualKITTI_train', virtual_kitti)
    MetadataCatalog.get('VirtualKITTI_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    DatasetCatalog.register('KITTIMOTS_train', kitti_train)
    MetadataCatalog.get('KITTIMOTS_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_test', kitti_test)
    MetadataCatalog.get('KITTIMOTS_test').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    task_b("MaskRCNN_R_50_FPN_Cityscapes_3", "Cityscapes/mask_rcnn_R_50_FPN.yaml", 5, augmentation=False)
    task_b("MaskRCNN_R_50_FPN_Cityscapes_E2_1", "Cityscapes/mask_rcnn_R_50_FPN.yaml", 5, augmentation=False)
    task_b("MaskRCNN_R_50_FPN_Cityscapes_E2_2_rand", "Cityscapes/mask_rcnn_R_50_FPN.yaml", 4, augmentation=False)
    task_b("MaskRCNN_R_50_FPN_Cityscapes_E2_3_rand", "Cityscapes/mask_rcnn_R_50_FPN.yaml", 3, augmentation=False)
    task_b("MaskRCNN_R_50_FPN_Cityscapes_E2_4_rand", "Cityscapes/mask_rcnn_R_50_FPN.yaml", 2, augmentation=False)
    task_b("MaskRCNN_R_50_FPN_Cityscapes_E2_5_rand", "Cityscapes/mask_rcnn_R_50_FPN.yaml", 1, augmentation=False)
    task_b("MaskRCNN_R_50_FPN_Cityscapes_E2_6_rand", "Cityscapes/mask_rcnn_R_50_FPN.yaml", 0.5, augmentation=False)
