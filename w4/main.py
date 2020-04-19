import os
from pathlib import Path
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from dataset import register_KITTI_MOTS, get_KITTI_MOTS_dicts, val_seqs
from loss import ValidationLoss, plot_validation_loss


def get_qualitative_results(cfg, save_path):
    predictor = DefaultPredictor(cfg)
    samples = random.sample(get_KITTI_MOTS_dicts(val_seqs), 30)
    model_training_metadata = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]
    )
    for i, sample in enumerate(samples):
        img = cv2.imread(sample["file_name"])
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=model_training_metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE,
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(
            save_path / f"inference_{i}.png",
            v.get_image()[:, :, ::-1],
        )


def base_cfg(model_file, save_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ("kitti-mots-train",)
    cfg.DATASETS.TEST = ("kitti-mots-val",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = save_path
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    return cfg


def task_a(model_name, model_file):

    save_path = Path("output/task_a") / model_name
    os.makedirs(save_path, exist_ok=True)
    cfg = base_cfg(model_file, save_path)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    evaluator = COCOEvaluator("kitti-mots-val", cfg, False, output_dir="./output")
    trainer = DefaultTrainer(cfg)
    trainer.test(cfg, model, evaluators=[evaluator])

    get_qualitative_results(cfg, save_path)


def task_b(model_name, model_file):
    save_path = Path("output/task_b") / model_name
    os.makedirs(save_path, exist_ok=True)

    cfg = base_cfg(model_file, save_path)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.SCORE_THRESH = 0.5

    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("kitti-mots-val", cfg, False, output_dir=save_path)
    trainer.model.load_state_dict(val_loss.weights)
    trainer.test(cfg, trainer.model, evaluators=[evaluator])
    plot_validation_loss(cfg, cfg.SOLVER.MAX_ITER, model_name, save_path)

    get_qualitative_results(cfg, save_path)


if __name__ == "__main__":

    register_KITTI_MOTS()

    nets_a = (
        ("MaskRCNN_R_50_C4", "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"),
        ("MaskRCNN_R_50_DC5", "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"),
        ("MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"),
        ("MaskRCNN_R_101_C4", "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"),
        ("MaskRCNN_R_101_DC5", "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"),
        ("MaskRCNN_R_101_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"),
        ("MaskRCNN_R_50_FPN_Cityscapes", "Cityscapes/mask_rcnn_R_50_FPN.yaml"),
    )

    for name, yaml_file in nets_a:
        task_a(name, yaml_file)

    nets_b = (
        ("MaskRCNN_R_50_C4", "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"),
        ("MaskRCNN_R_50_DC5", "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"),
        ("MaskRCNN_R_50_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"),
        ("MaskRCNN_R101-DC5", "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"),
        ("MaskRCNN_R_101_C4", "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"),
        ("MaskRCNN_R_101_FPN", "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"),
        ("MaskRCNN_R_50_FPN_Cityscapes", "Cityscapes/mask_rcnn_R_50_FPN.yaml"),
    )

    # for name, yaml_file in nets_b:
    #     task_b(name, yaml_file)
