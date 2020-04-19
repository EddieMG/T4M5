from pathlib import Path
from itertools import chain
import numpy as np
from pycocotools.mask import toBbox
from pycocotools import coco
from pycocotools import cocoeval
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from mots_tools_io import (
    load_txt,
)  # download and rename from https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py

base_dir = Path("/home/mcv/datasets")


def yield_dicts(img_dir, lbl_path):
    def to_XYXY(bbox):
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

    objs = load_txt(lbl_path)
    for img_path in img_dir.iterdir():
        if img_path.suffix not in [".jpg", ".png"]:
            continue
        idx = int(img_path.stem)
        if idx not in objs or len(objs[idx]) == 0:
            continue

        record = {
            "file_name": str(img_path),
            "image_id": idx,
            "height": objs[idx][0].mask["size"][0],
            "width": objs[idx][0].mask["size"][1],
            "annotations": [
                {
                    "bbox": to_XYXY(toBbox(obj.mask)).tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "category_id": obj.class_id - 1,
                    "category_id": 2 if obj.class_id == 1 else 0,
                    "iscrowd": 0,
                }
                for obj in objs[idx]
                if obj.class_id != 10
            ],
        }

        yield record


def get_MOTS_dicts():
    img_dir = lambda seq_num: base_dir / f"MOTSChallenge/train/images/{seq_num:04d}"
    lbl_path = (
        lambda seq_num: base_dir
        / f"MOTSChallenge/train/instances_txt/{seq_num:04d}.txt"
    )

    return list(chain(*(yield_dicts(img_dir(i), lbl_path(i)) for i in (2, 5, 9, 11))))


def get_KITTI_MOTS_dicts():
    img_dir = lambda seq_num: base_dir / f"KITTI-MOTS/training/image_02/{seq_num:04d}"
    lbl_path = lambda seq_num: base_dir / f"KITTI-MOTS/instances_txt/{seq_num:04d}.txt"

    return list(chain(*(yield_dicts(img_dir(i), lbl_path(i)) for i in range(20))))


def split(dicts, train=80, test=20):
    np.random.shuffle(dicts)
    train_split = train * len(dicts) // (train + test)
    return (dicts[:train_split], dicts[train_split:])


if __name__ == "__main__":
    import random
    import cv2
    import os
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.utils.visualizer import Visualizer
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
    from detectron2.data import build_detection_test_loader
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer

    thing_classes = ["Pedestrian", "WTF", "Car"]

    DatasetCatalog.register("mots", get_MOTS_dicts)
    MetadataCatalog.get("mots").set(thing_classes=thing_classes)

    DatasetCatalog.register("kitti-mots", lambda: get_KITTI_MOTS_dicts())
    MetadataCatalog.get("kitti-mots").set(thing_classes=thing_classes)


    # We train the model with the weight initialized.
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("kitti-mots",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 6000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # EVALUATION
    evaluator = COCOEvaluator("kitti-mots", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "kitti-mots")
    inference_on_dataset(trainer.model, val_loader, evaluator)
    # another equivalent way is to use trainer.test


    '''for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=MetadataCatalog.get("mots"), scale=0.5
        )
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("here", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''