from pathlib import Path
from itertools import chain
import numpy as np
import cv2
from pycocotools.mask import toBbox
from pycocotools import coco
from pycocotools import cocoeval
from detectron2.structures import BoxMode
from mots_tools_io import (
    load_txt,
)  # download and rename from https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py
import pycocotools
from pycocotools import coco
import pycocotools._mask as _mask
from imantics import Polygons, Mask

base_dir = Path("/home/mcv/datasets")

'''def yield_mask_dicts(img_dir, lbl_path):
    def to_XYXY(bbox):
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox
    print("----------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------")
    print("lbl_path: ", lbl_path)
    print("----------------------------------------------------------------------------")
    objs = load_txt(lbl_path)
    for img_path in img_dir.iterdir():
        print("img_path: ", img_path)

        if img_path.suffix not in [".jpg", ".png"]:
            continue
        idx = int(img_path.stem)
        if idx not in objs or len(objs[idx]) == 0:
            continue
        for index in range(len(objs)):
            print(objs[index][0].mask)
            print(isinstance(objs[index][0].mask, dict))
            break

        record = { #https://github.com/facebookresearch/detectron2/blob/master/docs/tutorials/datasets.md
            "file_name": str(img_path),
            "image_id": idx,
            "height": objs[idx][0].mask["size"][0],
            "width": objs[idx][0].mask["size"][1],
            "annotations": [
                {
                    "bbox": to_XYXY(toBbox(obj.mask)).tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": obj.mask,
                    "category_id": 2 if obj.class_id == 1 else 0,
                    "iscrowd": 0,
                }
                for obj in objs[idx]
                if obj.class_id != 10
            ],
        }

        if str(img_path) == "/home/mcv/datasets/KITTI-MOTS/training/image_02/0000/000128.png":
            print("*****************************************************************")
            print(record)
            print("*****************************************************************")
        print("---")
        yield record'''


def yield_mask_dicts(img_dir, lbl_path):
    def to_XYXY(bbox):
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

    # print("----------------------------------------------------------------------------")
    # print("----------------------------------------------------------------------------")
    # print("----------------------------------------------------------------------------")
    # print("lbl_path: ", lbl_path)
    # print("----------------------------------------------------------------------------")
    objs = load_txt(lbl_path)

    for img_path in img_dir.iterdir():
        # print("img_path: ", img_path)
        if img_path.suffix not in [".jpg", ".png"]:
            continue
        idx = int(img_path.stem)
        if idx not in objs or len(objs[idx]) == 0:
            continue

        record = {}  # https://github.com/facebookresearch/detectron2/blob/master/docs/tutorials/datasets.md
        record["file_name"] = str(img_path)
        record["image_id"] = idx
        record["height"] = objs[idx][0].mask["size"][0]
        record["width"] = objs[idx][0].mask["size"][1]

        annotations = []
        for obj in objs[idx]:
            if obj.class_id != 10:

                rle = {
                    'counts': obj.mask["counts"],
                    'size': obj.mask["size"]
                }

                mask = coco.maskUtils.decode(rle)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                seg = [[int(i) for i in c.flatten()] for c in contours]
                seg = [s for s in seg if len(s) >= 6]
                if not seg:
                    continue

                annotation = {
                    "bbox": to_XYXY(toBbox(obj.mask)).tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": seg,
                    "category_id": 2 if obj.class_id == 1 else 0,
                    "iscrowd": 0,
                }

                annotations.append(annotation)

        record["annotations"] = annotations

        #if str(img_path) == "/home/mcv/datasets/KITTI-MOTS/training/image_02/0000/000128.png":
            #print("*****************************************************************")
            #print(record)
            #print("*****************************************************************")
        # print("---")
        yield record


'''def yield_mask_dicts(img_dir, lbl_path):
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
                    "segmentation": [poly.reshape(-1) for poly in Mask(_mask.decode([obj.mask])).polygons().points],
                    "category_id": obj.class_id,
                }
                for obj in objs[idx]
                if obj.class_id != 10
            ],
        }

        yield record'''


def get_KITTI_MOTS_dicts():
    img_dir = lambda seq_num: base_dir / f"KITTI-MOTS/training/image_02/{seq_num:04d}"
    lbl_path = lambda seq_num: base_dir / f"KITTI-MOTS/instances_txt/{seq_num:04d}.txt"

    return list(chain(*(yield_mask_dicts(img_dir(i), lbl_path(i)) for i in range(20))))


def split(dicts, train=60, val=20, test=20):
    np.random.shuffle(dicts)
    weights = train + val + test
    train_split, val_split = (
        train * len(dicts) // weights,
        (train + val) * len(dicts) // weights,
    )
    return (dicts[:train_split], dicts[train_split:val_split], dicts[val_split])


def register_datasets():
    """Registers {kitti-mots, mots}_{train, validation, test} datasets (all 6 combinations)
    """
    names = [
        name + suffix
        for name in ("kitti-mots", "mots")
        for suffix in ("_train", "_validation", "_test")
    ]
    dicts_all = chain(split(get_KITTI_MOTS_dicts()), split(get_MOTS_dicts()))
    thing_classes = ["Pedestrian", "fts", "Car"]
    for name, dicts in zip(names, dicts_all):
        DatasetCatalog.register(name, lambda: dicts)
        MetadataCatalog.get(name).set(thing_classes=thing_classes)


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

    thing_classes = ["Pedestrian", "Car"]

    # DatasetCatalog.register("mots", get_MOTS_dicts)
    # MetadataCatalog.get("mots").set(thing_classes=thing_classes)

    DatasetCatalog.register("kitti-mots", get_KITTI_MOTS_dicts)
    MetadataCatalog.get("kitti-mots").set(thing_classes=thing_classes)

    # dataset_dicts = get_MOTS_dicts()

    # We train the model with the weight initialized.
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("kitti-mots",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 100  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # cfg.INPUT.MASK_FORMAT = 'rle'
    # cfg.INPUT.MASK_FORMAT='bitmask'

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
