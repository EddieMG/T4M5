from pathlib import Path
from itertools import chain
import numpy as np
from pycocotools.mask import toBbox
from detectron2.structures import BoxMode
from mots_tools_io import (
    load_txt,
)  # download and rename from https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py

base_dir = Path("/data")


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
                    "bbox": to_XYXY(toBbox(obj.mask)),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": obj.class_id - 1,
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
    thing_classes = ["Car", "Pedestrian"]
    for name, dicts in zip(names, dicts_all):
        DatasetCatalog.register(name, lambda: dicts)
        MetadataCatalog.get(name).set(thing_classes=thing_classes)


if __name__ == "__main__":
    import random
    import cv2
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.utils.visualizer import Visualizer

    thing_classes = ["Car", "Pedestrian"]
    DatasetCatalog.register("mots", get_MOTS_dicts)
    DatasetCatalog.register("kitti-mots", get_MOTS_dicts)
    MetadataCatalog.get("mots").set(thing_classes=thing_classes)
    MetadataCatalog.get("kitti-mots").set(thing_classes=thing_classes)

    dataset_dicts = get_KITTI_MOTS_dicts()
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=MetadataCatalog.get("mots"), scale=0.5
        )
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("here", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
