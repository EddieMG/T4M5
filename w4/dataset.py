from itertools import chain
from pathlib import Path
from types import Iterable, Union
from functools import lru_cache

import pycocotools._mask as _mask
from imantics import Mask
from pycocotools.mask import toBbox
from mots_tools_io import (
    load_txt,
)

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

base_dir = Path("/home/mcv/datasets")
val_seqs = [2, 6, 7, 8, 10, 13, 14, 16, 18]
train_seqs = list(filter(lambda i: i not in val_seqs, range(20)))
categories = {
    'None': 0,
    'Car': 1,
    'Pedestrian': 2
}


def yield_dicts(img_dir: Path, lbl_path: Path):
    def to_XYXY(bbox):
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

    def mask_as_poly_list(mask):
        polygons = Mask(_mask.decode([mask])).polygons().points
        # Detectron requirements
        polygons = map(lambda p: p.reshape(-1).tolist(), polygons)
        polygons = filter(lambda p: len(p) >= 6, polygons)
        return list(polygons)

    idx_base = int(img_dir.stem) * 1000
    objs = load_txt(lbl_path)
    for img_path in sorted(list(img_dir.iterdir())):
        if img_path.suffix not in [".jpg", ".png"]:
            continue
        idx = int(img_path.stem)
        if idx not in objs or len(objs[idx]) == 0:
            continue

        record = {
            "file_name": str(img_path),
            "image_id": idx + idx_base,
            "height": objs[idx][0].mask["size"][0],
            "width": objs[idx][0].mask["size"][1],
            "annotations": [
                {
                    "bbox": to_XYXY(toBbox(obj.mask)).tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": mask_as_poly_list(obj.mask),
                    "category_id": obj.class_id,
                }
                for obj in objs[idx]
                if obj.class_id != 10
            ],
        }

        yield record


@lru_cache(max_size=4)
def get_KITTI_MOTS_dicts_from_tuple(idxs: tuple(int)):
    img_dir = lambda seq_num: base_dir / f"KITTI-MOTS/training/image_02/{seq_num:04d}"  # noqa
    lbl_path = lambda seq_num: base_dir / f"KITTI-MOTS/instances_txt/{seq_num:04d}.txt"  # noqa
    return list(chain(*(yield_dicts(img_dir(i), lbl_path(i)) for i in idxs)))


def get_KITTI_MOTS_dicts(idxs: Union[Iterable, int]):
    try:
        return get_KITTI_MOTS_dicts_from_tuple(tuple(idxs))
    except TypeError:
        return get_KITTI_MOTS_dicts_from_tuple((idxs, ))


def register_KITTI_MOTS():
    DatasetCatalog.register('kitti-mots-train', lambda: get_KITTI_MOTS_dicts(train_seqs))
    DatasetCatalog.register('kitti-mots-val', lambda: get_KITTI_MOTS_dicts(val_seqs))
    MetadataCatalog.get('kitti-mots-train').set(thing_classes=list(categories.keys()))
    MetadataCatalog.get('kitti-mots-val').set(thing_classes=list(categories.keys()))
