from itertools import chain
from pathlib import Path

import pycocotools._mask as _mask
from imantics import Mask
from pycocotools.mask import toBbox
from tqdm.auto import tqdm
from mots_tools_io import (
    load_txt,
)

from detectron2.structures import BoxMode


base_dir = Path("/home/mcv/datasets")


def yield_dicts(img_dir, lbl_path):
    def to_XYXY(bbox):
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

    def mask_as_poly_list(mask):
        polygons = Mask(_mask.decode([mask])).polygons().points
        polygons = map(lambda p: p.reshape(-1), polygons)  # Detectron requirement
        polygons = filter(lambda p: len(p) >= 6, polygons)  # Detectron requirement
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


def get_KITTI_MOTS_dicts(idxs):
    img_dir = lambda seq_num: base_dir / f"KITTI-MOTS/training/image_02/{seq_num:04d}"
    lbl_path = lambda seq_num: base_dir / f"KITTI-MOTS/instances_txt/{seq_num:04d}.txt"

    try:
        return list(chain(*(yield_dicts(img_dir(i), lbl_path(i)) for i in idxs)))
    except TypeError:
        return list(yield_dicts(img_dir(idxs), lbl_path(idxs)))


import pickle

with open("dicts_t7.pkl", "rb") as f:
    t7dts = pickle.load(f)

dts_path = Path("dicts_t4.pkl")

if dts_path.exists():
    with open(dts_path, "rb") as f:
        dts = pickle.load(f)
else:
    KITTIMOTS_VAL = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    dts = get_KITTI_MOTS_dicts(KITTIMOTS_VAL)
    with open(dts_path, "wb") as f:
        dts = pickle.dump(dts, f)

print("Checking stuff")

def len_sort(l):
    return sorted(l, key=lambda l: len(l))

for d1, d2 in zip(t7dts, tqdm(dts)):
    assert d1["file_name"] == d2["file_name"]
    assert d1["image_id"] == d2["image_id"]
    assert d1["height"] == d2["height"]
    assert d1["width"] == d2["width"]
    assert len(d1["annotations"]) == len(d2["annotations"])
    for as1, as2 in zip(d1["annotations"], d2["annotations"]):
        for a1, a2 in zip(len_sort(as1["segmentation"]), len_sort(as2["segmentation"])):
            try:
                diff = (a1 != a2).sum()
            except AttributeError:
                print(f"Problem {d1['file_name']}")
            if diff:
                print(f"{d1['file_name']}: {diff}/{a1['segmantation'].size}")
