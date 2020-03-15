# %%
from pathlib import Path
from pycocotools.mask import toBbox
from detectron2.structures import BoxMode
from mots_tools_io import (
    load_txt,
)  # download and rename from https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py

base_dir = Path("/data")

# %%


def yield_dicts(img_paths, lbl_path):
    objs = load_txt(lbl_path)

    for img_path in img_paths.iterdir():
        idx = int(img_path.stem)
        if idx not in objs or len(objs[idx]) == 0:
            continue
        record = {
            "file_name": img_path,
            "image_id": idx,
            "height": objs[idx][0].mask["size"][0],
            "width": objs[idx][0].mask["size"][1],
            "annotations": [
                {
                    "bbox": toBbox(obj.mask),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": obj.class_id,
                    "iscrowd": 0,
                }
                for obj in objs[idx]
            ],
        }
        yield record


def get_MOTS_dicts(seq_name="0002"):
    img_paths = base_dir / f"MOTSChallenge/train/images/{seq_name}"
    lbl_path = base_dir / f"MOTSChallenge/train/instances_txt/{seq_name}.txt"

    return list(yield_dicts(img_paths, lbl_path))


def get_KITTI_MOTS_dicts(seq_name="0000"):
    img_paths = base_dir / f"KITTI-MOTS/training/image_02/{seq_name}"
    lbl_path = base_dir / f"KITTI-MOTS/instances_txt/{seq_name}"

    return list(yield_dicts(img_paths, lbl_path))

