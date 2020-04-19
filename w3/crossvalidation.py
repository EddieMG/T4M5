from itertools import chain
from datasets import split, get_MOTS_dicts, get_KITTI_MOTS_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog


for dicts in zip((get_MOTS_dicts(), get_KITTI_MOTS_dicts())):
    all_train_dicts, test_dicts = split(dicts)
    chunk_size = len(all_train_dicts) // 5
    all_train_dicts = all_train_dicts[: chunk_size * 5]
    split_beg = [i * chunk_size for i in range(6)]
    split_end = split_beg[1:]
    folds = [all_train_dicts[i:j] for i, j in zip(split_beg, split_end)]
    for i in range(5):
        train_dicts = list(chain(*(folds[j] for j in range(5) if j != i)))
        val_dicts = list(folds[i])
        DatasetCatalog.clear()
        DatasetCatalog.register("train", lambda: train_dicts)
        DatasetCatalog.register("validation", lambda: val_dicts)
        DatasetCatalog.register("test", lambda: test_dicts)
        thing_classes = ["Car", "Pedestrian"]
        MetadataCatalog.get("train").set(thing_classes=thing_classes)
        MetadataCatalog.get("validation").set(thing_classes=thing_classes)
        MetadataCatalog.get("test").set(thing_classes=thing_classes)

        # training code goes here
