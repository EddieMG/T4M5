import copy
import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
from detectron2.engine import HookBase


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.best_loss = float("inf")
        self.weights = None

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )
                if losses_reduced < self.best_loss:
                    self.best_loss = losses_reduced
                    self.weights = copy.deepcopy(self.trainer.model.state_dict())


def plot_validation_loss(cfg, iterations, model_name, savepath):
    val_loss = []
    train_loss = []
    for line in open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "r"):
        result = json.loads(line)
        if ("total_val_loss" in result.keys()) and ("total_loss" in result.keys()):
            val_loss.append(result["total_val_loss"])
            train_loss.append(result["total_loss"])
    val_idx = [int(item) for item in list(np.linspace(0, iterations, len(val_loss)))]
    train_idx = [
        int(item) for item in list(np.linspace(0, iterations, len(train_loss)))
    ]
    plt.figure(figsize=(10, 10))
    plt.plot(val_idx, val_loss, label="Validation Loss")
    plt.plot(train_idx, train_loss, label="Training Loss")
    plt.title("Validation Loss for model " + "{0}".format(model_name))
    plt.xlabel("Iterations")
    plt.ylabel("Validation_Loss")
    plt.grid("True")
    plt.legend()
    plt.savefig(os.path.join(savepath, "validation_loss.png"))
