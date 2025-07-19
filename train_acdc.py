from __future__ import annotations
import os
from os.path import join
from collections import defaultdict
import torch
import numpy as np
import monai
from monai import data
from monai.metrics import CumulativeAverage
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger
from lr_scheduler import LR_SCHEDULERS
from loss import LOSSES
from eval import eval_single_volume
# from model import build_model
from models import MODELS
from dataset_acdc import ACDCDataset
from torchvision.transforms import transforms
from typing import Callable
import argparse

torch.set_float32_matmul_precision("medium")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW
}

class ACDC(L.LightningModule):
    def __init__(self, name: str, model_name) -> None:
        super(ACDC, self).__init__()
        self.name = name
        if model_name is None:
            model_name = "BiSeNetV1"
        self.num_classes = 4
        self.max_epochs = 300
        self.freeze_encoder_epochs = 0
        self.deep_supervision = False

        self.model = MODELS[model_name](
            in_channels=3,
            num_classes=self.num_classes,
        ).to(device)
        self.model.flops()

        self.build_loss()
        self.tl_metric = CumulativeAverage()
        self.vs_metric = defaultdict(lambda: defaultdict(list))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def prepare_data(self) -> None:
        self.norm_x_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.train_dataset = ACDCDataset(
            base_dir="dataset/acdc",
            split="train",
            norm_x_transform=self.norm_x_transform,
            norm_y_transform=transforms.ToTensor(),
        )

        self.val_dataset = ACDCDataset(
            base_dir="dataset/acdc",
            split="test",
            # norm_x_transform=self.norm_x_transform,
            # norm_y_transform=transforms.ToTensor(),
        )

    def train_dataloader(self) -> data.DataLoader:
        tdl_0 = {
            "batch_size": 32,
            "num_workers": 6,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
            "worker_init_fn": None
        }

        tdl = tdl_0
        logger.info(f"Training dataloader: {tdl}")
        return data.DataLoader(self.train_dataset, **tdl)

    def val_dataloader(self) -> data.DataLoader:
        vdl_0 = {
            "batch_size": 1,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 11,
            "persistent_workers": True
        }

        vdl = vdl_0
        logger.info(f"Validation dataloader: {vdl}")
        return data.DataLoader(self.val_dataset, **vdl)

    def build_loss(self):
        loss_0 = ("DiceCELoss", {
            "ce_weight": 0.4,
            "dc_weight": 0.6,
            "ce_class_weights": None,
            "dc_class_weights": None
        })

        self.loss = LOSSES[loss_0[0]](**loss_0[1])

    @property
    def criterion(self) -> Callable[..., torch.Tensor]:
        return self.loss

    def configure_optimizers(self) -> dict:
        optimizer_0 = ("AdamW", {
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "eps": 1e-8,
            "amsgrad": False,
            "betas": (0.9, 0.999)
        })
        optimizer = OPTIMIZERS[optimizer_0[0]](self.model.parameters(), **optimizer_0[1])

        scheduler_1 = ("CosineAnnealingLR", {
            "T_max": self.max_epochs,
            "eta_min": 1e-6
        })
        scheduler = LR_SCHEDULERS[scheduler_1[0]](optimizer, **scheduler_1[1])

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def log_and_logger(self, name: str, value: ..., **kwargs: ...) -> None:
        self.log(name, value, **kwargs)
        logger.info(f"{name}: {value}")

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        image, label = batch["image"].to(device), batch["label"]
        label = label.to(device)
        if hasattr(self.model, "aux_mode"):
            self.model.aux_mode = "train"
            pred = self.forward(image)
            loss = self.criterion(pred[0], label)
            loss += self.criterion(pred[1], label)
            loss += self.criterion(pred[2], label)
        else:
            pred = self.forward(image)
            loss = self.criterion(pred, label)

        self.log("loss", loss.item(), prog_bar=True)
        self.tl_metric.append(loss.item())
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        tl = self.tl_metric.aggregate().item()
        self.log_and_logger("mean_train_loss", tl)
        self.tl_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        volume, label = batch["image"], batch["label"]
        if hasattr(self.model, "aux_mode"):
            self.model.aux_mode = "eval"
        metric = eval_single_volume(
            model=self.model,
            volume=volume,
            label=label,
            num_classes=self.num_classes,
            output=join(self.name, str(self.current_epoch)),
            patch_size=(224, 224),
            device=device,
            norm_x_transform=getattr(self, "norm_x_transform", None),
        )

        for metric_name, class_metric in metric.items():
            for class_name, value in class_metric.items():
                self.vs_metric[metric_name][class_name].append(np.mean(value))

    def on_validation_epoch_end(self) -> None:
        for metric_name, class_metric in self.vs_metric.items():
            avg_metric = []
            for class_name, value in class_metric.items():
                t = np.mean(value)
                self.log(f"val_{metric_name}_{class_name}", t)
                avg_metric.append(t)
            self.log_and_logger(f"val_mean_{metric_name}", np.mean(avg_metric))
        self.vs_metric = defaultdict(lambda: defaultdict(list))

def train(name: str, model_name) -> None:
    os.makedirs(name, exist_ok=True)
    logger.add(join(name, "training.log"))

    model = ACDC(name, model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=join(name, "checkpoints"),
        monitor="val_mean_dice",
        mode="max",
        filename="{epoch:02d}-{val_mean_dice:.4f}",
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="mean_train_loss",
        mode="min",
        min_delta=0.00,
        patience=15
    )

    trainer = L.Trainer(
        precision=32,
        accelerator=device,
        devices=[0, ],
        max_epochs=model.max_epochs,
        check_val_every_n_epoch=20,
        gradient_clip_val=None,
        default_root_dir=name,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True
    )

    trainer.fit(model, ckpt_path=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R3Net")
    parser.add_argument("--model_name", type=str, required=False)
    args = parser.parse_args()
    model_name = args.model_name
    if model_name is None:
        model_name = "VMUNetR3"
    L.seed_everything(42)
    monai.utils.set_determinism(42)
    train(f"log/acdc/{model_name}", model_name)

