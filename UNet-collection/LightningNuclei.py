from albumentations import pytorch
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import CenterCrop

import numpy as np
from cv2 import cv2, transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl

from torchsummary import summary
from pathlib import Path
from UNet import UNet_2, AttentionUNet

import argparse


class NucleiData(Dataset):
    def __init__(
        self, data_dir="./data/data-science-bowl-2018/stage1_train/", transforms=None
    ):
        train_dir = Path(data_dir)
        self.images = list(train_dir.glob("*/images/*.png"))
        self.masks = list(train_dir.glob("*/masks/*.*"))
        self.transforms = transforms

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx].as_posix(), cv2.IMREAD_COLOR)
        mask = self.get_mask(self.images[idx].parent.parent.glob("masks/*.*"))

        if self.transforms is not None:
            transform = self.transforms(image=image)

        transformed_image = transform["image"]
        transformed_mask = ToTensorV2()(image=mask)

        return transformed_image, transformed_mask["image"]

    def get_mask(self, masks_gen):
        H, W = 256, 256
        target_mask = np.zeros((H, W, 1), dtype=np.uint8)
        for mask in masks_gen:
            curr_mask = cv2.imread(mask.as_posix(), cv2.IMREAD_GRAYSCALE)
            transform = A.Resize(height=H, width=W)(image=curr_mask)
            mask_ = np.expand_dims(transform["image"], axis=-1)
            target_mask = np.maximum(target_mask, mask_)
        return target_mask

    def __len__(self):
        return len(self.images)


class NucleiDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.image_transforms = A.Compose(
            [A.Resize(256, 256), A.Normalize(), A.pytorch.ToTensorV2()]
        )
        self.dims = (3, 256, 256)

    def setup(self, stage) -> None:
        if stage == "fit" or stage is None:
            data = NucleiData(transforms=self.image_transforms)
            lengths = [int(len(data) * 0.8), int(len(data) * 0.2)]
            self.train_data, self.val_data = random_split(data, lengths)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=8, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=8, num_workers=8)


class LitNuclei(pl.LightningModule):
    def __init__(self, use_attention=True):
        super(LitNuclei, self).__init__()
        self.model = AttentionUNet(3, 1) if use_attention else UNet_2(3, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch

        preds = self.forward(image)

        loss = F.binary_cross_entropy_with_logits(input=preds, target=mask.float())

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch

        preds = self.forward(image)

        loss = F.binary_cross_entropy_with_logits(input=preds, target=mask.float())

        self.log("val_loss", loss)

        return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Train Unet from scratch model for Nuclei Segmentation",
    )

    parser.add_argument(
        "attn",
        default=True,
        type=bool,
        help="Specify it to `False` if you want to use Original UNet and not Attention-UNet",
    )
    args = parser.parse_args()

    model = LitNuclei(use_attention=args.attn)
    dm = NucleiDataModule()
    
    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=True,
        gpus=1,
        max_epochs=3,
    )

    trainer.fit(model, datamodule=dm)
