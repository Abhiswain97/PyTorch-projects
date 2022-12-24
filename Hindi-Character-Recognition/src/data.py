from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as tfms
import torch

import config as CFG

# the train & test transforms
transforms = {
    "train": tfms.Compose(
        [
            tfms.PILToTensor(),
            tfms.ConvertImageDtype(torch.float),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tfms.RandomHorizontalFlip(p=0.5),
        ]
    ),
    "test": tfms.Compose(
        [
            tfms.PILToTensor(),
            tfms.ConvertImageDtype(torch.float),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

# creating the datasets
train_ds = ImageFolder(root=CFG.TRAIN_PATH, transform=transforms["train"])
test_ds = ImageFolder(root=CFG.TEST_PATH, transform=transforms["test"])

# Train/val splitting
lengths = [int(len(train_ds) * 0.8), len(train_ds) - int(len(train_ds) * 0.8)]
train_ds, val_ds = random_split(dataset=train_ds, lengths=lengths)

# creating the dataloaders
train_dl = DataLoader(dataset=train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=CFG.BATCH_SIZE)
test_dl = DataLoader(dataset=test_ds, batch_size=CFG.BATCH_SIZE)
