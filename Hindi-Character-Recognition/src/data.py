from torch.utils.data import DataLoader
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
        ]
    ),
    "test": tfms.PILToTensor(),
}

# creating the datasets
train_ds = ImageFolder(root=CFG.TRAIN_PATH, transform=transforms["train"])
test_ds = ImageFolder(root=CFG.TEST_PATH, transform=transforms["test"])

# creating the dataloaders
train_dl = DataLoader(dataset=train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
test_dl = DataLoader(dataset=test_ds, batch_size=CFG.BATCH_SIZE)
