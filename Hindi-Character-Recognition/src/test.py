import torch
import torch.nn as nn
from model import model
from data import test_dl, test_ds
from tqdm import tqdm

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

running_corrects = 0

for images, labels in tqdm(test_dl):

    outputs = model(images)

    loss = nn.CrossEntropyLoss()(outputs, labels)

    _, preds = torch.max(outputs, 1)

    running_corrects += torch.sum(preds == labels)

print(f"Test Accuracy: {round(running_corrects.double()/len(test_ds) * 100, 3)}%")
