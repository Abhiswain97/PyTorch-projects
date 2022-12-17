import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data import train_dl, test_dl
from model import ResNet18
import config as CFG

# the model
model = ResNet18()
model.to(CFG.DEVICE)

# Setting up optimizer and loss
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()


def train_one_epoch(train_dl, model, optimizer, loss):

    avg_train_loss = 0
    running_corrects = 0

    for batch_idx, batch in enumerate(train_dl):

        images, labels = batch

        # move the images & labels to device
        images = images.to(CFG.DEVICE)
        labels = labels.to(CFG.DEVICE)

        # zero_grad the optimizer
        optimizer.zero_grad()

        # get the logits
        logits = model(images)

        print(logits)

        break
        # calculate loss
        loss = criterion(logits, labels)

        # keep track of loss
        avg_train_loss += loss.item() * images.size(0)

        # backpropagate & step
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch loss for batch {batch_idx + 1}: {loss.item()}")

            _, preds = torch.max(logits, 1)
            print(
                f"Accuracy for batch {batch_idx + 1}: {100 * torch.mean(torch.sum(preds == labels))}"
            )


train_one_epoch(train_dl=train_dl, model=model, optimizer=optimizer, loss=criterion)
