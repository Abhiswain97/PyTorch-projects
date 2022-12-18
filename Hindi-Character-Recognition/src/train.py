import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data import train_dl, test_dl
from model import ResNet18, HNet
import config as CFG
from tqdm import tqdm
from prettytable import PrettyTable

# the model
model = HNet()
model.to(CFG.DEVICE)

# Setting up optimizer and loss
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()


def train_one_epoch(train_dl, model, optimizer, loss):

    # putting model in train mode
    model.train()

    avg_train_loss = 0
    running_corrects = 0

    for batch in tqdm(train_dl):

        images, labels = batch

        # move the images & labels to device
        images = images.to(CFG.DEVICE)
        labels = labels.to(CFG.DEVICE)

        # zero_grad the optimizer
        optimizer.zero_grad()

        # get the logits
        logits = model(images)

        # calculate loss
        loss = criterion(logits, labels)

        avg_train_loss += loss.item() * labels.size(0)
        _, preds = torch.max(logits, 1)
        running_corrects += torch.sum(preds == labels)

        # backpropagate & step
        loss.backward()
        optimizer.step()

    avg_loss = avg_train_loss / len(train_dl.sampler)
    avg_acc = 100 * (running_corrects / len(train_dl.sampler))

    return avg_loss, avg_acc


def train():

    table = PrettyTable(field_names=["Epoch", "Train Loss", "Train Accuracy"])

    for i in range(10):
        avg_loss, avg_acc = train_one_epoch(
            train_dl=train_dl, model=model, optimizer=optimizer, loss=criterion
        )
        table.add_row(row=[i + 1, round(avg_loss, 3), round(avg_acc.item(), 3)])
        print(table)

    # Save the table to .txt file
    table_string = table.get_string()
    with open("results.txt", "w") as f:
        f.write(table_string)

if __name__ == "__main__":
    train()
