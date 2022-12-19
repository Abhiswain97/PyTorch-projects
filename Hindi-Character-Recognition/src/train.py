import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from data import train_dl, val_dl, train_ds, val_ds, DataLoader
from model import model, nn
import config as CFG
from tqdm import tqdm
from prettytable import PrettyTable
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict
import time
import logging


# Set up logger
logging.basicConfig(
    filename="train.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filemode="w",
)


best_acc = 0.0


def run_one_epoch(
    ds_sizes: Dict[str, int],
    dataloaders: Dict[str, DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: nn.Module,
):
    """
    Run one complete train-val loop

    Parameter
    ---------

    ds_sizes: Dictionary containing dataset sizes
    dataloaders: Dictionary containing dataloaders
    model: The model
    optimizer: The optimizer
    loss: The loss

    Returns
    -------

    metrics: Dictionary containing Train(loss/accuracy) &
             Validation(loss/accuracy)

    """
    global best_acc

    metrics = {}

    for phase in ["train", "val"]:
        logging.info(f"{phase} phase")

        if phase == "train":
            model.train()
        else:
            model.eval()

        avg_loss = 0
        running_corrects = 0

        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloaders[phase], total=len(dataloaders[phase]))
        ):

            images = images.to(CFG.DEVICE)
            labels = labels.to(CFG.DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Track history if in phase == "train"
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            avg_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels)

            if batch_idx % CFG.INTERVAL == 0:
                corrects = torch.sum(preds == labels)

                logging.info(f"{phase} loss = {loss.item()}")
                logging.info(f"{phase} accuracy = {100 * corrects/CFG.BATCH_SIZE}")

        epoch_loss = avg_loss / ds_sizes[phase]
        epoch_acc = running_corrects.double() / ds_sizes[phase]

        # save best model wts
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, "best_model.pt")

        # Metrics tracking
        if phase == "train":
            metrics["train_loss"] = round(epoch_loss, 3)
            metrics["train_acc"] = round(100 * epoch_acc.item(), 3)
        else:
            metrics["val_loss"] = round(epoch_loss, 3)
            metrics["val_acc"] = round(100 * epoch_acc.item(), 3)

    return metrics


if __name__ == "__main__":

    parser = ArgumentParser(description="Train model for Hindi Character Recognition")
    parser.add_argument("--epochs", type=int, help="number of epochs", default=5)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    CFG.EPOCHS = args.epochs
    CFG.LR = args.lr

    # table
    table = PrettyTable(
        field_names=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
    )

    # the model
    model.to(CFG.DEVICE)

    # Setting up optimizer and loss
    optimizer = Adam(model.parameters(), lr=CFG.LR)
    criterion = CrossEntropyLoss()

    dataloaders = {"train": train_dl, "val": val_dl}
    ds_sizes = {"train": len(train_ds), "val": len(val_ds)}

    detail = f"""
    Training details: 
    ------------------------    
    Model: {model._get_name()}
    Epochs: {CFG.EPOCHS}
    Optimizer: {type(optimizer).__name__}
    Loss: {criterion._get_name()}
    Learning Rate: {CFG.LR}
    Train-dataset samples: {len(train_ds)}
    Validation-dataset samples: {len(val_ds)} 
    -------------------------
    """

    print(detail)

    start_train = time.time()

    for epoch in range(CFG.EPOCHS):

        start = time.time()

        metrics = run_one_epoch(
            ds_sizes=ds_sizes,
            dataloaders=dataloaders,
            model=model,
            optimizer=optimizer,
            loss=criterion,
        )

        end = time.time() - start

        print(f"Epoch completed in: {end/60} mins")

        table.add_row(
            row=[
                epoch + 1,
                metrics["train_loss"],
                metrics["train_acc"],
                metrics["val_loss"],
                metrics["val_acc"],
            ]
        )
        print(table)

    # Write results to file
    with open("results.txt", "w") as f:
        results = table.get_string()
        f.write(results)

    end_train = time.time() - start_train

    print(f"Training completed in: {end_train/60} mins")
