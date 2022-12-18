import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data import train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
from model import ResNet18, HNet
import config as CFG
from tqdm import tqdm
from prettytable import PrettyTable
from argparse import ArgumentParser

def run_one_epoch(ds_sizes, dataloaders, model, optimizer, loss):

    metrics = {}

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        avg_loss = 0
        running_corrects = 0

        for (images, labels) in tqdm(dataloaders[phase]):

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

        epoch_loss = avg_loss / ds_sizes[phase]
        epoch_acc = running_corrects.double() / ds_sizes[phase]

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

    args = parser.parse_args()
    CFG.EPOCHS = args.epochs

    # table
    table = PrettyTable(
        field_names=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
    )

    # the model
    model = HNet()
    model.to(CFG.DEVICE)

    # Setting up optimizer and loss
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()

    dataloaders = {"train": train_dl, "val": val_dl}
    ds_sizes = {"train": len(train_ds), "val": len(val_ds)}

    detail = f"""
    Training details: 
    ------------------------    
        Model: HNet()
        Epochs: {CFG.EPOCHS}
        Optimizer: {type(optimizer).__name__}
        Loss: {criterion._get_name()}
        Train-dataset samples: {len(train_ds)}
        Validation-dataset samples: {len(val_ds)} 
    -------------------------
    """

    print(detail)

    for epoch in range(CFG.EPOCHS):
        metrics = run_one_epoch(
            ds_sizes=ds_sizes,
            dataloaders=dataloaders,
            model=model,
            optimizer=optimizer,
            loss=criterion,
        )

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
