import sys
import os
from typing import Tuple
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

from OCR.nets.LeNet5 import OCRNUM


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_step(
    model: Module,
    optimizer: Optimizer,
    lossfunc,
    x: Tensor,  # images
    y: Tensor,  # labels
    train: bool = True,
):
    outputs = model(x)
    loss_value = lossfunc(outputs, y)
    if train:
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    return loss_value


def fit_epoch(
    model: Module,
    optimizer: Optimizer,
    lossfunc,
    traindata: DataLoader,
    valdata: DataLoader,
    device,
) -> Tuple[float, float]:
    loss = 0
    valoss = 0
    model = model.to(device)

    model = model.train()
    loop = tqdm(enumerate(traindata), total=len(traindata), desc="Train", ncols=100)
    for step, (x, y) in loop:
        x = x.to(device)
        y = y.to(device)

        loss_value = fit_step(model, optimizer, lossfunc, x, y)
        loss += loss_value.item()

        loop.set_postfix(
            {"loss": f"{loss / (step + 1):.4f}", "lr": f"{get_lr(optimizer):.6f}"}
        )

    model = model.eval()
    loop = tqdm(enumerate(valdata), total=len(valdata), desc="Eval ", ncols=100)
    with torch.no_grad():
        for step, (x, y) in loop:
            x = x.to(device)
            y = y.to(device)

            loss_value = fit_step(model, optimizer, lossfunc, x, y, False)
            valoss += loss_value.item()

            loop.set_postfix(
                {"loss": f"{valoss / (step + 1):.4f}", "lr": f"{get_lr(optimizer):.6f}"}
            )

    return loss, valoss


def download():
    DATASET_PATH = "./datasets"
    MNIST(root=DATASET_PATH, download=True)


def train():
    LR = 1e-3
    EPOCH = 100
    BATCH_SIZE = 50
    DATASET_PATH = "./datasets"
    SAVE_PATH = "./.output"

    transform = transforms.ToTensor()
    trainDataset = MNIST(root=DATASET_PATH, transform=transform, train=True)
    valDataset = MNIST(root=DATASET_PATH, transform=transform, train=False)
    valDataset = Subset(valDataset, indices=range(1000))
    trainLoader = Data.DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE)
    valLoader = Data.DataLoader(dataset=valDataset, batch_size=2000, shuffle=False)

    model = OCRNUM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lossfunc = nn.CrossEntropyLoss()  # 对应one-hot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_history = []
    valoss_history = []
    for epoch in range(EPOCH):
        loss, valoss = fit_epoch(
            model, optimizer, lossfunc, trainLoader, valLoader, device
        )

        loss = loss / len(trainLoader)
        valoss = valoss / len(valLoader)
        x, y = next(iter(valLoader))
        pred_y = torch.max(model(x), 1)[1].data.numpy()
        accuracy = (pred_y == y.data.numpy()).astype(int).sum() / y.size(0)
        print(f"EPOCH: {epoch} | loss: {loss},valoss: {valoss}, accuracy: {accuracy}")

        loss_history.append(loss)
        valoss_history.append(valoss)
        if len(valoss_history) <= 1 or valoss <= min(valoss_history):
            print("Save best model to best_weights.pth")
            save_state_dict = model.state_dict()
            torch.save(save_state_dict, f"{SAVE_PATH}/best_weights.pth")

            with open(f"{SAVE_PATH}/loss.txt", "w", encoding="utf-8") as f:
                for i in loss_history:
                    f.write(str(i) + "\n")
            with open(f"{SAVE_PATH}/val_loss.txt", "w", encoding="utf-8") as f:
                for i in valoss_history:
                    f.write(str(i) + "\n")


def test():
    TESTNUM = 32
    DATASET_PATH = "./datasets"
    SAVE_PATH = "./.output"

    transform = transforms.ToTensor()
    testDataset = MNIST(root=DATASET_PATH, transform=transform)
    trainLoader = Data.DataLoader(dataset=testDataset, batch_size=TESTNUM)

    model = OCRNUM()
    model.load_state_dict(torch.load(f"{SAVE_PATH}/best_weights.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = next(iter(trainLoader))
    x = x.to(device)
    y = y.to(device)

    pred_y = torch.max(model(x), 1)[1].data.numpy()
    accuracy = (pred_y == y.data.numpy()).astype(int).sum() / y.size(0)

    print("accuracy          :", accuracy)
    print("prediction number :", pred_y)
    print("real number       :", y.numpy())

    img = torchvision.utils.make_grid(x)
    img = img.numpy().transpose(1, 2, 0)

    cv2.imshow("win", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    test()
