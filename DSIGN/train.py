import os
import sys
from tqdm import tqdm
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

from DSIGN.nets.yolo import YoloBody
from DSIGN.nets.yolo_training import Loss
from DSIGN.VOCdataset import VOCDataset


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes


def fit_step(
    model: Module,
    optimizer: Optimizer,
    lossfunc,
    x: Tensor,  # images
    y: Tensor,  # bboxes
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


def train():
    LR = 1e-3
    EPOCH = 300
    BATCH_SIZE = 16
    INPUT_SHAPE = [640, 640]
    DATASET_PATH = "./datasets/traffic_light_VOC"
    SAVE_PATH = "./.output"

    trainDataset = VOCDataset(DATASET_PATH, INPUT_SHAPE, sets="train")
    valDataset = VOCDataset(DATASET_PATH, INPUT_SHAPE, sets="val")
    classes = trainDataset.classes + valDataset.classes
    trainDataset.reload(classes=classes)
    valDataset.reload(classes=classes)
    trainLoader = Data.DataLoader(
        dataset=trainDataset,
        batch_size=BATCH_SIZE,
        collate_fn=yolo_dataset_collate,
    )
    valLoader = Data.DataLoader(
        dataset=valDataset,
        batch_size=BATCH_SIZE,
        collate_fn=yolo_dataset_collate,
    )

    model = YoloBody(INPUT_SHAPE, len(classes), "l")
    optimizer = optim.Adam(model.parameters(), LR, betas=(0.937, 0.999))
    lossfunc = Loss(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_history = []
    valoss_history = []
    for epoch in range(EPOCH):
        loss, valoss = fit_epoch(
            model, optimizer, lossfunc, trainLoader, valLoader, device
        )

        loss = loss / len(trainLoader)
        valoss = valoss / len(valLoader)
        print(f"EPOCH: {epoch} | loss: {loss},valoss: {valoss}")

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


if __name__ == "__main__":
    train()
