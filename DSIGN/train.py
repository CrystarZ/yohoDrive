import os
import sys
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.nn.modules import Module
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.utils.data as Data
from torch.utils.data import DataLoader
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


def fit_step(model: Module, optimizer: Optimizer, lossfunc, x):
    images, bboxes = x

    outputs = model(images)
    optimizer.zero_grad()
    loss_value = lossfunc(outputs, bboxes)
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
    model = model.train()
    model = model.to(device)
    for step, batch in enumerate(traindata):
        images, bboxes = batch
        images = images.to(device)
        bboxes = bboxes.to(device)

        loss_value = fit_step(model, optimizer, lossfunc, (images, bboxes))
        loss += loss_value.item()
        print(f"train {step}/{len(traindata)}| loss: {loss}, lr: {get_lr(optimizer)}")

    for step, batch in enumerate(valdata):
        images, bboxes = batch
        images = images.to(device)
        bboxes = bboxes.to(device)

        loss_value = fit_step(model, optimizer, lossfunc, (images, bboxes))
        valoss += loss_value.item()
        print(f"val {step}/{len(valdata)}| valoss: {loss}, lr: {get_lr(optimizer)}")

    return loss, valoss


if __name__ == "__main__":
    LR = 1e-3
    EPOCH = 100
    BATCH_SIZE = 25
    INPUT_SHAPE = [640, 640]

    trainDataset = VOCDataset("./datasets/traffic_light_VOC", INPUT_SHAPE, sets="train")
    valDataset = VOCDataset("./datasets/traffic_light_VOC", INPUT_SHAPE, sets="val")
    classes = trainDataset.classes
    model = YoloBody(INPUT_SHAPE, len(classes), "l")
    optimizer = optim.Adam(model.parameters(), LR, betas=(0.937, 0.999))
    lossfunc = Loss(model)

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
            save_state_dict = model.state_dict()
            print("Save best model to best_epoch_weights.pth")
            torch.save(save_state_dict, "./.output/best_epoch_weights.pth")
