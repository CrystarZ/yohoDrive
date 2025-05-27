import os
import sys
import time
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
from torchvision.ops import box_iou
import numpy as np

from DSIGN.nets.yolo import YoloBody
from DSIGN.nets.yolo_training import Loss, xywh2xyxy
from DSIGN.VOCdataset import VOCDataset
from DSIGN.utils.utils_bbox import DecodeBox

BATCH_SIZE = 16
INPUT_SHAPE = [640, 640]
PHI = "s"
BACKBONE = "MNV3"
DATASET_PATH = "./datasets/traffic_light_VOC"
CLASSES_PATH = "./weights/tl_classes.txt"
SAVE_PATH = "./.output"
WEIGHT_PATH = f"{SAVE_PATH}/best_weights.pth"


def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


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

    classes, classes_len = get_classes(CLASSES_PATH)
    trainDataset = VOCDataset(DATASET_PATH, INPUT_SHAPE, sets="train", classes=classes)
    valDataset = VOCDataset(DATASET_PATH, INPUT_SHAPE, sets="val", classes=classes)
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

    model = YoloBody(num_classes=classes_len, phi=PHI, bb=BACKBONE)
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


def compute_precision(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0.0
    if len(gt_boxes) == 0:
        return 0.0

    ious = box_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]
    max_iou, max_indices = ious.max(dim=1)  # 每个预测框匹配一个 gt 框

    tp = 0
    for i in range(len(pred_boxes)):
        if max_iou[i] >= iou_threshold:
            pred_label = pred_labels[i]
            matched_gt_label = gt_labels[max_indices[i]]
            if pred_label == matched_gt_label:
                tp += 1

    fp = len(pred_boxes) - tp
    precision = tp / (tp + fp + 1e-6)
    return precision


def test():
    classes, num_classes = get_classes(CLASSES_PATH)

    testDataset = VOCDataset(DATASET_PATH, INPUT_SHAPE, sets="val", classes=classes)
    testLoader = Data.DataLoader(
        dataset=testDataset,
        batch_size=BATCH_SIZE,
        collate_fn=yolo_dataset_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloBody(num_classes=num_classes, phi=PHI, bb=BACKBONE)
    model = model.to(device).eval()
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))

    total_precision = 0.0
    num_samples = 0

    dbbox = DecodeBox(num_classes, INPUT_SHAPE)

    with torch.no_grad():
        for x, y in tqdm(testLoader, desc="Testing"):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            outputs = dbbox.decode_box(outputs)
            outputs = dbbox.non_max_suppression(
                outputs,
                num_classes,
                INPUT_SHAPE,
                INPUT_SHAPE,
                True,
            )

            targets = []
            for i in range(BATCH_SIZE):
                img_b = y[y[:, 0] == i]
                targets.append(img_b)

            for output, target in zip(outputs, targets):
                if output is None:
                    break

                pred_boxes = output[:, :4]
                pred_boxes = pred_boxes[:, [1, 0, 3, 2]]
                pred_boxes = torch.from_numpy(pred_boxes).float().to(device)
                gt_boxes = target[:, -4:]
                gt_boxes = xywh2xyxy(gt_boxes)
                gt_boxes[0::2] *= INPUT_SHAPE[1]  # x1, x2
                gt_boxes[1::2] *= INPUT_SHAPE[0]  # y1, y2

                pred_label = output[:, 5]
                gt_label = target[:, 1]

                precision = compute_precision(
                    pred_boxes, pred_label, gt_boxes, gt_label
                )
                total_precision += precision
                num_samples += 1

    avg_precision = total_precision / (num_samples + 1e-6)
    print(f"Precision: {avg_precision:.4f}")
    return avg_precision


def inference_time():
    R = 320
    classes, num_classes = get_classes(CLASSES_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloBody(num_classes=num_classes, phi=PHI, bb=BACKBONE)
    model = model.to(device).eval()
    input = torch.randn(1, 3, INPUT_SHAPE[0], INPUT_SHAPE[1]).to(device)

    for _ in range(10):  # 预热
        _ = model(input)

    start = time.time()
    for _ in range(R):
        _ = model(input)
    end = time.time()
    avg_time = (end - start) / 100

    print(f"Inference time: {avg_time * 1000:.2f} ms")


def throughput():
    R = 10
    classes, num_classes = get_classes(CLASSES_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloBody(num_classes=num_classes, phi=PHI, bb=BACKBONE)
    model = model.to(device).eval()
    input = torch.randn(BATCH_SIZE, 3, INPUT_SHAPE[0], INPUT_SHAPE[1]).to(device)

    start = time.time()
    for _ in range(R):
        _ = model(input)
    end = time.time()

    throughput = BATCH_SIZE * R / (end - start)
    print(f"Throughput: {throughput:.2f} images/sec")


if __name__ == "__main__":
    test()
