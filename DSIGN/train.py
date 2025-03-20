import os
import datetime
import random
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (
    Loss,
    ModelEMA,
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
)
from utils.callbacks import LossHistory
from utils.utils_fit import fit_one_epoch
from dataset import VOCDataset


def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# dataloader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.floattensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.floattensor)
    return images, bboxes


#
if __name__ == "__main__":
    Cuda = False
    seed = 11
    fp16 = False  # 是否使用混合精度训练
    classes_path = "model_data/voc_classes.txt"
    model_path = "model_data/yolov8_s.pth"
    input_shape = [640, 640]
    phi = "s"
    pretrained = False
    # ------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认50%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
    # ------------------------------------------------------------------#
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    label_smoothing = 0  # 标签平滑。一般0.01以下。如0.01、0.005。

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 32
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
    #                           Adam可以使用相对较小的UnFreeze_Epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = 16
    Freeze_Train = True  # 是否进行冻结训练

    Init_lr = 1e-2  # 模型的最大学习率
    Min_lr = Init_lr * 0.01  # 模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = "cos"  #  使用到的学习率下降方式，可选的有step、cos
    save_period = 10  #  多少个epoch保存一次权值
    save_dir = "logs"
    eval_flag = True  # 是否在训练时进行评估，评估对象为验证集
    eval_period = 10  # 代表多少个epoch评估一次，不建议频繁的评估

    num_workers = 4  # 用于设置是否使用多线程读取数据
    #####################################################################
    seed_everything(seed)

    #   设置用到的显卡
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    rank = 0

    #   获取classes和anchor
    class_names, num_classes = get_classes(classes_path)

    #   创建yolo模型
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != "":
        if local_rank == 0:
            print("Load weights {}.".format(model_path))

        #   根据预训练权重的Key和模型的Key进行加载
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        #   显示没有匹配上的Key
        if local_rank == 0:
            print(
                "\nSuccessful Load Key:",
                str(load_key)[:500],
                "……\nSuccessful Load Key Num:",
                len(load_key),
            )
            print(
                "\nFail To Load Key:",
                str(no_load_key)[:500],
                "……\nFail To Load Key num:",
                len(no_load_key),
            )
            print(
                "\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m"
            )

    #   获得损失函数
    yolo_loss = Loss(model)
    #   记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
        )
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #   权值平滑
    ema = ModelEMA(model_train)

    # 数据集
    train_dataset = VOCDataset(
        "./VOCdevkit/VOC2007",
        input_shape=input_shape,
        classes=class_names,
        sets="train",
    )
    val_dataset = VOCDataset(
        "./VOCdevkit/VOC2007", input_shape=input_shape, classes=class_names, sets="val"
    )
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    if local_rank == 0:
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"
                % (optimizer_type, wanted_step)
            )
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"
                % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step)
            )
            print(
                "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"
                % (total_step, wanted_step, wanted_epoch)
            )

    if True:
        UnFreeze_flag = False
        #   冻结一定部分训练
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #   判断当前batch_size，自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == "adam" else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(
            max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
        )

        #   根据optimizer_type选择优化器
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            "adam": optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            "sgd": optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True),
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(
            lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
        )

        #   判断每一个世代的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        train_sampler = None
        val_sampler = None
        shuffle = True

        gen = DataLoader(
            train_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=yolo_dataset_collate,
            sampler=train_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        )
        gen_val = DataLoader(
            val_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=yolo_dataset_collate,
            sampler=val_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        )

        eval_callback = None

        #   开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #   如果模型有冻结学习部分则解冻，并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #   判断当前batch_size，自适应调整学习率
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == "adam" else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
                Init_lr_fit = min(
                    max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max
                )
                Min_lr_fit = min(
                    max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2),
                    lr_limit_max * 1e-2,
                )

                #   获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(
                    lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
                )

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if ema:
                    ema.updates = epoch_step * epoch

                gen = DataLoader(
                    train_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=yolo_dataset_collate,
                    sampler=train_sampler,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                )
                gen_val = DataLoader(
                    val_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=yolo_dataset_collate,
                    sampler=val_sampler,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                )

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(
                model_train,
                model,
                ema,
                yolo_loss,
                loss_history,
                eval_callback,
                optimizer,
                epoch,
                epoch_step,
                epoch_step_val,
                gen,
                gen_val,
                UnFreeze_Epoch,
                Cuda,
                fp16,
                scaler,
                save_period,
                save_dir,
                local_rank,
            )

        if local_rank == 0:
            loss_history.writer.close()
