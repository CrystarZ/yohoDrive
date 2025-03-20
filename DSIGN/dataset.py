from typing import Tuple
import numpy as np
import os
import time
import random
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        input_shape: list[int],
        sets: str | None = None,
        set_path_root: bool = True,
        classes: list | None = None,
        train: bool = False,
    ) -> None:
        # input_shape = [w,h] #
        super(VOCDataset, self).__init__()
        if not os.path.exists(root):
            raise FileExistsError(root)
        self.root = root
        self.input_shape = input_shape
        self.sets_root_path = f"{self.root}/ImageSets/Main"
        self.annotations_root_path = f"{self.root}/Annotations"
        self.images_root_path = f"{self.root}/JPEGImages"
        self.reload(sets, set_path_root, classes)

    def reload(
        self,
        sets: str | None = None,
        set_path_root: bool = True,
        classes: list | None = None,
    ) -> None:
        if sets is None:
            xmls = list(f.stem for f in Path(self.annotations_root_path).glob("*.xml"))
        else:
            x_path = f"{self.sets_root_path}/{sets}.txt"
            if not set_path_root:
                x_path = sets
            with open(x_path, encoding="utf-8") as f:
                xmls = f.readlines()
        xmls = [c.strip() for c in xmls]
        self.xmls = xmls
        self.photo_num = len(xmls)
        if classes is None:
            _, self.classes, _ = self.Classes
        else:
            self.classes = classes

    @property
    def totalXml(self) -> list[str]:
        return self.xmls

    @property
    def photoNum(self) -> int:
        return self.photo_num

    def getmeta(self, id: int | str) -> Tuple[str, list[str], list[list[int]]]:
        vocid = id if id is str else str(id).zfill(6)
        metadata = ET.parse(f"{self.annotations_root_path}/{vocid}.xml")
        image = f"{self.images_root_path}/{vocid}.jpg"
        if not os.path.exists(image) or not os.path.isfile(image):
            raise FileExistsError(image)
        labels = []
        boxes = []

        for obj in metadata.getroot().findall("object"):
            name = obj.find("name").text
            xmlbox = obj.find("bndbox")
            b = (
                int(float(xmlbox.find("xmin").text)),
                int(float(xmlbox.find("ymin").text)),
                int(float(xmlbox.find("xmax").text)),
                int(float(xmlbox.find("ymax").text)),
            )
            labels.append(name)
            boxes.append(b)
        return image, labels, boxes

    @property
    def Classes(self) -> Tuple[int, list[str], list[int]]:
        num = 0
        clses = []
        clsesnum = []
        for xml in self.xmls:
            _, labels, _ = self.getmeta(xml)
            for label in labels:
                if label in clses:
                    index = clses.index(label)
                    clsesnum[index] += 1
                else:
                    clses.append(label)
                    clsesnum.append(1)
            num += len(labels)
        return num, clses, clsesnum

    def __len__(self):
        return self.photoNum

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        index = index % self.__len__()
        image, box = self.getdata(
            metadata=self.getmeta(self.xmls[index]),
            input_shape=self.input_shape,
            random=False,
        )
        image_data = np.array(image, dtype="float32")
        image_data /= 255.0  # 归一化
        image_data = np.transpose(image_data, (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        #   对真实框进行预处理
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            #   对真实框进行归一化，调整到0-1之间
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image_data, labels_out

    def genSets(self, ratio=0.7):
        # 需要指定 sets = None
        keys = self.xmls
        random.shuffle(self.keys)
        split_point = int(len(keys) * ratio)  # 计算 7:3 的分界点
        keys_1, keys_2 = keys[:split_point], keys[split_point:]  # 划分键列表

        with open(
            f"{self.sets_root_paths}/train_{time.strftime('%Y%m%d%H%M%S')}.txt", "w"
        ) as f_train:
            for key in keys_1:
                f_train.write(f"{key}\n")

        with open(
            f"{self.sets_root_paths}/val_{time.strftime('%Y%m%d%H%M%S')}.txt", "w"
        ) as f_val:
            for key in keys_2:
                f_val.write(f"{key}\n")

    def getdata(
        self,
        metadata: Tuple[str, list[str], list[list[int]]],
        input_shape: list[int],
        random: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_path = metadata[0]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        labels = [self.classes.index(s) for s in metadata[1]]
        box = np.hstack((np.array(metadata[2]), np.array(labels).reshape(-1, 1)))

        iw, ih = image.size
        w, h = input_shape
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        if True:  # 是否填充
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.Resampling.BICUBIC)
            image_data = Image.new("RGB", self.input_shape, (128, 128, 128))
            image_data.paste(image, (dx, dy))
            image_data = np.array(image_data, np.float32)
        else:
            image_data = image.resize(self.input_shape, Image.Resampling.BICUBIC)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        return image_data, box
