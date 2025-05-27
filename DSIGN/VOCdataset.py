####################VOC数据集工具

from typing import Tuple
import numpy as np
import os
import bisect
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms

xml = str
xmls = list[xml]
box = list[int]
cls = str
classes = list[cls]
meta = Tuple[str, list[str], list[box]]  # img_path,labels,boxes
shape = list[int]  # w,h
COLOR = (0, 0, 255)


########################sets


def scan_annotations(path: str) -> xmls:
    xmls = list(f.stem for f in Path(path).glob("*.xml"))
    return xmls


def loadSets(path: str) -> xmls:
    with open(path, encoding="utf-8") as f:
        xmls = f.readlines()
    xmls = [c.strip() for c in xmls]
    return xmls


def split_sets(xmls: xmls, ratio=0.7) -> Tuple[xmls, xmls]:
    keys = xmls
    random.shuffle(keys)
    split_point = int(len(keys) * ratio)  # 计算 7:3 的分界点
    keys_1, keys_2 = keys[:split_point], keys[split_point:]  # 划分键列表

    return keys_1, keys_2


def write_sets(xmls: xmls, path: str, random: bool = True) -> None:
    if random:
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        path = f"{dirname}/{name}_{time.strftime('%y%m%d%h%m%s')}{ext}"

    with open(path, "w") as f:
        for k in xmls:
            f.write(f"{k}\n")


########################meta


def getmeta(xml_file: xml, images_root_path: str) -> meta:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_filename = root.find("filename").text
    img_path = f"{images_root_path}/{img_filename}"

    image = img_path
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        box = [xmin, ymin, xmax, ymax]
        boxes.append(box)
        labels.append(name)

    return image, labels, boxes


def render_meta(meta: meta, rule: dict) -> meta:
    t_meta = []
    ls = []
    bs = []
    oi, ols, obs = meta
    for i, ol in enumerate(ols):
        if ol in rule:
            ls.append(rule[ol])
            bs.append(obs[i])
    t_meta = (oi, ls, bs)
    return t_meta


########################pic


def check_bboxes(meta: meta) -> None:
    img_path, labels, boxes = meta
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, label in enumerate(labels):
        box = boxes[i]
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=COLOR, width=3)
        text_size = draw.textbbox((0, 0), label, font=None)
        text_w = text_size[2] - text_size[0]
        text_h = text_size[3] - text_size[1]
        text_bg = (xmin, ymin - text_h, xmin + text_w, ymin)  # 计算文本框位置

        draw.rectangle(text_bg, fill=COLOR)
        draw.text((xmin, ymin - text_h), label, fill=(255, 255, 255))

    img.show()


def resize(image: Image.Image, shape: shape, fill: bool = True) -> Image.Image:
    iw, ih = image.size
    w, h = shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    if fill:
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        newImage = Image.new(image.mode, (w, h), tuple([128] * len(image.mode)))
        newImage.paste(image, (dx, dy))
        image = newImage
    else:
        image = image.resize((w, h), Image.Resampling.BICUBIC)
    return image


def genXMl(
    img: Image.Image,
    labels: list[str],
    boxes: list[box],
    id: str,
    annotations_root_path: str,
    images_root_path: str,
):
    img_w, img_h = img.size
    img_ch = len(img.mode)
    img.save(f"{images_root_path}/{id}.jpg")
    with open(f"{annotations_root_path}/{id}.xml", "w") as xml_files:
        xml_files.write("<annotation>\n")
        xml_files.write("   <folder>folder</folder>\n")
        xml_files.write(f"   <filename>{id}.jpg</filename>\n")
        xml_files.write("   <source>\n")
        xml_files.write("   <database>Unknown</database>\n")
        xml_files.write("   </source>\n")
        xml_files.write("   <size>\n")
        xml_files.write(f"     <width>{img_w}</width>\n")
        xml_files.write(f"     <height>{img_h}</height>\n")
        xml_files.write(f"     <depth>{img_ch}</depth>\n")
        xml_files.write("   </size>\n")
        xml_files.write("   <segmented>0</segmented>\n")
        for i, l in enumerate(labels):
            xml_files.write("   <object>\n")
            xml_files.write(f"      <name>{l}</name>\n")
            xml_files.write("      <pose>Unspecified</pose>\n")
            xml_files.write("      <truncated>0</truncated>\n")
            xml_files.write("      <difficult>0</difficult>\n")
            xml_files.write("      <bndbox>\n")
            xmin, ymin, xmax, ymax = boxes[i]
            xml_files.write(f"         <xmin>{xmin}</xmin>\n")
            xml_files.write(f"         <ymin>{ymin}</ymin>\n")
            xml_files.write(f"         <xmax>{xmax}</xmax>\n")
            xml_files.write(f"         <ymax>{ymax}</ymax>\n")
            xml_files.write("      </bndbox>\n")
            xml_files.write("   </object>\n")
        xml_files.write("</annotation>")


def decodeLabel(label: np.ndarray, shape: shape) -> Tuple[int, box]:
    img_w, img_h = shape
    c = int(label[1])
    center_x = round(float(label[2]) * img_w)
    center_y = round(float(label[3]) * img_h)
    bbox_w = round(float(label[4]) * img_w)
    bbox_h = round(float(label[5]) * img_h)
    xmin = int(center_x - bbox_w / 2)
    ymin = int(center_y - bbox_h / 2)
    xmax = int(center_x + bbox_w / 2)
    ymax = int(center_y + bbox_h / 2)
    return c, [xmin, ymin, xmax, ymax]


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        input_shape: shape | None = None,
        sets: str | None = None,
        set_path_root: bool = True,
        classes: list | None = None,
        train: bool = False,
        transform: transforms.Compose | None = None,
        render: dict | None = None,
    ) -> None:
        super(VOCDataset, self).__init__()
        if not os.path.exists(root):
            raise FileExistsError(root)
        self.root = root
        self.input_shape = input_shape

        self.sets_root_path = f"{self.root}/ImageSets/Main"
        self.annotations_root_path = f"{self.root}/Annotations"
        self.images_root_path = f"{self.root}/JPEGImages"

        self.render = render
        self.reload(sets, set_path_root, classes)

        def ensure_rgb(image: Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")  # 将灰度图转换为 RGB
            return image

        RGBtransform = transforms.Compose([transforms.Lambda(lambda x: ensure_rgb(x))])
        if transform is None:
            self.transform = RGBtransform
        else:
            self.transform = transform

    def reload(
        self,
        sets: str | None = None,
        set_path_root: bool = True,
        classes: classes | None = None,
    ) -> None:
        if sets is None:
            xmls = scan_annotations(self.annotations_root_path)
        else:
            x_path = f"{self.sets_root_path}/{sets}.txt"
            if not set_path_root:
                x_path = sets
            xmls = loadSets(x_path)
        self.xmls = xmls
        if classes is None or self.render is not None:
            self.classes = None
            _, clses, _ = self.Classes
            self.classes = clses
        else:
            self.classes = classes

    def getmeta(self, id: int | str) -> meta:
        if isinstance(id, int):
            id = str(id).zfill(6)
        m = getmeta(f"{self.annotations_root_path}/{id}.xml", self.images_root_path)
        if self.render is not None:
            m = render_meta(m, self.render)
        return m

    def genSets(self, ratio=0.7):
        k1, k2 = split_sets(self.xmls, ratio)
        os.makedirs(self.sets_root_path, exist_ok=True)
        write_sets(k1, f"{self.sets_root_path}/train.txt")
        write_sets(k2, f"{self.sets_root_path}/val.txt")

    @property
    def totalXml(self) -> list[str]:
        return self.xmls

    @property
    def photoNum(self) -> int:
        return len(self.xmls)

    @property
    def Classes(self) -> Tuple[int, classes, list[int]]:
        num = 0
        clses = [] if self.classes is None else self.classes
        if self.render is not None:
            clses = list(self.render.values())
        clsesnum = [0] * len(clses)
        scan_flag = len(clses) == 0
        for xml in self.xmls:
            _, labels, _ = self.getmeta(xml)
            if len(labels) == 0:
                self.xmls.remove(xml)
                continue
            for label in labels:
                if label in clses:
                    index = clses.index(label)
                    clsesnum[index] += 1
                elif scan_flag:
                    clses.append(label)
                    clsesnum.append(1)
            num += len(labels)
        return num, clses, clsesnum

    def __len__(self):
        return self.photoNum

    def __getitem__(self, index) -> Tuple[Tensor, NDArray[np.float64]]:
        index = index % self.__len__()
        image, box = self.getdata(
            metadata=self.getmeta(self.xmls[index]),
            input_shape=self.input_shape,
            random=False,
        )
        transform = transforms.ToTensor()
        image_data = self.transform(image)
        image_data = transform(image_data)

        box = np.array(box, dtype=np.float32)

        shape = self.input_shape
        if shape is None:
            shape = image.size

        #   对真实框进行预处理
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            #   对真实框进行归一化，调整到0-1之间
            box[:, [0, 2]] = box[:, [0, 2]] / shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / shape[0]
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

    def getdata(
        self,
        metadata: meta,
        input_shape: shape | None,
        random: bool = False,
        fill_with_resize=True,
    ) -> Tuple[Image.Image, np.ndarray]:
        image_path = metadata[0]
        image = Image.open(image_path)
        labels = [self.classes.index(s) for s in metadata[1]]
        box = np.hstack((np.array(metadata[2]), np.array(labels).reshape(-1, 1)))

        if input_shape is None:
            return image, box
        return self.adjSize(input_shape, image, box)

    def adjSize(self, input_shape: shape, image: Image.Image, box: np.ndarray):
        iw, ih = image.size
        w, h = input_shape

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = resize(image, input_shape, fill=True)

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
        return image, box

    def freeze_dataset(self, path: str):
        for i in range(len(self)):
            id = str(i).zfill(6)
            img_path, labels, boxes = self.getmeta(self.xmls[i])
            img = Image.open(img_path)
            annotations_root_path = f"{path}/Annotations"
            images_root_path = f"{path}/JPEGImages"
            os.makedirs(path, exist_ok=True)
            os.makedirs(annotations_root_path, exist_ok=True)
            os.makedirs(images_root_path, exist_ok=True)

            genXMl(img, labels, boxes, id, annotations_root_path, images_root_path)


class VOCDatasetPer(VOCDataset):
    @property
    def Metas(self) -> list[meta]:
        xmls = self.xmls
        metas = [self.getmeta(xml) for xml in xmls]
        return metas

    def reload(
        self,
        sets: str | None = None,
        set_path_root: bool = True,
        classes: classes | None = None,
    ) -> None:
        super().reload(sets, set_path_root, classes)
        self.metas = self.Metas
        labels = [i[1] for i in self.metas]
        n_xmls = [len(i) for i in labels]  # 每个xml所含label的个数
        p_xmls = [0]  # 前缀和
        for num in n_xmls:
            p_xmls.append(p_xmls[-1] + num)
        self._n_xmls = n_xmls
        self._p_xmls = p_xmls
        self.obj_num = sum(n_xmls)

    def find_index(self, i: int) -> Tuple[int, int]:
        # il = next(idx for idx, val in enumerate(self._p_xmls) if val > 1) - 1
        il = bisect.bisect_right(self._p_xmls, i) - 1
        ie = i - self._p_xmls[il]
        return il, ie

    def __len__(self):
        return self.obj_num

    def __getitem__(self, index) -> Tuple[Tensor, str]:
        il, ie = self.find_index(index)
        image, box = self.getdata(
            metadata=self.metas[il],
            input_shape=None,
            random=False,
        )
        xmin, ymin, xmax, ymax, label = box[ie]
        image = image.crop((xmin, ymin, xmax, ymax))
        if self.input_shape is not None:
            image, _ = self.adjSize(input_shape=self.input_shape, image=image, box=[])
        label = self.classes[label]

        transform = transforms.ToTensor()
        image_data = self.transform(image)
        image_data = transform(image_data)

        return image_data, label

    def freeze_dataset(self, path: str):
        trans_to_pil = transforms.ToPILImage()
        for i in range(len(self)):
            img, label = self[i]
            img = trans_to_pil(img)
            img_w, img_h = img.size
            annotations_root_path = f"{path}/Annotations"
            images_root_path = f"{path}/JPEGImages"
            os.makedirs(path, exist_ok=True)
            os.makedirs(annotations_root_path, exist_ok=True)
            os.makedirs(images_root_path, exist_ok=True)

            genXMl(
                img=img,
                labels=[label],
                boxes=[[0, 0, img_w, img_h]],
                id=str(i).zfill(6),
                annotations_root_path=annotations_root_path,
                images_root_path=images_root_path,
            )
