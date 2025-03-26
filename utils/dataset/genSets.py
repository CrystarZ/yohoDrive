import os

from PIL import Image
from torchvision.transforms import transforms
from VOCdataset import (
    VOCDataset,
    VOCDatasetPer,
    check_bboxes,
    genXMl,
    getmeta,
)

trans_to_pil = transforms.ToPILImage(mode="RGB")

labels = []
d = {}


# 冻结当前加载的数据为数据集
def Freeze(dataset_dir: str, output: str):
    s = VOCDataset(dataset_dir, input_shape=None, classes=labels, render=d)
    s.freeze_dataset(output)


# 冻结当前加载的数据为数据集
def freeze(dataset_dir: str, output: str):
    s = VOCDatasetPer(dataset_dir, input_shape=None, classes=labels, render=d)
    s.freeze_dataset(output)


# 划分训练集测试集
def genSets(dataset_dir: str, ratio=0.7):
    s = VOCDataset(dataset_dir)
    s.genSets(ratio)


# 合并两个数据集
def fusionDatasets(dataset1_dir: str, dataset2_dir: str, output: str):
    s1 = VOCDataset(dataset1_dir)
    s2 = VOCDataset(dataset2_dir)

    annotations_root_path = f"{output}/Annotations"
    images_root_path = f"{output}/JPEGImages"

    os.makedirs(output, exist_ok=True)
    os.makedirs(annotations_root_path, exist_ok=True)
    os.makedirs(images_root_path, exist_ok=True)

    def gen(s: VOCDataset, start: int):
        for i in range(len(s)):
            id = str(i + start).zfill(6)
            img_path, labels, boxes = s.getmeta(s.xmls[i])
            img = Image.open(img_path)
            genXMl(img, labels, boxes, id, annotations_root_path, images_root_path)

    gen(s1, 0)
    gen(s2, len(s1))


if __name__ == "__main__":
    dataset_dir = "../../tlVOC"
    annotations_root_path = f"{dataset_dir}/Annotations"
    images_root_path = f"{dataset_dir}/JPEGImages"
    genSets("dataset_dir")
    for i in range(1000):
        id = str(i).zfill(6)
        meta = getmeta(f"{annotations_root_path}/{id}.xml", images_root_path)
        check_bboxes(meta)
        input()
