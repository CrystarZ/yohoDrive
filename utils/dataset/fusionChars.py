#############融合两个字符成为两位数

import os
import random
from typing import Tuple

from PIL import Image
from VOCdataset import VOCDataset, genXMl
from torchvision.transforms import transforms

trans_to_pil = transforms.ToPILImage(mode="RGB")

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
s = VOCDataset("./.output/in", input_shape=None, classes=classes)
total_num = len(s)


def gen2char() -> Tuple[Image.Image, str]:
    img1, l1 = s[random.randint(0, total_num - 1)]
    img2, l2 = s[random.randint(0, total_num - 1)]
    img1: Image.Image = trans_to_pil(img1).convert("L")
    img2: Image.Image = trans_to_pil(img2).convert("L")

    height = max(img1.height, img2.height)
    w1 = int(img1.width * (height / img1.height))
    w2 = int(img2.width * (height / img2.height))
    img1 = img1.resize((w1, height))
    img2 = img2.resize((w2, height))
    width = w1 + w2
    img = Image.new("L", (width, height), 255)
    img.paste(img1, (0, 0))
    img.paste(img2, (w1, 0))
    l1 = classes[int(l1[0][1])]
    l2 = classes[int(l2[0][1])]
    label = l1 + l2
    return img, label


if __name__ == "__main__":
    output_dir = "./.output/out"
    annotations_root_path = f"{output_dir}/Annotations"
    images_root_path = f"{output_dir}/JPEGImages"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(annotations_root_path, exist_ok=True)
    os.makedirs(images_root_path, exist_ok=True)

    num_2char_image = 1

    for i in range(num_2char_image):
        id = str(i).zfill(6)
        img, label = gen2char()
        genXMl(
            img,
            [label],
            [[0, 0, img.width, img.height]],
            id,
            annotations_root_path,
            images_root_path,
        )
