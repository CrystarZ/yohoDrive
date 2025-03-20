import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter

from trdg.generators import GeneratorFromStrings


def gen_bg(fonts, font_size):
    text = "00"
    image_font = ImageFont.truetype(fonts[0], font_size)
    width, height = image_font.getsize(text)
    return width, height, gaussian_noise_black(height, width)


def gaussian_noise_black(height, width):
    image = np.zeros((height, width), dtype=np.uint8)
    cv2.randn(image, 10, 0)
    # cv2.randn(image, 235, 10)
    return Image.fromarray(image).convert("RGBA")


def gen_dict():
    numbers = []
    for i in range(100):
        if i < 10:
            numbers.append(f"{i:02d}")
            numbers.append(f"{i}")
        else:
            numbers.append(f"{i:02d}")
    return numbers


def gen_img(color, count, fonts, bgdir):
    generator = GeneratorFromStrings(
        strings=gen_dict(),
        count=count,
        fonts=fonts,
        size=font_size,
        text_color=color,
        background_type=3,
        image_dir=bgdir,
        skewing_angle=5,
        random_skew=True,
        blur=0,
        random_blur=True,
        margins=(1, 1, 1, 1),
    )
    return generator


def add_glow_effect(
    image, glow_color=(255, 255, 0), blur_radius=2, intensity=1, contrast_factor=10
):
    # 创建辉光层（将文本复制一层）
    glow_layer = image.convert("RGB")

    # 应用颜色叠加
    # glow_layer = ImageEnhance.Color(glow_layer).enhance(0)  # 转灰度
    glow_layer = ImageEnhance.Brightness(glow_layer).enhance(2)  # 提高亮度
    glow_layer = ImageEnhance.Color(glow_layer).enhance(50)  # 恢复颜色
    glow_layer = ImageEnhance.Brightness(glow_layer).enhance(intensity)  # 增强亮度
    glow_layer = ImageEnhance.Contrast(glow_layer).enhance(contrast_factor)

    # 应用模糊
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    # 叠加辉光层和原始图像
    final_image = Image.blend(image.convert("RGB"), glow_layer, alpha=0.4)

    return final_image


def gen_VOC(id, img, label, root_dir):
    annotations_root_path = f"{root_dir}/Annotations"
    images_root_path = f"{root_dir}/JPEGImages"
    os.makedirs(annotations_root_path, exist_ok=True)
    os.makedirs(images_root_path, exist_ok=True)
    voc_id = f"{id:06}"
    img.save(f"{images_root_path}/{voc_id}.jpg")

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = f"{voc_id}.jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img.width)
    ET.SubElement(size, "height").text = str(img.height)
    ET.SubElement(size, "depth").text = "3"

    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = label
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = "0"
    ET.SubElement(bndbox, "ymin").text = "0"
    ET.SubElement(bndbox, "xmax").text = str(img.width)
    ET.SubElement(bndbox, "ymax").text = str(img.height)

    tree = ET.ElementTree(root)
    tree.write(f"{annotations_root_path}/{voc_id}.xml")


if __name__ == "__main__":
    fonts = ["./.output/fonts/DS-DIGIB-2.ttf", "./.output/fonts/DS-DIGI-1.ttf"]
    font_size = 32
    count = 2000
    mode = True  # OCR_mode or DSIGN_mode
    bg_path = "./.output/images/bg.png"
    out_put = "./.output/dataset"
    w, h, bg = gen_bg(fonts, font_size)
    bg.save(bg_path)

    for r in range(3):
        if r == 0:
            c = "#FF0000"
            la = "red"
        elif r == 1:
            c = "#00FF00"
            la = "green"
        else:
            c = "#FFFF00"
            la = "yellow"

        generator = gen_img(c, count // 3, fonts, "./.output/images")

        for i, (image, label) in enumerate(generator):
            # image = add_glow_effect(image)
            if mode:
                gen_VOC(i + ((count // 3) * r), image, label, out_put)
            else:
                gen_VOC(i + ((count // 3) * r), image, f"cd_{la}", out_put)
