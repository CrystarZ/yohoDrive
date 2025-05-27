from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from PIL import Image, ImageDraw
from .utils.utils_bbox import DecodeBox
from .nets.yolo import YoloBody

imgshape = Tuple[int, int]  # w,h


class YOLO(object):
    def __init__(
        self,
        module_path: str,  # 权重路径
        classes: list[str],  # 标签
        input_shape: imgshape = (640, 640),  # 输入图像尺寸(32的倍数)
        phi: str = "s",  # 对应yolov8版本
        bb: str = "CSPDarknet",
        confidence: float = 0.5,  # 置信度
        nms_iou: float = 0.3,  # 非极大抑制所用到的nms_iou大小
        letterbox_image: bool = True,  # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    ) -> None:
        self.module_path = module_path
        self.input_shape = input_shape
        self.phi = phi
        self.bb = bb
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names, self.num_classes = classes, len(classes)
        self.bbox_util = DecodeBox(
            self.num_classes, (self.input_shape[0], self.input_shape[1])
        )
        self.net = self.generate()

    def generate(self):
        # 建立yolo模型
        net = YoloBody(self.num_classes, self.phi, self.bb)
        net.load_state_dict(torch.load(self.module_path, map_location=self.device))
        net = net.to(self.device)
        net = net.fuse().eval()
        print("{} model, and classes loaded.".format(self.module_path))

        #     if self.device.type == "cuda" and torch.cuda.device_count() > 1:
        #         net = nn.DataParallel(net)

        return net

    def preprocessing_image(self, image: Image.Image) -> Tuple[Tensor, imgshape]:
        #  图像预处理
        image_shape = image.size  # 获取图像尺寸
        if len(image.mode) != 3:  # 强制转换成RGB图像，防止灰度图报错
            image = image.convert("RGB")

        # RESIZE
        if self.letterbox_image:  # 是否填充
            iw, ih = image_shape
            w, h = self.input_shape

            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.Resampling.BICUBIC)
            image_data = Image.new("RGB", self.input_shape, (128, 128, 128))
            image_data.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            image_data = image.resize(self.input_shape, Image.Resampling.BICUBIC)

        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        transform = transforms.ToTensor()
        image_data = transform(image_data)
        image_data = image_data.unsqueeze(0)
        # image_data = np.array(image_data, dtype="float32")
        # image_data /= 255.0  # 归一化
        # image_data = np.transpose(image_data, (2, 0, 1))
        # image_data = np.expand_dims(image_data, axis=0)

        return image_data, image_shape

    res = Tuple[np.ndarray, np.ndarray, np.ndarray]
    fres = Tuple[int, float, int, int, int, int]

    def detect_image(self, image_data: Tensor, image_shape: imgshape) -> res | None:
        # 模型预测
        with torch.no_grad():
            images = image_data.to(self.device)
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            w, h = image_shape
            # 将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(
                outputs,
                self.num_classes,
                self.input_shape,
                (h, w),
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
            )

            if results[0] is None:
                return None

            top_label = np.array(results[0][:, 5], dtype="int32")
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]
            return top_label, top_conf, top_boxes

    def formatBoxes(self, results: res, image_shape: imgshape) -> list[fres]:
        top_label, top_conf, top_boxes = results
        Results = list()
        for i in range(len(top_label)):
            label = top_label[i]
            score = top_conf[i]
            box = top_boxes[i]
            top, left, bottom, right = box

            top = max(0, np.floor(top).astype("int32"))
            left = max(0, np.floor(left).astype("int32"))
            bottom = min(image_shape[1], np.floor(bottom).astype("int32"))
            right = min(image_shape[0], np.floor(right).astype("int32"))

            Results.append((label, score, top, left, bottom, right))
        return Results

    def drawResults(
        self, image: Image.Image, results: list[fres] | None
    ) -> Image.Image:
        if results is None:
            return image
        COLOR = (0, 0, 255)
        draw = ImageDraw.Draw(image)

        for i in results:
            label, score, top, left, bottom, right = i

            predicted_class = self.class_names[label]
            text_label = "{} {:.2f}".format(predicted_class, score).encode("utf-8")

            xmin, ymin, xmax, ymax = (left, top, right, bottom)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=COLOR, width=3)
            text_size = draw.textbbox((0, 0), text_label, font=None)
            text_w = text_size[2] - text_size[0]
            text_h = text_size[3] - text_size[1]
            text_bg = (xmin, ymin - text_h, xmin + text_w, ymin)  # 计算文本框位置

            draw.rectangle(text_bg, fill=COLOR)
            draw.text((xmin, ymin - text_h), text_label, fill=(255, 255, 255))

        del draw
        return image

    def predict(self, image: Image.Image) -> list[fres] | None:
        img_data, shape = self.preprocessing_image(image)
        res = self.detect_image(img_data, shape)
        if res is None:
            return None
        result = self.formatBoxes(res, shape)
        return result
