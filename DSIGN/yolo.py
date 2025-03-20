import numpy as np
import colorsys
from utils.utils_bbox import DecodeBox
from nets.yolo import YoloBody
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


class YOLO(object):
    def __init__(
        self,
        module_path: str,  # 权重路径
        classes: list[str],  # 标签
        input_shape: tuple[int, int] = (640, 640),  # 输入图像尺寸(32的倍数)
        phi: str = "s",  # 对应yolov8版本
        confidence: float = 0.5,  # 置信度
        nms_iou: float = 0.3,  # 非极大抑制所用到的nms_iou大小
        letterbox_image: bool = True,  # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        cuda: bool = True,  # 是否使用Cuda
    ) -> None:
        self.module_path = module_path
        self.input_shape = input_shape
        self.phi = phi
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.cuda = cuda

        self.class_names, self.num_classes = classes, len(classes)
        self.bbox_util = DecodeBox(
            self.num_classes, (self.input_shape[0], self.input_shape[1])
        )
        self.net = self.generate()

    def generate(self, onnx=False):
        # 建立yolo模型
        net = YoloBody(self.input_shape, self.num_classes, self.phi)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load(self.module_path, map_location=device))
        net = net.fuse().eval()
        print("{} model, and classes loaded.".format(self.module_path))
        if not onnx:
            if self.cuda:
                net = nn.DataParallel(net)
                net = net.cuda()
        return net

    def preprocessing_image(self, image):
        # ---  图像预处理  --- #
        #

        image_shape = np.array(np.shape(image)[0:2])  # 获取图像尺寸

        # 强制转换成RGB图像，防止灰度图报错
        if len(np.shape(image)) != 3 and np.shape(image)[2] == 3:
            image = image.convert("RGB")

        # RESIZE 到 input_shape
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
        image_data = np.array(image_data, dtype="float32")
        image_data /= 255.0  # 归一化
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)

        return image_data, image_shape

    def detect_image(self, image_data, image_shape):
        # --- 模型预测 --- #
        #

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            # 将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(
                outputs,
                self.num_classes,
                self.input_shape,
                image_shape,
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

    def formatBoxes(self, results, image_shape):
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

    def drawResults(self, image, results):
        if results is None:
            pass
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(
            font="model_data/simhei.ttf",
            size=np.floor(3e-2 * image.height + 0.5).astype("int32"),
        )
        thickness = int(  # 边框大小
            max((image.width + image.height) // np.mean(self.input_shape), 1)
        )

        hsv_tuples = [(x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors)
        )

        image_shape = (image.width, image.height)
        Results = self.formatBoxes(results, image_shape)

        for i in Results:
            label, score, top, left, bottom, right = i

            predicted_class = self.class_names[label]
            text_label = "{} {:.2f}".format(predicted_class, score).encode("utf-8")

            _, _, l_width, l_height = draw.textbbox((0, 0), text=text_label, font=font)
            label_size = (l_width, l_height)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j], outline=colors[label]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[label],
            )
            draw.text(text_origin, str(text_label, "UTF-8"), fill=(0, 0, 0), font=font)
        del draw
        return image
