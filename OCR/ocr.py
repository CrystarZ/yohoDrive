from .nets.LeNet5 import OCRNUM
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2


def tomygray(image: Image.Image, ch: str) -> Image.Image:
    gray = image.getchannel(ch)
    return gray


class OCR(object):
    def __init__(self, module_path: str = "./weights/cnn2.pth", cuda: bool = True):
        self.module_path = module_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.generate()

    def generate(self):
        net = OCRNUM()
        net.load_state_dict(torch.load(self.module_path, map_location=self.device))
        net = net.to(self.device)
        net = net.eval()
        print("{} model, and classes loaded.".format(self.module_path))

        return net

    def test(self, gray_img: np.ndarray):
        transform = transforms.ToTensor()
        input = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_LINEAR)
        input = transform(input)
        inputs = torch.unsqueeze(input, dim=0)
        inputs = inputs.to(self.device)

        test_output = self.net(inputs)
        pred_y = torch.max(test_output, 1)[1].data.numpy()

        return pred_y

    def predict(
        self, image: Image.Image, bin_threshold=65, dilate_kernel_size=(3, 3)
    ) -> str:
        gray_img = image.convert("L")
        gray_img = np.array(gray_img, dtype=np.uint8)

        # 二值化
        meanvalue = gray_img.mean()

        if meanvalue >= 200:
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            _, _, _, max_index = cv2.minMaxLoc(hist)
            _, image_bin = cv2.threshold(
                gray_img, int(max_index[1]) - 7, 255, cv2.THRESH_BINARY
            )
        else:
            _, image_bin = cv2.threshold(
                gray_img, meanvalue + bin_threshold, 255, cv2.THRESH_BINARY
            )

        # 膨胀处理
        kernel = np.ones(dilate_kernel_size, np.int8)
        image_dil = cv2.dilate(image_bin, kernel, iterations=1)

        # cv2.imshow("", image_dil)
        # cv2.waitKey(0)

        # ROI检测
        contours, _ = cv2.findContours(
            image_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boundRect = []  # 轮廓的边界框
        predict_text = ""  # 识别的结果
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            if h / w > 1:
                # red_dil = cv2.rectangle(image_dil, (x, y), (x + w, y + h), 255, 2)
                if w * h >= 80:
                    boundRect.append([x, y, w, h])

                    height = image_dil.shape[0]
                    width = image_dil.shape[1]
                    ymin, ymax = y - 2, y + h + 2
                    xmin, xmax = x - 2, x + w + 2
                    if xmin < 0:
                        xmin = 0
                    if xmax > width:
                        xmax = width
                    if ymin < 0:
                        ymin = 0
                    if ymax > height:
                        ymax = height

                    roi = image_dil[ymin:ymax, xmin:xmax]
                    # cv2.imshow("", roi)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # 拼接输出
                    predict = self.test(roi)
                    if len(boundRect) != 0:
                        if x > boundRect[0][0]:
                            predict_text += str(predict[0])
                        else:
                            predict_text = str(predict[0]) + predict_text
                    else:
                        predict_text += str(predict[0])

        if len(predict_text) > 2:
            predict_text = predict_text[0:2]

        return predict_text
