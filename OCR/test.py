import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from OCR.ocr import OCR, tomygray
from PIL import Image

if __name__ == "__main__":
    img = Image.open("./img/test.jpg")
    img.show()

    img = tomygray(img, "R")
    ocr = OCR(module_path="./weights/cnn2.pth", cuda=False)
    print(ocr.predict(img))
