import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OCR.ocr import OCR, tomygray
from PIL import Image

WEIGHT_PATH = "./weights/tl_v8s.pth"
TEST_PATH = "./img/OCR_test.jpg"
CH = "R"

if __name__ == "__main__":
    img = Image.open(TEST_PATH)
    img.show()

    img = tomygray(img, CH)
    ocr = OCR(module_path=WEIGHT_PATH)
    print(ocr.predict(img))
