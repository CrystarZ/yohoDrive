import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DSIGN.yolo import YOLO
from PIL import Image
from torchvision import transforms


def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


to_pil = transforms.ToPILImage()

if __name__ == "__main__":
    classes, _ = get_classes("./tl_classes.txt")
    yolo = YOLO(
        module_path="./weights/tl_v8s.pth",
        classes=classes,
        cuda=False,
    )

    img = Image.open("./img/tl_test.jpg")
    result = yolo.predict(img)
    image = yolo.drawResults(img, result)
    image.show()

    if result is not None:
        for i in result:
            label, score, top, left, bottom, right = i
            predicted_class = yolo.class_names[label]
            text_lable = "{} {:.2f}".format(predicted_class, score).encode("utf-8")

            print(text_lable, top, left, bottom, right)
