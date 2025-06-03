import sys
import os
import torch
from torchsummary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DSIGN.nets.yolo import YoloBody

CLASSES_PATH = "./weights/tl_classes.txt"
INPUT_SHAPE = (3, 640, 640)
PHI = "s"
BACKBONE = "MNV3"


def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


if __name__ == "__main__":
    classes, num_classes = get_classes(CLASSES_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloBody(num_classes=num_classes, phi=PHI, bb=BACKBONE)
    model = model.to(device)
    summary(model, INPUT_SHAPE)
