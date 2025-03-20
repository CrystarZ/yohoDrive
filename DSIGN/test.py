from myyolo import YOLO
from PIL import Image
import matplotlib.pyplot as plt

from dataset import VOCDataset as ds

d = ds("./VOCdevkit/VOC2007", input_shape=[640, 640], sets="val")
i, label = d[d.xmls.index("003229")]
i = i.transpose(1, 2, 0)
plt.imshow(i)
plt.show()
print(label)


# yolo = YOLO(
#     cuda=False,
# )
# img = Image.open("./img/street.jpg")
# image_data, shape = yolo.preprocessing_image(img)
# # print(image.shape)
#
# result = yolo.detect_image(image_data, shape)
# Result = yolo.formatBoxes(result, shape)
# image = yolo.drawResults(img, result)
# image.show()
#
# for i in Result:
#     label, score, top, left, bottom, right = i
#     predicted_class = yolo.class_names[label]
#     text_lable = "{} {:.2f}".format(predicted_class, score).encode("utf-8")
#
#     print(text_lable, top, left, bottom, right)
