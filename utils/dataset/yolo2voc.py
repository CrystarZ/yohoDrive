import os
from PIL import Image
import glob

yolo_img_dir = "./.output/cd_yolo/img"
yolo_txt_dir = "./.output/cd_yolo/label"
output_dir = "./.output/dataset"

yolo_img_dir = "../../LEDdataset/image"
yolo_txt_dir = "../../LEDdataset/label/5.9"
output_dir = "../../temp2"
annotations_root_path = f"{output_dir}/Annotations"
images_root_path = f"{output_dir}/JPEGImages"
isMV = True

# 目标类别
labels = [
    "tlr",
    "tly",
    "tlg",
    "tlra",
    "tlya",
    "tlga",
    "tsra",
    "tsya",
    "tsga",
    "trra",
    "trya",
    "trga",
    "tura",
    "tuya",
    "tuga",
    "tln",
    "117",
    "118",
    "119",
    "120",
    "121",
    "122",
    "123",
    "124",
    "125",
    "126",
    "127",
    "128",
    "129",
    "130",
    "131",
    "132",
    "133",
    "134",
    "135",
    "136",
    "137",
    "138",
    "139",
    "140",
    "141",
    "tnr",
    "tny",
    "tng",
    "tcr",
    "tcg",
    "tcn",
    "tllv",
    "tllvr",
    "tllvy",
    "tllvg",
    "tlrv",
    "tlrvr",
    "tlrvy",
    "tlrvg",
    "tlbv",
    "tclv",
    "tclvr",
    "tclvg",
    "tcrv",
    "tcrvr",
    "tcrvgv",
    "tcbv",
]

labels = [
    "0",
    "red_1",
    "red_2",
    "red_3",
    "red_4",
    "red_5",
    "red_6",
    "red_7",
    "red_8",
    "red_9",
    "green_1",
    "green_2",
    "green_3",
    "green_4",
    "green_5",
    "green_6",
    "green_7",
    "green_8",
    "green_9",
    "yellow_1",
    "yellow_3",
    "yellow_2",
]


def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


labels, _ = get_classes(f"{yolo_txt_dir}/classes.txt")

# 匹配文件路径下的所有txt文件，并返回列表
txt_glob = glob.glob(f"{yolo_txt_dir}/*.txt")
txt_base_names = []
txt_pre_name = []

for txt in txt_glob:
    # os.path.basename:取文件的后缀名
    txt_base_names.append(os.path.basename(txt))

for txt in txt_base_names:
    # os.path.splitext:将文件按照后缀切分为两块
    temp1, temp2 = os.path.splitext(txt)
    txt_pre_name.append(temp1)
    # print(f"imgpre:{img_pre_name}")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(annotations_root_path, exist_ok=True)
os.makedirs(images_root_path, exist_ok=True)

for id, txt in enumerate(txt_pre_name):
    id = str(id).zfill(6)
    if not os.path.exists(f"{yolo_img_dir}/{txt}.jpg"):
        continue
    os.rename(f"{yolo_img_dir}/{txt}.jpg", f"{images_root_path}/{id}.jpg")
    with open(f"{annotations_root_path}/{id}.xml", "w") as xml_files:
        image = Image.open(f"{images_root_path}/{id}.jpg")
        img_w, img_h = image.size
        img_ch = len(image.mode)
        xml_files.write("<annotation>\n")
        xml_files.write("   <folder>folder</folder>\n")
        xml_files.write(f"   <filename>{id}.jpg</filename>\n")
        xml_files.write("   <source>\n")
        xml_files.write("   <database>Unknown</database>\n")
        xml_files.write("   </source>\n")
        xml_files.write("   <size>\n")
        xml_files.write(f"     <width>{img_w}</width>\n")
        xml_files.write(f"     <height>{img_h}</height>\n")
        xml_files.write(f"     <depth>{img_ch}</depth>\n")
        xml_files.write("   </size>\n")
        xml_files.write("   <segmented>0</segmented>\n")
        with open(f"{yolo_txt_dir}/{txt}.txt", "r") as f:
            # 以列表形式返回每一行
            lines = f.read().splitlines()
            for each_line in lines:
                line = each_line.split(" ")
                xml_files.write("   <object>\n")
                xml_files.write(f"      <name>{labels[int(line[0])]}</name>\n")
                xml_files.write("      <pose>Unspecified</pose>\n")
                xml_files.write("      <truncated>0</truncated>\n")
                xml_files.write("      <difficult>0</difficult>\n")
                xml_files.write("      <bndbox>\n")
                center_x = round(float(line[1]) * img_w)
                center_y = round(float(line[2]) * img_h)
                bbox_w = round(float(line[3]) * img_w)
                bbox_h = round(float(line[4]) * img_h)
                xmin = str(int(center_x - bbox_w / 2))
                ymin = str(int(center_y - bbox_h / 2))
                xmax = str(int(center_x + bbox_w / 2))
                ymax = str(int(center_y + bbox_h / 2))
                xml_files.write(f"         <xmin>{xmin}</xmin>\n")
                xml_files.write(f"         <ymin>{ymin}</ymin>\n")
                xml_files.write(f"         <xmax>{xmax}</xmax>\n")
                xml_files.write(f"         <ymax>{ymax}</ymax>\n")
                xml_files.write("      </bndbox>\n")
                xml_files.write("   </object>\n")
        xml_files.write("</annotation>")
