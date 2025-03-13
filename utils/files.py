import os
import time
import random
from enum import IntEnum


class SrcType(IntEnum):
    default = 0
    pic = 11
    video = 12
    avatar = 13


def uniqueFileName(origin_name: str) -> str:
    ext = os.path.splitext(origin_name)[-1]  # 扩展名
    timestamp = int(time.time() * 1_000_000)
    random_num = random.randint(1000, 9999)
    unique_name = f"{timestamp}_{random_num}{ext}"
    return unique_name


def checkFileType(filename):
    ext = filename[filename.rfind(".") :].lower()
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
    video_extensions = [".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm"]

    if ext in image_extensions:
        return SrcType.pic
    elif ext in video_extensions:
        return SrcType.video
    else:
        return SrcType.default


# only for test
if __name__ == "__main__":
    filename = "test_file"
    uniqueName = uniqueFileName(filename)
    print(uniqueName)
