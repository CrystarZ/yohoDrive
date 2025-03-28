import base64
import io
import json
from typing import Tuple
from PIL import Image
from fastapi import APIRouter, WebSocket, HTTPException
from pydantic import BaseModel
from .uploads import c_fd_upload, find_upload, save_frame
from .users import c_fd_user, find_user
from db.mysql.models import Upload, Detections
from DSIGN.yolo import YOLO
from OCR.ocr import OCR, tomygray
from . import pwd
from db.mysql.database import database as mysql
from config import decoder as conf


def s2b(s: str) -> bool:
    return s.lower() in ["true"]


def get_classes(classes_path) -> Tuple[list[str], int]:
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


conf_ts = conf(f"{pwd}/config.conf").Section("DSIGN").dict
conf_tl = conf(f"{pwd}/config.conf").Section("TL").dict
conf_ocr = conf(f"{pwd}/config.conf").Section("OCR").dict
conf_db = conf(f"{pwd}/config.conf").Section("database").dict

ts_mpath = conf_ts["module_path"]
ts_classes, _ = get_classes(conf_ts["classes_path"])
ts_cuda = s2b(conf_ts["cuda"])
ts_yolo = YOLO(module_path=ts_mpath, classes=ts_classes, phi="n", cuda=ts_cuda)

tl_mpath = conf_tl["module_path"]
tl_classes, _ = get_classes(conf_tl["classes_path"])
tl_cuda = s2b(conf_tl["cuda"])
tl_yolo = YOLO(module_path=tl_mpath, classes=tl_classes, cuda=tl_cuda)

ocr_mpath = conf_ocr["module_path"]
ocr_cuda = s2b(conf_ocr["cuda"])
ocr = OCR(module_path=ocr_mpath, cuda=ocr_cuda)


type det = Tuple[str, float, int, int, int, int, str | None, int | None]
detkeys = ("class_name", "confidence", "xmin", "ymin", "xmax", "ymax", "tag", "num")


def predict_traffic_signs(img: Image.Image) -> list[det] | None:
    tss = ts_yolo.predict(img)
    if tss is None:
        return None
    ttss = []
    for i in tss:
        C, c, t, l, b, r = i
        C = str(ts_classes[C])
        n = None
        ttss.append((C, float(c), int(t), int(l), int(b), int(r), None, n))
    return ttss


def predict_traffic_lights(img: Image.Image) -> list[det] | None:
    tls = tl_yolo.predict(img)
    if tls is None:
        return None
    ttls = []
    for i in tls:
        C, c, t, l, b, r = i
        C = str(tl_classes[C])
        n = None
        if C == "redtime":
            box = (l, t, r, b)
            roi = img.crop(box)
            gray_roi = tomygray(roi, "R")
            n = int(ocr.predict(gray_roi))
        if C == "greentime":
            box = (l, t, r, b)
            roi = img.crop(box)
            gray_roi = tomygray(roi, "G")
            n = int(ocr.predict(gray_roi))
        ttls.append((C, float(c), int(t), int(l), int(b), int(r), None, n))
    return ttls


def Detecton(detection: det, upid) -> Detections:
    C, c, t, l, b, r, T, n = detection
    if T is None:
        T = 0
    if n is None:
        n = 0
    d = Detections(
        class_name=C,
        confidence=c,
        x_min=l,
        y_min=t,
        x_max=r,
        y_max=b,
        tag=t,
        num=n,
        upload_id=upid,
    )
    return d


def detect(img: Image.Image, isSign: bool = False, isTl: bool = False) -> list[det]:
    detections: list[det] = []
    if isSign:
        ts_detections = predict_traffic_signs(img)
        if ts_detections is not None:
            detections += ts_detections
    if isTl:
        tl_detections = predict_traffic_lights(img)
        if tl_detections is not None:
            detections += tl_detections
    return detections


def detectQuest(
    img: Image.Image,
    isSign: bool = False,
    isTl: bool = False,
    isSave: bool = False,
    userid: int = 0,
    savename: str = "frame.jpg",
):
    db = mysql(**conf_db)
    detections = detect(img=img, isSign=isSign, isTl=isTl)
    json_detections = []
    for d in detections:
        dict_data = dict(zip(detkeys, d))
        json_detections.append(dict_data)

    if isSave:
        user = find_user(c_fd_user(id=userid))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        fn = savename
        fp = save_frame(fn, img)
        up = Upload(filename=fn, filepath=fp, user_id=user.id)
        db.add(up)
        db.refresh(up)
        for d in detections:
            D = Detecton(d, up.id)
            db.add(D)

    return json_detections


router = APIRouter()


class c_detect_idopt(BaseModel):
    upid: int
    sign: bool = False
    tl: bool = False


@router.post("/detect/id")
def detect_id(opt: c_detect_idopt):
    upload = find_upload(c_fd_upload(id=opt.upid))
    path = str(upload.filepath)
    img = Image.open(path)

    return detectQuest(img, isSign=opt.sign, isTl=opt.tl)


# import numpy as np
# import cv2


@router.websocket("/detect/con")
async def receive_video(websocket: WebSocket):
    await websocket.accept()
    db = mysql(**conf_db)
    print("WebSocket 连接已建立，等待视频流...")
    try:
        while True:
            message = await websocket.receive_text()  # 接收数据
            data = json.loads(message)

            img_data = base64.b64decode(data["frame"])
            img = Image.open(io.BytesIO(img_data))  # 转换为 PIL.Image

            # 显示视频帧
            # np_arr = np.frombuffer(img_data, dtype=np.uint8)
            # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # cv2.imshow("Received Frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

            isSign = data.get("sign", False)
            isTl = data.get("tl", False)
            # save options
            isSave = data.get("save", False)
            savename = data.get("savename", "frame.jpg")
            userid = int(data.get("user_id", 0))

            response = detectQuest(
                img,
                isSign=isSign,
                isTl=isTl,
                isSave=isSave,
                savename=savename,
                userid=userid,
            )
            await websocket.send_text(json.dumps(response))

    except Exception as e:
        print(f"WebSocket 连接错误: {e}")
    finally:
        db.close()
        print("WebSocket 连接已关闭")
