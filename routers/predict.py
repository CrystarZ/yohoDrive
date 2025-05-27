import base64
import io
import json
from typing import Tuple
from collections import deque
from PIL import Image
from fastapi import APIRouter, WebSocket, HTTPException
from pydantic import BaseModel
from .uploads import c_fd_upload, r_fd_up, save_frame
from .users import c_fd_user, r_fd_user
from db.mysql.models import Upload, Detections, UserLog
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
ts_phi = conf_ts["phi"]
ts_yolo = YOLO(module_path=ts_mpath, classes=ts_classes, phi=ts_phi)

tl_mpath = conf_tl["module_path"]
tl_classes, _ = get_classes(conf_tl["classes_path"])
tl_phi = conf_tl["phi"]
tl_yolo = YOLO(module_path=tl_mpath, classes=tl_classes, phi=tl_phi)

ocr_mpath = conf_ocr["module_path"]
ocr = OCR(module_path=ocr_mpath)


# NOTE: DETECT

det = Tuple[str, float, int, int, int, int, str | None, int | None]
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
        ttss.append((C, float(c), int(l), int(t), int(r), int(b), None, n))
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
            n = ocr.predict(gray_roi)
        elif C == "greentime":
            box = (l, t, r, b)
            roi = img.crop(box)
            gray_roi = tomygray(roi, "G")
            n = ocr.predict(gray_roi)
        if n is None:
            n = None
        ttls.append((C, float(c), int(l), int(t), int(r), int(b), None, n))
    return ttls


def Detecton(detection: det, upid) -> Detections:
    C, c, x1, y1, x2, y2, T, n = detection
    if T is None:
        T = 0
    if n is None:
        n = 0
    d = Detections(
        class_name=C,
        confidence=c,
        x_min=x1,
        y_min=y1,
        x_max=x2,
        y_max=y2,
        tag=T,
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
    timestamp: int | None = None,  # 同步序号
    upload_id: int | None = None,  # WARN: do not use
) -> list[dict]:
    db = mysql(**conf_db)
    detections = detect(img=img, isSign=isSign, isTl=isTl)
    json_detections = []
    for d in detections:
        dict_data = dict(zip(detkeys, d))
        if timestamp is not None:
            dict_data["timestamp"] = timestamp  # type: ignore
        json_detections.append(dict_data)

    if isSave:
        if upload_id is None:
            user = r_fd_user(db, c_fd_user(id=userid))
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            fn = savename
            fp = save_frame(fn, img)
            up = Upload(filename=fn, filepath=fp, user_id=user.id)
            db.add(up)
            db.refresh(up)
        else:
            up = r_fd_up(db, c_fd_upload(id=upload_id))

        if up is not None:
            for d in detections:
                D = Detecton(d, up.id)
                db.add(D)

    return json_detections


# NOTE: PREDICT

LABEL_TO_WARNING = {  # for tt100k
    "ip": "人行横道",
    "i2": "非机动车行驶",
    "i2r": "非机动车行驶",
    "i4": "机动车行驶",
    "i4l": "机动车行驶",
    "i5": "靠右侧道路行驶",
    "il60": "出口60",
    "il80": "出口80",
    "il100": "出口100",
    "p1": "禁止超车",
    "p5": "禁止掉头",
    "p6": "禁止非机动车进入",
    "p10": "禁止机动车驶入",
    "p11": "禁止鸣喇叭",
    "p12": "禁止二轮摩托车驶入",
    "p13": "禁止某两种车驶入",
    "p14": "禁止直行",
    "p19": "禁止向右转弯",
    "p20": "禁止向左向右转弯",
    "p21": "禁止直行和向右转弯",
    "p23": "禁止向左转弯",
    "p28": "禁止直行和向左转弯",
    "pb": "禁止通行",
    "pc": "停车检查",
    "pd": "海关",
    "pe": "会车让行",
    "pg": "减速让行",
    "pn": "禁止停车",
    "pne": "禁止驶入",
    "pr40": "解除限制速度",
    "ps": "停车让行",
    "w13": "十字交叉路口",
    "w32": "施工",
    "w55": "注意儿童",
    "w57": "注意行人",
    "w59": "注意合流",
    "ph3.5": "限高3.5米",
    "ph4": "限高4米",
    "ph5": "限高5米",
    "ph4.5": "限高4.5米",
    "pl5": "限速5",
    "pl15": "限速15",
    "pl20": "限速20",
    "pl30": "限速30",
    "pl40": "限速40",
    "pl50": "限速50",
    "pl60": "限速60",
    "pl70": "限速70",
    "pl80": "限速80",
    "pl100": "限速100",
    "pl120": "限速120",
    "pm20": "限制质量20吨",
    "pm30": "限制质量30吨",
    "pm55": "限制质量55吨",
    "p3": "禁止大型客车驶入",
    "p26": "禁止载货汽车驶入",
    "p27": "禁止运输危险物品车辆驶入",
    "redleft": "左转红灯",
    "greenleft": "左转绿灯",
    "yellowleft": "左转黄灯",
    "red": "红灯",
    "green": "绿灯",
    "yellow": "黄灯",
    "redstraight": "直行红灯",
    "greenstraight": "直行绿灯",
    "yellowstraight": "直行黄灯",
    "redtime": "红灯等待",
    "greentime": "绿灯等待",
    "yellowtime": "黄灯等待",
    "redpeople": "人行红灯",
    "greenpeople": "人行绿灯",
}
predict_log_maxlen = 20
predict_log = deque(maxlen=predict_log_maxlen)
predict_set = set()  # quick search


def predictQuest(data: list[dict]) -> list[str]:
    res = []
    for i in data:
        num = "_" + str(i["num"]) if i["num"] is not None else ""
        label = f"{i['class_name']}-{LABEL_TO_WARNING.get(i['class_name'], i['class_name'])}{num}"

        if label not in predict_set:
            if len(predict_log) >= predict_log_maxlen:
                removed = predict_log.popleft()
                predict_set.remove(removed)
            predict_log.append(label)
            predict_set.add(label)
            res.append(label)
    return res


# NOTE: ROUTERS


router = APIRouter()


class c_detect_idopt(BaseModel):
    upid: int
    save: bool = False
    sign: bool = False
    tl: bool = False


class c_fd_detect(BaseModel):
    user_id: int
    class_name: str


@router.post("/detect/id")
def detect_id(opt: c_detect_idopt):
    db = mysql(**conf_db)
    try:
        upload = r_fd_up(db, c_fd_upload(id=opt.upid))
        if upload is None:
            raise HTTPException(status_code=404, detail="未找到指定资源")
        path = str(upload.filepath)
        img = Image.open(path)
        detections = detectQuest(
            img, isSign=opt.sign, isTl=opt.tl, isSave=opt.save, upload_id=upload.id
        )

        log = UserLog(
            user_id=upload.user_id, upload_id=upload.id, action="detect_upload"
        )
        db.add(log)

        return detections

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/detect/fd")
def detect_find(opt: c_fd_detect):
    db = mysql(**conf_db)
    try:
        user_id = opt.user_id
        class_name = opt.class_name
        user = r_fd_user(db, c_fd_user(id=user_id))
        if user is None:
            raise HTTPException(status_code=404, detail="Not find user")

        results = (
            db.query(Detections)
            .join(Upload, Detections.upload_id == Upload.id)
            .filter(Upload.id == user_id, Detections.class_name == class_name)
            .all()
        )

        log = UserLog(user_id=user_id, action="find_detections")
        db.add(log)

        for i in results:
            db.refresh(i)

        return results

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


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
            userid = data.get("user_id", 0)
            predict = data.get("predict", None)
            timestamp = data.get("timestamp", None)

            result = detectQuest(
                img,
                isSign=isSign,
                isTl=isTl,
                isSave=isSave,
                savename=savename,
                userid=userid,
            )

            response = dict()
            response["result"] = result

            if timestamp is not None:
                response["timestamp"] = timestamp

            if predict is not None:
                if predict:
                    response["predicts"] = predictQuest(result)

            await websocket.send_text(json.dumps(response))

    except Exception as e:
        print(f"WebSocket 连接错误: {e}")
    finally:
        db.close()
        print("WebSocket 连接已关闭")
