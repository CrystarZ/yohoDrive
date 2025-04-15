from typing import Tuple
import cv2
import base64
import asyncio
import websockets
import json

uri = "ws://localhost:8000/detect/con"
d_tl: bool = True
d_sign: bool = True
d_pridict: bool = True


# 还原box
def rect(
    box: Tuple[int, int, int, int], orgsize: Tuple[int, int], resize: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    xm, ym, xM, yM = box
    ow, oh = orgsize
    w, h = resize

    # 还原到原始分辨率
    x1 = xm * (ow / w)
    y1 = ym * (oh / h)
    x2 = xM * (ow / w)
    y2 = yM * (oh / h)

    return int(x1), int(y1), int(x2), int(y2)


async def send_video():
    seq = 0
    async with websockets.connect(uri) as websocket:
        # cap = cv2.VideoCapture(0)  # 0 代表摄像头
        cap = cv2.VideoCapture("./assets/sign.mp4")
        weidth = 320
        height = 240
        color = (0, 255, 0)

        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频的原始帧率
        sample_fps = 5  # 目标采样帧率
        frame_interval = int(fps / sample_fps)  # 计算跳帧间隔
        wait_time = 1.0 / sample_fps

        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 获取当前帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + frame_interval)  # 跳帧

            ret, frame = cap.read()
            if not ret:
                break

            buffer = cv2.resize(frame, (weidth, height))  # resize
            _, buffer = cv2.imencode(".jpg", buffer)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            message = json.dumps(
                {
                    "frame": jpg_as_text,
                    "tl": d_tl,
                    "sign": d_sign,
                    "predict": d_pridict,
                    "timestamp": seq,
                }
            )
            seq += 1
            await websocket.send(message)

            response = await websocket.recv()
            data = json.loads(response)
            print("服务器返回:", data)

            # 画 bboxes
            for i in data["result"]:
                xm, ym, xM, yM = rect(
                    (i["xmin"], i["ymin"], i["xmax"], i["ymax"]),
                    (w_orig, h_orig),
                    (weidth, height),
                )
                num = "_" + str(i["num"]) if i["num"] is not None else ""
                cv2.rectangle(frame, (xm, ym), (xM, yM), color, 2)
                label = f"{i['class_name']}{num}: {i['confidence']:.2f}"
                label_position = (xm, max(ym - 10, 20))  # 避免超出边界
                cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            cv2.imshow("", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(wait_time)

        cap.release()
        cv2.destroyAllWindows()


asyncio.run(send_video())
