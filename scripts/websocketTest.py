import cv2
import base64
import asyncio
import websockets
import json

uri = "ws://localhost:8000/detect/con"


async def send_video():
    async with websockets.connect(uri) as websocket:
        # cap = cv2.VideoCapture(0)  # 0 代表摄像头
        cap = cv2.VideoCapture("./.output/video.mp4")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)  # 降低分辨率
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            message = json.dumps({"frame": jpg_as_text, "tl": True})
            await websocket.send(message)

            response = await websocket.recv()
            form_data = json.loads(response)
            print("服务器返回:", form_data)

            await asyncio.sleep(0.003)  # 控制帧率


asyncio.run(send_video())
