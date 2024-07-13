import asyncio
import time
import numpy as np
import websockets
import cv2
from common.draw_fps import draw_fps  # 导入绘制帧率的函数

CAM_DEVICE_ID = 0  # 摄像头设备 ID
CAM_WIDTH = 640  # 视频流宽度
CAM_HEIGHT = 360  # 视频流高度

fps = 0
fgbg = cv2.createBackgroundSubtractorMOG2()

last_time = time.time()
cap = None


async def send_frame(websocket, path):
    global last_time
    global cap
    global fps

    try:
        while cap.isOpened():
            start_time = time.time()  # 记录开始时间

            ret, frame = cap.read()  # 读取视频帧
            if not ret:
                break

            excl_frame = fgbg.apply(frame)  # 应用背景减法

            # 可选：对 excl_frame 应用形态学操作（例如开运算）以改善检测效果
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            excl_frame = cv2.erode(excl_frame, kernel, iterations=1)
            excl_frame = cv2.dilate(excl_frame, kernel, iterations=1)
            excl_frame = cv2.morphologyEx(excl_frame, cv2.MORPH_OPEN, kernel)

            # 查找轮廓并绘制边界框
            contours, _ = cv2.findContours(excl_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 250:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

            show_frame = draw_fps(
                np.hstack([frame,
                           cv2.cvtColor(excl_frame, cv2.COLOR_GRAY2BGR),
                           ]),
                fps
            )

            cur_time = time.time()
            interval = cur_time - last_time
            fps = 0.0 if interval == 0.0 else 1.0 / interval
            last_time = cur_time

            _, encoded_buffer = cv2.imencode(".png", show_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await websocket.send(encoded_buffer.tobytes())

            end_time = time.time()  # 记录结束时间

            frame_interval = end_time - start_time
            fps = 0 if frame_interval <= 0 else (1 / frame_interval)  # 计算帧率

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")
    finally:
        cap.release()


async def main():
    global cap
    cap = cv2.VideoCapture(CAM_DEVICE_ID)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)  # 设置视频流宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)  # 设置视频流高度

    async with websockets.serve(send_frame, "0.0.0.0", 8900):
        print("WebSocket server listening on port 8900")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
