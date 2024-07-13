import asyncio
import time
import websockets
import cv2
import numpy as np

from common.draw_fps import draw_fps

CAM_DEVICE_ID = 0
CAM_WIDTH = 320
CAM_HEIGHT = 240

AREA_THRESHOLD = 250
MOTION_THRESHOLD = 2

fps = 0
prev_gray = None

def prepare_frame(frame):
    proc_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return proc_frame

def draw_optical_flow(frame, flow, step=16):
    h, w = frame.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

async def send_frame(websocket):
    global fps
    global prev_gray

    cap = cv2.VideoCapture(CAM_DEVICE_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    _, cap_frame = cap.read()
    prev_gray = prepare_frame(cap_frame)

    try:
        while cap.isOpened():
            start_time = time.time()

            ret, cap_frame = cap.read()
            if not ret:
                break

            gray_frame = prepare_frame(cap_frame)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prev_gray = gray_frame

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = mag > MOTION_THRESHOLD

            contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < AREA_THRESHOLD:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(cap_frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

            flow_frame = draw_optical_flow(gray_frame, flow)
            show_frame = draw_fps(np.hstack([cap_frame, flow_frame]), fps)

            _, encoded_buffer = cv2.imencode(".png", show_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await websocket.send(encoded_buffer.tobytes())

            end_time = time.time()
            frame_interval = end_time - start_time
            fps = 0 if frame_interval <= 0 else (1 / frame_interval)

            if cv2.waitKey(33) == 27:
                break

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")

    cap.release()

async def main():
    async with websockets.serve(send_frame, "0.0.0.0", 8900):
        print("WebSocket server listening on port 8900")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
