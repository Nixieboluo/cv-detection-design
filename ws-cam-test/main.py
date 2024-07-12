import asyncio
import time

import websockets
import cv2

from common.draw_fps import draw_fps

last_time = time.time()
cap = None
fps = 0


async def send_frame(websocket):
    global last_time
    global cap
    global fps

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cur_time = time.time()
            interval = cur_time - last_time
            fps = 0.0 if interval == 0.0 else 1.0 / interval
            last_time = cur_time

            frame = draw_fps(frame, fps)
            _, encoded_buffer = cv2.imencode(".png", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await websocket.send(encoded_buffer.tobytes())
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")


async def main():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    async with websockets.serve(send_frame, "0.0.0.0", 8900):
        print("WebSocket server listening on port 8900")
        await asyncio.Future()  # run forever

    cap.release()


if __name__ == "__main__":
    asyncio.run(main())
