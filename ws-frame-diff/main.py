import asyncio
import time

import numpy as np
import pandas as pd
import websockets
import cv2

from common.draw_fps import draw_fps

CAM_DEVICE_ID = 0
CAM_WIDTH = 320
CAM_HEIGHT = 240
AREA_THRESHOLD = 250

last_time = time.time()
cap = None
fps = 0
bg_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.uint8)
bin_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.uint8)


def prepare_frame(frame):
    # Convert to Grayscale
    proc_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    proc_frame = cv2.GaussianBlur(proc_frame, (21, 21), 0)

    return proc_frame


async def send_frame(websocket, path):
    global last_time
    global cap
    global fps
    global bg_frame
    global bin_frame

    try:
        while cap.isOpened():
            ret, cap_frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            # Transform
            this_frame = prepare_frame(cap_frame)
            # Calculate difference
            diff = cv2.absdiff(bg_frame, this_frame)

            # Update bg frame when difference < 20% or > 75%
            flat_diff = pd.Series(diff.flatten())
            diff_ratio = 1 - len(flat_diff[flat_diff == 0]) / len(flat_diff)
            if (diff_ratio < 0.2) | (diff_ratio > 0.75):
                bg_frame = this_frame
            else:
                # Binarize
                _, bin_frame = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                # Erode and dilate
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                bin_frame = cv2.dilate(bin_frame, kernel, iterations=1)
                bin_frame = cv2.erode(bin_frame, kernel, iterations=1)

                # Find contours
                contours, _ = cv2.findContours(bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    # Ignore small contours
                    if cv2.contourArea(c) < AREA_THRESHOLD:
                        continue

                    # Draw bounding box
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(cap_frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

            show_frame = draw_fps(
                np.hstack([cap_frame,
                           cv2.cvtColor(this_frame, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)
                           ]),
                fps
            )

            cur_time = time.time()
            interval = cur_time - last_time
            fps = 0.0 if interval == 0.0 else 1.0 / interval
            last_time = cur_time

            _, encoded_buffer = cv2.imencode(".png", show_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await websocket.send(encoded_buffer.tobytes())

            end_time = time.time()

            # Calculate FPS
            frame_interval = end_time - start_time
            fps = 0 if frame_interval <= 0 else (1 / frame_interval)

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")


async def main():
    global cap
    cap = cv2.VideoCapture(CAM_DEVICE_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    async with websockets.serve(send_frame, "0.0.0.0", 8900):
        print("WebSocket server listening on port 8900")
        await asyncio.Future()  # run forever

    cap.release()


if __name__ == "__main__":
    asyncio.run(main())
