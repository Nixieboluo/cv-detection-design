import time

import cv2
import numpy as np
import pandas as pd

from common.draw_fps import draw_fps

CAM_DEVICE_ID = 0
CAM_WIDTH = 320
CAM_HEIGHT = 240

AREA_THRESHOLD = 250

fps = 0
bg_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.uint8)
bin_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.uint8)


def prepare_frame(frame):
    # Convert to Grayscale
    proc_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    proc_frame = cv2.GaussianBlur(proc_frame, (21, 21), 0)

    return proc_frame


def main():
    global fps
    global bg_frame
    global bin_frame

    cap = cv2.VideoCapture(CAM_DEVICE_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # initialize bg frame
    _, cap_frame = cap.read()
    bg_frame = prepare_frame(cap_frame)

    while True:
        start_time = time.time()

        _, cap_frame = cap.read()

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
            contours, hierarchy = cv2.findContours(bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        cv2.imshow("frame", show_frame)

        end_time = time.time()

        # Exit on "ESC" key is pressed
        if cv2.waitKey(33) == 27:
            break

        frame_interval = end_time - start_time
        fps = 0 if frame_interval <= 0 else (1 / frame_interval)


if __name__ == '__main__':
    main()
