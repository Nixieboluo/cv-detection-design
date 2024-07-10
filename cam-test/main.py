import time

import cv2

from common.draw_fps import draw_fps

CAM_DEVICE_ID = 0
CAM_WIDTH = 640
CAM_HEIGHT = 360

fps = 0


def main():
    global fps

    cap = cv2.VideoCapture(CAM_DEVICE_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    while True:
        start_time = time.time()

        _, frame = cap.read()
        frame = draw_fps(frame, fps)

        cv2.imshow("frame", frame)

        end_time = time.time()

        # Exit on "ESC" key is pressed
        if cv2.waitKey(33) == 27:
            break

        frame_interval = end_time - start_time
        fps = 0 if frame_interval <= 0 else (1 / frame_interval)


if __name__ == '__main__':
    main()
