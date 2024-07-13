import time
import cv2
import numpy as np

from common.draw_fps import draw_fps  # 导入绘制帧率的函数

CAM_DEVICE_ID = 0  # 摄像头设备 ID
CAM_WIDTH = 640  # 视频流宽度
CAM_HEIGHT = 360  # 视频流高度

fps = 0

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorMOG2()


def main():
    global fps

    cap = cv2.VideoCapture(CAM_DEVICE_ID)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)  # 设置视频流宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)  # 设置视频流高度

    while True:
        start_time = time.time()  # 记录开始时间

        _, frame = cap.read()  # 读取视频帧
        excl_frame = fgbg.apply(frame)  # 应用背景减法


        # 可选：对 fgmask 应用形态学操作（例如开运算）以改善检测效果
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
        cv2.imshow("frame", show_frame)  # 显示帧

        end_time = time.time()  # 记录结束时间

        if cv2.waitKey(33) == 27:  # 如果按下 "ESC" 键，退出循环
            break

        frame_interval = end_time - start_time
        fps = 0 if frame_interval <= 0 else (1 / frame_interval)  # 计算帧率


if __name__ == '__main__':
    main()
