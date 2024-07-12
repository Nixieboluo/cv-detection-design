# frame-diff/main.py
import time
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
    # Convert to Grayscale
    proc_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return proc_frame

#绘制光流场，并在图像上显示光流向量
def draw_optical_flow(frame, flow, step=16):
    # 计算图像的高度和宽度
    h, w = frame.shape[:2]
    # 生成网格点
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    # 获取光流矢量
    fx, fy = flow[y, x].T
    # 生成光流线条
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # 将灰度图像转换为彩色图像
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # 绘制多边形线条
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    # 绘制光流矢量的起点
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis



def main():
    global fps
    global prev_gray

    # 初始化视频捕获设备
    cap = cv2.VideoCapture(CAM_DEVICE_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # 初始化前一帧
    _, cap_frame = cap.read()
    prev_gray = prepare_frame(cap_frame)

    while True:
        start_time = time.time()

        _, cap_frame = cap.read()

        # 初始化前一帧
        gray_frame = prepare_frame(cap_frame)
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray_frame

        # 计算光流的大小和角度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # 根据阈值找到运动区域
        motion_mask = mag > MOTION_THRESHOLD

        # 找到运动区域的轮廓
        contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # 忽略小轮廓
            if cv2.contourArea(c) < AREA_THRESHOLD:
                continue

            # 绘制边界框
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(cap_frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

        # 绘制光流用于可视化
        flow_frame = draw_optical_flow(gray_frame, flow)
        # 绘制FPS
        show_frame = draw_fps(np.hstack([cap_frame, flow_frame]), fps)
        cv2.imshow("frame", show_frame)
        # 更新帧率
        end_time = time.time()

        # 按下“ESC”键退出
        if cv2.waitKey(33) == 27:
            break

        frame_interval = end_time - start_time
        fps = 0 if frame_interval <= 0 else (1 / frame_interval)


if __name__ == '__main__':
    main()
