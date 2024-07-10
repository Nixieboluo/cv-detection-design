import cv2
import numpy as np


def draw_fps(frame, fps):
    # White text on non full-color images
    if len(np.shape(frame)) < 3:
        text_color = (255, 255, 255)
    # Green text by default
    else:
        text_color = (0, 255, 0)

    fps_text = "FPS: {:.1f}".format(fps)

    cv2.putText(frame, fps_text, (24, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, text_color, 1)

    return frame
