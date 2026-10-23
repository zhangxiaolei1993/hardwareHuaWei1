# -*- coding: utf-8 -*-
"""摄像头打开工具：统一 V4L2 后端与 MJPG 压缩格式"""
import cv2


def open_camera(index=0, width=1280, height=720, mjpg=True):
    """
    打开摄像头并协商采集格式。

    - 显式锁定 V4L2 后端，避免 OpenCV 在失败时回退到 obsensor 后端，
      报出误导性的 "Camera index out of range"
    - 优先 MJPG：USB 2.0 带宽下 YUYV 最高只能到 640x480，720p 必须用压缩格式
    - 首选项打不开时回退默认后端，兼容少数不认 CAP_V4L2 的平台

    Args:
        index: 设备号（/dev/video{index}）
        width/height: 期望分辨率
        mjpg: 是否强制 MJPG 压缩格式

    Returns:
        cv2.VideoCapture 对象（可能未打开，调用方需检查 isOpened()）
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return cap
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
