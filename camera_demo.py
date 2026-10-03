#!/usr/bin/env python3
"""
摄像头实时表情识别演示（昇腾 NPU 加速）

功能：
  - 实时人脸检测 + 8 类表情识别
  - 画面显示人脸框、表情、置信度、FPS、耗时
  - 右侧面板显示各类表情概率条
  - 按 q 退出，按 s 保存当前画面截图
  - 支持摄像头物理方向校正（本机摄像头倒装，默认旋转 180）与可选镜像

用法（在开发板桌面终端）：
  source activate.sh
  python camera_demo.py                 # 默认摄像头 0
  python camera_demo.py --camera 1      # 指定摄像头
  python camera_demo.py --rotate 0      # 摄像头正装时关闭旋转
  python camera_demo.py --mirror        # 转正后再镜像
  python camera_demo.py --width 1280 --height 720
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expression_recognizer import ExpressionRecognizer

# 英文标签，避免 OpenCV putText 中文乱码
LABEL_EN = {
    "neutral": "Neutral",
    "happiness": "Happy",
    "surprise": "Surprise",
    "sadness": "Sad",
    "anger": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "contempt": "Contempt",
}
PANEL_WIDTH = 220
BAR_MAX = 120


def draw_panel(frame, result, fps, infer_ms):
    """在画面右侧绘制概率面板"""
    h = frame.shape[0]
    panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "Expression", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, f"FPS: {fps:.1f}", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(panel, f"NPU: {infer_ms:.0f} ms", (110, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    y = 90
    probs = result.get("probabilities") or [0.0] * len(ExpressionRecognizer.LABELS)
    top_idx = int(np.argmax(probs)) if result.get("success") else -1

    for i, (label, prob) in enumerate(zip(ExpressionRecognizer.LABELS, probs)):
        name = LABEL_EN[label]
        color = (0, 255, 0) if i == top_idx else (180, 180, 180)
        cv2.putText(panel, name, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        bar_len = int(prob * BAR_MAX)
        cv2.rectangle(panel, (12, y + 8), (12 + BAR_MAX, y + 18),
                      (70, 70, 70), -1)
        cv2.rectangle(panel, (12, y + 8), (12 + bar_len, y + 18),
                      (0, 200, 0) if i == top_idx else (100, 100, 255), -1)
        cv2.putText(panel, f"{prob * 100:5.1f}%", (140, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y += 34

    cv2.putText(panel, "q: quit   s: save", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return np.hstack([frame, panel])


def main():
    parser = argparse.ArgumentParser(description="摄像头实时表情识别（昇腾 NPU）")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备 ID")
    parser.add_argument("--model", default=None, help="OM 模型路径")
    parser.add_argument("--width", type=int, default=640, help="采集宽度")
    parser.add_argument("--height", type=int, default=480, help="采集高度")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="摄像头物理方向校正角度（默认 0；画面上下颠倒时用 180）")
    parser.add_argument("--mirror", action="store_true", help="校正后再水平镜像")
    args = parser.parse_args()

    model_path = args.model or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "emotion-ferplus.om")

    print("加载 NPU 模型...")
    recognizer = ExpressionRecognizer(model_path, device_id=0)
    print("模型加载完成")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 /dev/video{args.camera}")
        print("请检查：1) USB 摄像头是否插好  2) 设备 ID 是否正确（ls /dev/video*）")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 预热：丢弃前几帧，等待自动曝光/白平衡收敛
    print("摄像头预热中...")
    for _ in range(15):
        cap.read()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(save_dir, exist_ok=True)

    print("摄像头已开启，按 q 退出，按 s 保存截图")
    fps = 0.0
    infer_ms = 0.0
    last_t = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头帧失败")
                break

            # 物理方向校正：本机摄像头倒装，默认旋转 180
            if args.rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif args.rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if args.mirror:
                frame = cv2.flip(frame, 1)

            t0 = time.time()
            result = recognizer.predict(frame)
            infer_ms = (time.time() - t0) * 1000

            if result["success"]:
                x, y, w, h = result["face_box"]
                label = f"{LABEL_EN[result['expression']]} {result['confidence'] * 100:.0f}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y - 28), (x + max(160, len(label) * 13), y),
                              (0, 180, 0), -1)
                cv2.putText(frame, label, (x + 6, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No face", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # FPS 滑动平均
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            display = draw_panel(frame, result, fps, infer_ms)
            cv2.imshow("Expression Recognition (Ascend NPU)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = os.path.join(save_dir, f"shot_{int(time.time())}.jpg")
                cv2.imwrite(fname, display)
                print(f"截图已保存: {fname}")
    except KeyboardInterrupt:
        print("\n手动中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("已退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
