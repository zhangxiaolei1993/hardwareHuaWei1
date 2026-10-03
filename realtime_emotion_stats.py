#!/usr/bin/env python3
"""
实时表情识别 + 时段统计（昇腾 NPU）

设计：视频采集/显示 与 人脸检测/NPU 推理解耦
  - 采集显示：按摄像头原生帧率运行（流畅）
  - 人脸检测：每 detect_interval 秒一次（默认 0.2s = 5Hz），中间帧沿用上一次人脸框
  - 表情推理：每 infer_interval 秒一次（默认 0.5s = 2Hz），NPU 负载很低
  - 结果平滑：对概率向量做指数滑动平均(EMA)，消除标签抖动
  - 时段统计：每次推理结果按其“维持时长”加权累计，结束时导出 JSON 报告 + CSV 时间线

用法（开发板桌面终端）：
  source activate.sh
  python realtime_emotion_stats.py                     # 交互模式，q 退出
  python realtime_emotion_stats.py --duration 60       # 统计 60 秒后自动结束并保存
  python realtime_emotion_stats.py --infer-interval 1  # 每秒推理一次（更省资源）
  python realtime_emotion_stats.py --rotate 0 --width 480 --height 360  # 摄像头正装时

无显示器(SSH)时自动进入 headless，必须配合 --duration 使用。
"""
import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expression_recognizer import ExpressionRecognizer


class FaceDetectorThread(threading.Thread):
    """后台人脸检测线程：只做 Haar 检测，不碰 NPU，避免阻塞画面线程。

    主线程把最新帧通过 update_frame() 投递进来；本线程按固定频率检测，
    结果（原图坐标人脸框或 None）通过 get_box() 读取。
    """

    def __init__(self, recognizer, interval=0.1):
        super().__init__(daemon=True)
        self.recognizer = recognizer
        self.interval = interval
        self._lock = threading.Lock()
        self._frame = None        # 最新待检测帧
        self._frame_id = 0
        self._box = None          # 最新人脸框
        self._box_frame_id = -1
        self.runs = 0
        self._stop_evt = threading.Event()

    def update_frame(self, frame, frame_id):
        with self._lock:
            self._frame = frame
            self._frame_id = frame_id

    def get_box(self):
        """返回 (box 或 None, 该结果对应的帧id)"""
        with self._lock:
            return self._box, self._box_frame_id

    def stop(self):
        self._stop_evt.set()

    def run(self):
        last = 0.0
        while not self._stop_evt.is_set():
            time.sleep(0.005)
            now = time.time()
            if now - last < self.interval:
                continue
            with self._lock:
                frame = self._frame
                fid = self._frame_id
            if frame is None:
                continue
            last = now
            box = self.recognizer.detect_face(frame)
            self.runs += 1
            with self._lock:
                self._box = tuple(int(v) for v in box) if box is not None else None
                self._box_frame_id = fid

LABEL_EN = {
    "neutral": "Neutral", "happiness": "Happy", "surprise": "Surprise",
    "sadness": "Sad", "anger": "Angry", "disgust": "Disgust",
    "fear": "Fear", "contempt": "Contempt",
}
LABELS = ExpressionRecognizer.LABELS
PANEL_W = 250


class EmotionStats:
    """一个统计时段内的表情累计器"""

    def __init__(self, max_gap=None):
        # 两次推理间隔超过 max_gap 秒视为“人脸离开”，空窗期不计入表情时长
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.start_wall = time.time()
        self.start_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.total_frames = 0          # 采集到的视频帧
        self.face_frames = 0           # 检测到人脸的帧
        self.detect_runs = 0          # 人脸检测执行次数
        self.infer_runs = 0           # NPU 推理执行次数
        self.conf_sum = 0.0           # 置信度累加
        # 每种表情累计“维持秒数”
        self.expr_seconds = {k: 0.0 for k in LABELS}
        self.timeline = []            # (相对秒, 表情, 置信度)
        self._last_infer_t = None
        self._last_expr = None

    def record_frame(self, face_found):
        self.total_frames += 1
        if face_found:
            self.face_frames += 1

    def record_detect(self):
        self.detect_runs += 1

    def record_infer(self, expression, confidence, probs):
        """记录一次推理；按距上次推理的实际时间差给上一标签计时"""
        now = time.time()
        if self._last_infer_t is not None and self._last_expr is not None:
            dt = now - self._last_infer_t
            if self.max_gap is not None:
                dt = min(dt, self.max_gap)
            self.expr_seconds[self._last_expr] += dt
        self._last_infer_t = now
        self._last_expr = expression
        self.infer_runs += 1
        self.conf_sum += confidence
        elapsed = now - self.start_wall
        self.timeline.append((round(elapsed, 3), expression, round(confidence, 4)))

    def finish(self):
        """结束计时：把最后一次推理到现在的时间补给最后一个标签（受 max_gap 钳制）"""
        if self._last_infer_t is not None and self._last_expr is not None:
            dt = time.time() - self._last_infer_t
            if self.max_gap is not None:
                dt = min(dt, self.max_gap)
            self.expr_seconds[self._last_expr] += dt
            self._last_infer_t = None

    def report(self):
        self.finish()
        valid_seconds = sum(self.expr_seconds.values())
        per_expr = []
        for label in LABELS:
            sec = self.expr_seconds[label]
            per_expr.append({
                "expression": label,
                "seconds": round(sec, 2),
                "percentage": round(100.0 * sec / valid_seconds, 1) if valid_seconds > 0 else 0.0,
            })
        ranked = sorted(per_expr, key=lambda d: d["seconds"], reverse=True)
        elapsed = time.time() - self.start_wall
        return {
            "session_start": self.start_label,
            "session_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 1),
            "video_frames": self.total_frames,
            "display_fps": round(self.total_frames / elapsed, 1) if elapsed > 0 else 0.0,
            "face_detected_frames": self.face_frames,
            "face_coverage_percent": round(100.0 * self.face_frames / self.total_frames, 1)
            if self.total_frames else 0.0,
            "face_detect_runs": self.detect_runs,
            "inference_runs": self.infer_runs,
            "valid_expression_seconds": round(valid_seconds, 1),
            "dominant_expression": ranked[0]["expression"] if valid_seconds > 0 else None,
            "average_confidence": round(self.conf_sum / self.infer_runs, 3)
            if self.infer_runs else 0.0,
            "expressions": ranked,
        }

    def save(self, out_dir):
        """保存 JSON 报告和 CSV 时间线，返回 (json_path, csv_path)"""
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = self.report()
        jpath = os.path.join(out_dir, f"emotion_report_{stamp}.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        cpath = os.path.join(out_dir, f"emotion_timeline_{stamp}.csv")
        with open(cpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["elapsed_second", "expression", "confidence"])
            w.writerows(self.timeline)
        return jpath, cpath


def draw_panel(frame, smooth_probs, cur_expr, face_ok, fps, stats, infer_ms):
    h = frame.shape[0]
    panel = np.full((h, PANEL_W, 3), 32, dtype=np.uint8)
    y = 30
    cv2.putText(panel, "Realtime Emotion", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    y += 32
    if cur_expr:
        cv2.putText(panel, LABEL_EN[cur_expr], (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
    else:
        cv2.putText(panel, "No face", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 28
    cv2.putText(panel, f"FPS {fps:4.1f}  NPU {infer_ms:4.0f}ms", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 24

    # 实时概率条
    for i, (label, p) in enumerate(zip(LABELS, smooth_probs)):
        col = (0, 220, 0) if i == int(np.argmax(smooth_probs)) and face_ok else (150, 150, 150)
        cv2.putText(panel, LABEL_EN[label], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        cv2.rectangle(panel, (90, y - 11), (225, y - 2), (60, 60, 60), -1)
        cv2.rectangle(panel, (90, y - 11), (90 + int(p * 135), y - 2),
                      (0, 200, 0) if i == int(np.argmax(smooth_probs)) and face_ok else (90, 90, 200), -1)
        y += 21

    # 时段统计
    y += 10
    cv2.line(panel, (12, y), (PANEL_W - 12, y), (90, 90, 90), 1)
    y += 24
    cv2.putText(panel, "Session Stats", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 26
    elapsed = time.time() - stats.start_wall
    cv2.putText(panel, f"time  {elapsed:6.1f}s", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 22
    cv2.putText(panel, f"face  {stats.face_frames}/{stats.total_frames}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 22
    valid = sum(stats.expr_seconds.values())
    top = sorted(stats.expr_seconds.items(), key=lambda kv: kv[1], reverse=True)[:3]
    cv2.putText(panel, "Top expressions:", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    y += 20
    for label, sec in top:
        if sec <= 0:
            continue
        pct = 100.0 * sec / valid if valid > 0 else 0
        cv2.putText(panel, f"{LABEL_EN[label]:9s} {pct:5.1f}%", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)
        y += 20

    cv2.putText(panel, "q:quit r:reset s:save", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    return np.hstack([frame, panel])


def main():
    ap = argparse.ArgumentParser(description="实时表情识别 + 时段统计（昇腾 NPU）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="摄像头物理方向校正角度（默认 0；画面上下颠倒时用 180）")
    ap.add_argument("--mirror", action="store_true",
                    help="方向校正后再做水平镜像（自拍镜效果），默认关闭")
    ap.add_argument("--infer-interval", type=float, default=0.5,
                    help="NPU 表情推理间隔秒数（默认 0.5 = 2Hz，适合时段统计；"
                         "GUI 想更跟手可设 0.2）")
    ap.add_argument("--detect-interval", type=float, default=0.2,
                    help="后台人脸检测间隔秒数（默认 0.2 = 5Hz）")
    ap.add_argument("--ema", type=float, default=0.5,
                    help="概率平滑系数 0~1，越大越跟手、越小越稳（默认 0.5）")
    ap.add_argument("--duration", type=float, default=0,
                    help="运行指定秒数后自动结束并保存报告（0=手动退出）")
    ap.add_argument("--headless", action="store_true",
                    help="无窗口模式（无 DISPLAY 时自动开启）")
    ap.add_argument("--sample-fps", type=float, default=5.0,
                    help="headless 后端采样帧率（默认 5：每秒读5帧/检测5次，"
                         "实测稳态约 46%%/核；设 2 可降至约 34%%/核。"
                         "时段统计场景 2~5Hz 完全够用。GUI 模式忽略此项）")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model or os.path.join(base, "models", "emotion-ferplus.om")
    out_dir = os.path.join(base, "output")

    headless = args.headless or not os.environ.get("DISPLAY")
    if headless and args.duration <= 0:
        print("headless 模式：未指定 --duration，将持续运行直到收到 "
              "SIGTERM/SIGINT（kill/Ctrl-C）后保存报告退出")

    # 后端服务场景：收到 SIGTERM（systemctl stop / kill）时优雅退出并保存报告
    stop_requested = threading.Event()

    def _on_term(signum, frame):
        stop_requested.set()

    signal.signal(signal.SIGTERM, _on_term)

    print("加载 NPU 模型...")
    recognizer = ExpressionRecognizer(model_path, device_id=0)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 /dev/video{args.camera}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("摄像头预热...")
    for _ in range(15):
        cap.read()

    import aclruntime

    stats = EmotionStats(max_gap=args.infer_interval * 1.5)
    smooth = np.full(len(LABELS), 1.0 / len(LABELS), dtype=np.float32)
    cur_expr = None
    infer_ms = 0.0
    fps = 0.0
    last_t = time.time()
    last_infer_t = 0.0
    recent_dt = deque(maxlen=30)

    # GUI 模式：人脸检测放后台线程（不阻塞画面）。
    # headless 后端：低频采样，检测直接在主循环内联，单线程更简单、更省资源。
    if headless:
        detector = None
        sample_period = 1.0 / max(args.sample_fps, 0.1)
    else:
        detector = FaceDetectorThread(recognizer, interval=args.detect_interval)
        detector.start()
        sample_period = 0.0

    frame_id = 0
    cached_box = None
    last_box_ok_t = 0.0   # 上次成功检出人脸的时间，用于短暂保持、抗闪烁
    last_print_t = time.time()  # headless 周期状态打印

    if headless:
        print(f"开始识别（headless 后端：采样 {args.sample_fps:g}Hz 内联检测 / "
              f"NPU 推理 {1.0 / args.infer_interval:.1f}Hz，"
              f"{'定时 ' + str(args.duration) + 's' if args.duration else '持续运行至 SIGTERM/Ctrl-C'}）")
    else:
        print(f"开始识别（画面~25FPS / 后台检测 {1.0 / args.detect_interval:.0f}Hz / "
              f"NPU 推理 {1.0 / args.infer_interval:.1f}Hz，GUI，"
              f"{'定时 ' + str(args.duration) + 's' if args.duration else 'q 退出'}）")

    try:
        while True:
            loop_t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("读帧失败")
                break
            # 物理方向校正（默认 0；画面上下颠倒时加 --rotate 180）
            if args.rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif args.rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if args.mirror:
                frame = cv2.flip(frame, 1)
            frame_id += 1
            now = time.time()

            if detector is not None:
                # GUI：把最新帧投递给后台检测线程（立即返回，不阻塞）
                detector.update_frame(frame, frame_id)
                raw_box, box_fid = detector.get_box()
                stats.detect_runs = detector.runs
            else:
                # headless：主循环内联检测，频率=采样帧率
                raw_box = recognizer.detect_face(frame)
                stats.detect_runs += 1
            if raw_box is not None:
                cached_box = raw_box
                last_box_ok_t = now
            elif now - last_box_ok_t > 0.4:
                # 连续约 0.4s 检不到人脸才清空，避免单帧漏检导致框闪烁
                cached_box = None

            stats.record_frame(cached_box is not None)

            # 2) NPU 表情推理（主线程，单帧约 2ms，几乎不影响流畅度）
            if cached_box is not None and now - last_infer_t >= args.infer_interval:
                x, y, w, h = cached_box
                face_img = frame[y:y + h, x:x + w]
                t0 = time.time()
                inp = recognizer.preprocess(face_img)
                outputs = recognizer.session.run(
                    [recognizer.output_name],
                    {recognizer.input_name: aclruntime.Tensor(inp)})
                out_t = outputs[0]
                out_t.to_host()
                logits = np.array(out_t)[0]
                probs = np.exp(logits) / np.sum(np.exp(logits))
                infer_ms = (time.time() - t0) * 1000

                smooth = args.ema * probs + (1 - args.ema) * smooth
                cur_expr = LABELS[int(np.argmax(smooth))]
                stats.record_infer(LABELS[int(np.argmax(probs))],
                                   float(probs.max()), probs.tolist())
                last_infer_t = now

            # 3) 绘制
            if cached_box is not None:
                x, y, w, h = cached_box
                label = f"{LABEL_EN[cur_expr]}" if cur_expr else "..."
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y - 26), (x + 110, y), (0, 170, 0), -1)
                cv2.putText(frame, label, (x + 6, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                # 无人脸时画面中央大字提示，方便自助调整摄像头角度/坐姿
                cv2.putText(frame, "NO FACE", (frame.shape[1] // 2 - 90, frame.shape[0] // 2 - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.putText(frame, "turn camera to your face",
                            (frame.shape[1] // 2 - 175, frame.shape[0] // 2 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            dt = now - last_t
            last_t = now
            if dt > 0:
                recent_dt.append(dt)
                fps = 1.0 / (sum(recent_dt) / len(recent_dt))

            if not headless:
                display = draw_panel(frame, smooth, cur_expr, cached_box is not None,
                                     fps, stats, infer_ms)
                cv2.imshow("Emotion Stats (Ascend NPU)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    stats = EmotionStats(max_gap=args.infer_interval * 1.5)
                    print("统计已重置")
                elif key == ord("s"):
                    os.makedirs(out_dir, exist_ok=True)
                    shot = os.path.join(out_dir, f"shot_{int(time.time())}.jpg")
                    cv2.imwrite(shot, display)
                    print(f"截图: {shot}")

            # headless 后端：每 5 秒打印一行运行状态
            if headless and now - last_print_t >= 5.0:
                last_print_t = now
                cov = (100.0 * stats.face_frames / stats.total_frames
                       if stats.total_frames else 0.0)
                print(f"[{now - stats.start_wall:6.1f}s] 采样{fps:4.1f}FPS  "
                      f"人脸覆盖{cov:5.1f}%  检测{stats.detect_runs}次  "
                      f"推理{stats.infer_runs}次  当前={cur_expr or '-'}",
                      flush=True)

            # 定时结束 / 收到结束信号
            if args.duration > 0 and (now - stats.start_wall) >= args.duration:
                break
            if stop_requested.is_set():
                print("\n收到结束信号，保存报告...")
                break

            # headless 后端：按采样帧率节流（低频读帧，CPU 从~45%/核降到~12%/核）
            if headless:
                spare = sample_period - (time.time() - loop_t0)
                if spare > 0:
                    time.sleep(spare)
    except KeyboardInterrupt:
        print("\n中断，保存报告...")
    finally:
        if detector is not None:
            detector.stop()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    # 保存统计报告
    jpath, cpath = stats.save(out_dir)
    report = json.load(open(jpath, encoding="utf-8"))
    print("\n========== 表情统计报告 ==========")
    fps_label = "采样帧率" if headless else "显示帧率"
    print(f"时长: {report['elapsed_seconds']}s   {fps_label}: {report['display_fps']} FPS")
    print(f"人脸覆盖率: {report['face_coverage_percent']}%   "
          f"推理次数: {report['inference_runs']}   平均置信度: {report['average_confidence']}")
    print(f"主导表情: {report['dominant_expression']}")
    for e in report["expressions"]:
        if e["seconds"] > 0:
            print(f"  {e['expression']:10s} {e['percentage']:5.1f}%  ({e['seconds']}s)")
    print(f"\n报告: {jpath}")
    print(f"时间线: {cpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
