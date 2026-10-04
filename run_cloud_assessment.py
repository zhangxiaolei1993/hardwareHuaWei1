#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端对接主程序：Atlas 设备端跑一次完整的表情测评并批量上传 psychology-server。

完整链路：
  读取配置/已存 token → (必要时)注册并落盘 token → 启动 20s 心跳
  → 建立 session（status=running）→ 摄像头采集/人脸检测/NPU 推理/记录 timeline
  → 测评结束 → 一次性批量上传（client_request_id 幂等 + 指数退避重试）
  → session 置 completed → 打印云端返回

headless 运行（SSH 无 DISPLAY），无窗口；SIGTERM / Ctrl-C 优雅结束并上传。

用法：
  source activate.sh
  python run_cloud_assessment.py --duration 60      # 测评 60 秒
  python run_cloud_assessment.py                    # 一直跑到 kill/Ctrl-C
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aclruntime
from expression_recognizer import ExpressionRecognizer
from cloud.device_config import AppConfig, DEFAULT_CONFIG, setup_logging
from cloud.cloud_client import CloudClient, CloudAPIError, CloudValidationError
from cloud.heartbeat import HeartbeatThread

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_registered(cfg, client, log):
    """返回 (token, 是否本次新注册)。token 已存在则验证复用，不重复注册。"""
    token = cfg.load_token()
    if token:
        client.set_token(token)
        try:
            dev = client.get_device()
            log.info("复用已保存 token，设备信息: id=%s name=%s status=%s",
                     dev.get("device_id"), dev.get("device_name"),
                     dev.get("status"))
            return token, False
        except CloudAPIError as e:
            if e.status_code in (401, 403):
                log.warning("已保存 token 无效(%s)，重新注册", e.status_code)
            else:
                raise
    log.info("向云端注册设备 %s ...", cfg.device_id)
    resp = client.register(cfg.register_payload())
    token = resp["device_token"]
    cfg.save_token(token)
    client.set_token(token)
    log.info("注册成功并已落盘 token: created=%s status=%s -> %s",
             resp.get("created"), resp.get("status"), cfg.token_file)
    return token, True


def main():
    ap = argparse.ArgumentParser(description="Atlas 表情测评云端联调主程序")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--duration", type=float, default=0,
                    help="测评秒数；0=一直运行到 SIGTERM/Ctrl-C")
    ap.add_argument("--sample-fps", type=float, default=None)
    ap.add_argument("--infer-interval", type=float, default=None)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270])
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--user-id", default=None)
    args = ap.parse_args()

    cfg = AppConfig(args.config)
    log = setup_logging(cfg.log_file)

    sample_fps = args.sample_fps if args.sample_fps is not None else cfg.sample_fps
    infer_interval = (args.infer_interval if args.infer_interval is not None
                      else cfg.infer_interval)
    cam_idx = args.camera if args.camera is not None else cfg.camera_index
    width = args.width if args.width is not None else cfg.width
    height = args.height if args.height is not None else cfg.height
    rotate = args.rotate if args.rotate is not None else cfg.rotate
    mirror = args.mirror or cfg.mirror

    stop_evt = threading.Event()

    def _on_signal(signum, frame):
        log.info("收到信号 %s，准备结束测评并上传...", signum)
        stop_evt.set()

    signal.signal(signal.SIGTERM, _on_signal)

    # 1) 健康检查 + 注册/复用 token
    client = CloudClient(cfg.server_base_url, cfg.device_id,
                         timeout=cfg.http_timeout,
                         max_retries=cfg.upload_max_retries)
    try:
        h = client.health()
        log.info("云端健康检查: %s", h)
        ensure_registered(cfg, client, log)
    except CloudAPIError as e:
        log.error("无法连接/注册云端，终止: %s", e)
        return 1

    # 2) 心跳线程（贯穿全程，与采集/推理互不阻塞）
    hb = HeartbeatThread(client, interval=cfg.heartbeat_interval,
                         firmware_version=cfg.firmware_version)
    hb.start()

    # 3) 先初始化 NPU 与摄像头，避免占着 session 却采不到流
    model_path = os.path.join(PROJECT_ROOT, "models", "emotion-ferplus.om")
    log.info("加载 NPU 模型 %s ...", model_path)
    recognizer = ExpressionRecognizer(model_path, device_id=0)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        log.error("无法打开摄像头 /dev/video%s", cam_idx)
        hb.stop()
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    log.info("摄像头预热 15 帧...")
    for _ in range(15):
        cap.read()

    # 4) 建立测评会话
    sess = client.create_session(user_id=args.user_id, session_type="emotion")
    session_id = sess["session_id"]
    log.info("会话已建立: session_id=%s status=%s", session_id, sess.get("status"))
    client.update_session_status(session_id, "running")

    # 5) 采集 / 检测 / 推理 / 本地 timeline
    timeline = []
    video_frames = 0
    face_frames = 0
    infer_runs = 0
    t0 = time.time()
    wall_start = datetime.now()
    last_infer_t = 0.0
    cached_box = None
    last_box_ok_t = 0.0
    last_print_t = t0
    sample_period = 1.0 / max(sample_fps, 0.1)

    log.info("开始测评（采样 %.1fHz / 推理 %.1fHz / %s）",
             sample_fps, 1.0 / infer_interval,
             f"定时{args.duration}s" if args.duration else "持续至信号")
    try:
        while not stop_evt.is_set():
            loop_t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                log.warning("读帧失败，跳过")
                time.sleep(0.05)
                continue
            if rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if mirror:
                frame = cv2.flip(frame, 1)

            video_frames += 1
            now = time.time()

            box = recognizer.detect_face(frame)
            if box is not None:
                cached_box = box
                last_box_ok_t = now
            elif now - last_box_ok_t > 0.4:
                cached_box = None  # 0.4s 抗闪烁保持
            if cached_box is not None:
                face_frames += 1

            if cached_box is not None and now - last_infer_t >= infer_interval:
                x, y, w, h = cached_box
                face_img = frame[y:y + h, x:x + w]
                inp = recognizer.preprocess(face_img)
                outputs = recognizer.session.run(
                    [recognizer.output_name],
                    {recognizer.input_name: aclruntime.Tensor(inp)})
                out_t = outputs[0]
                out_t.to_host()
                logits = np.array(out_t)[0]
                probs = np.exp(logits) / np.sum(np.exp(logits))
                idx = int(np.argmax(probs))
                rel = max(0.0, now - t0)
                timeline.append({
                    "relative_seconds": round(rel, 3),
                    "expression": ExpressionRecognizer.LABELS[idx],
                    "confidence": round(float(probs[idx]), 6),
                })
                infer_runs += 1
                last_infer_t = now

            if now - last_print_t >= 5.0:
                last_print_t = now
                cur = timeline[-1]["expression"] if timeline else "-"
                log.info("[%5.1fs] 帧=%d 人脸帧=%d 推理=%d 当前=%s 心跳ok=%d",
                         now - t0, video_frames, face_frames, infer_runs, cur,
                         hb.beats_ok)

            if args.duration > 0 and (now - t0) >= args.duration:
                break

            spare = sample_period - (time.time() - loop_t0)
            if spare > 0:
                stop_evt.wait(spare)
    except KeyboardInterrupt:
        log.info("Ctrl-C 中断，准备上传...")
    finally:
        cap.release()

    wall_end = datetime.now()
    elapsed = time.time() - t0
    if infer_runs == 0:
        log.error("本次测评没有任何有效推理（无人脸？），不上传空数据；"
                  "会话 %s 保留在云端", session_id)
        hb.stop()
        return 2

    # 6) 组装上传体（统计结果一律由云端计算，设备端只给原始 timeline + 帧信息）
    request_id = str(uuid.uuid4())
    meta = {
        "session_start": wall_start.isoformat(timespec="seconds"),
        "session_end": wall_end.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "video_frames": video_frames,
        "display_fps": round(video_frames / elapsed, 2) if elapsed > 0 else 0.0,
        "face_detected_frames": face_frames,
        "inference_runs": infer_runs,
    }
    payload = {
        "client_request_id": request_id,
        "session_meta": meta,
        "timeline": timeline,
    }

    # 本地留底（便于与云端重算结果核对）
    out_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    stamp = wall_end.strftime("%Y%m%d_%H%M%S")
    audit_path = os.path.join(out_dir, f"cloud_upload_{stamp}.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump({"session_id": session_id, **payload}, f,
                  ensure_ascii=False, indent=2)
    log.info("本地留底: %s（%d 条 timeline，request_id=%s）",
             audit_path, len(timeline), request_id)

    # 7) 批量上传（重试复用同一 request_id）
    upload_ok = False
    upload_resp = None
    try:
        upload_resp = client.upload_emotion_data(
            session_id, request_id, meta, timeline,
            max_retries=cfg.upload_max_retries)
        upload_ok = True
        if upload_resp.get("duplicate"):
            log.warning("云端返回 duplicate=true：该 request_id 已提交过，"
                        "未重复入库（幂等生效）")
        else:
            log.info("上传成功，云端已重算结果")
    except CloudValidationError as e:
        log.error("上传被云端 422 拒绝（不重试）: %s", e)
    except CloudAPIError as e:
        log.error("上传最终失败: %s；数据留底于 %s，网络恢复后可补传",
                  e, audit_path)

    if upload_ok:
        client.update_session_status(session_id, "completed")
        log.info("会话已置为 completed")

    hb.stop()
    hb.join(timeout=3)

    print("\n========== 云端返回 ==========")
    print(f"server        : {cfg.server_base_url}")
    print(f"device_id     : {cfg.device_id}")
    print(f"session_id    : {session_id}")
    print(f"request_id    : {request_id}（同一次测评的重试复用，幂等）")
    print(f"timeline 条数 : {len(timeline)}")
    if upload_resp is not None:
        print(json.dumps(upload_resp, ensure_ascii=False, indent=2))
        result_url = (f"{cfg.server_base_url}/api/v1/emotion/sessions/"
                      f"{session_id}/result")
        print(f"结果查询: GET {result_url}")
    else:
        print("上传失败，请查看日志与本地留底文件")
    return 0 if upload_ok else 1


if __name__ == "__main__":
    sys.exit(main())
