#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atlas 常驻测评主程序：测评时长完全跟随手机端会话生命周期。

状态机（外循环）：
  注册/复用 token → 启动 20s 心跳（常驻）→ 加载 NPU/打开摄像头（仅一次）
  → 启动 active-session 轮询线程（每 2s，免认证，常驻）
  ┌─ IDLE      ：无活跃会话，低耗等待，心跳与轮询照常
  │   观察到 session_id(created/running)
  ▼
  CLAIM       ：created 则置 running；t0=现在；重置 timeline/帧计数/UUID；
  │             启动 3s 实时状态上报
  ▼
  ASSESSING   ：摄像头采集/人脸检测/NPU 推理/记录 timeline（无时长上限，
  │             不再有 60s 限制）；每帧读轮询线程的内存快照（纯内存、
  │             无网络 IO），仅当快照可达且会话消失/被置终态时结束
  ▼
  FINISH      ：停实时上报 → 本地留底 → 一次性批量上传（同 UUID 指数退避，
  │             409/422 记录明确日志不静默）→ 打印云端结果
  └─→ 回到 IDLE，等待下一次测评（旧会话上传完成前不会开始新会话，数据不串）

解耦与可靠性：
- 心跳线程 / 会话轮询线程 / 实时状态上报线程 三者互相独立；
- 采集循环只做内存读与 NPU 推理，从不同步阻塞发 HTTP；
- 轮询网络失败时保留上次快照（reachable=False），绝不误判会话结束，
  采集继续，网络恢复后同节拍自动拿到最新状态；
- SIGTERM / Ctrl-C：测评中先收尾上传再退出，空闲则直接退出。

用法：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ./venv/bin/python run_cloud_assessment.py                # 常驻，跟随手机会话
  ./venv/bin/python run_cloud_assessment.py --duration 60  # 本地保护上限 60s（调试用）
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
from cloud.status_reporter import StatusReporter
from cloud.session_watcher import SessionWatcher

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_registered(cfg, client, log):
    """返回 token。token 已存在则验证复用，不重复注册。"""
    token = cfg.load_token()
    if token:
        client.set_token(token)
        try:
            dev = client.get_device()
            log.info("复用已保存 token，设备信息: id=%s name=%s status=%s",
                     dev.get("device_id"), dev.get("device_name"),
                     dev.get("status"))
            return token
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
    return token


def run_one_assessment(cfg, client, log, recognizer, cap, watcher, hb,
                       session_id, initial_status, cam_args, global_stop,
                       max_duration=0.0):
    """CLAIM → ASSESSING → FINISH，完成一次会话。返回 True=上传成功/duplicate。"""
    sample_fps, infer_interval = cam_args["sample_fps"], cam_args["infer_interval"]
    rotate, mirror = cam_args["rotate"], cam_args["mirror"]

    # ---------- CLAIM ----------
    if initial_status == "created":
        try:
            client.update_session_status(session_id, "running")
            log.info("会话 %s 已由设备置为 running", session_id)
        except CloudAPIError as e:
            # 可能手机已自行置 running（云端幂等）或网络抖动；不阻断采集
            log.warning("置 running 失败（继续采集）: %s", e)

    timeline = []
    video_frames = face_frames = infer_runs = 0
    t0 = time.time()
    wall_start = datetime.now()
    last_infer_t = 0.0
    cached_box = None
    last_box_ok_t = 0.0
    last_print_t = t0
    sample_period = 1.0 / max(sample_fps, 0.1)
    last_expr, last_conf, last_expr_t = None, None, 0.0
    expr_stale = max(1.0, infer_interval * 2)
    request_id = str(uuid.uuid4())

    reporter = StatusReporter(client, interval=cfg.status_interval,
                              timeout=cfg.status_timeout)
    reporter.start()
    log.info("====== 测评开始 session=%s（request_id=%s，实时上报每 %.0fs）======",
             session_id, request_id, cfg.status_interval)

    end_reason = None

    # ---------- ASSESSING ----------
    try:
        while not global_stop.is_set():
            loop_t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                log.warning("读帧失败，跳过")
                global_stop.wait(0.05)
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

            # 实时状态（每帧仅写内存，HTTP 由 reporter 线程完成）
            face_now = cached_box is not None
            if not face_now:
                reporter.update(False, False)  # 情况 A
            elif last_expr is None or (now - last_expr_t) > expr_stale:
                reporter.update(True, False)   # 情况 B

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
                timeline.append({
                    "relative_seconds": round(max(0.0, now - t0), 3),
                    "expression": ExpressionRecognizer.LABELS[idx],
                    "confidence": round(float(probs[idx]), 6),
                })
                infer_runs += 1
                last_infer_t = now
                last_expr = ExpressionRecognizer.LABELS[idx]
                last_conf = float(probs[idx])
                last_expr_t = now
                reporter.update(True, True, last_expr, last_conf)  # 情况 C

            if now - last_print_t >= 5.0:
                last_print_t = now
                cur = timeline[-1]["expression"] if timeline else "-"
                log.info("[%5.1fs] 帧=%d 人脸帧=%d 推理=%d 当前=%s 心跳ok=%d "
                         "状态上报ok=%d/失败%d 轮询ok=%d/失败%d",
                         now - t0, video_frames, face_frames, infer_runs, cur,
                         hb.beats_ok, reporter.sent_ok, reporter.sent_fail,
                         watcher.polls_ok, watcher.polls_fail)

            # 会话生命周期判定：只采信"可达且已变化"的快照，网络失败绝不断测
            sid, status, reachable = watcher.snapshot()
            if reachable and sid != session_id:
                end_reason = ("会话已结束（云端无活跃会话）" if sid is None
                              else f"会话已切换为新会话 {sid}")
                break
            if reachable and status in ("completed", "interrupted"):
                end_reason = f"会话被置为 {status}"
                break
            if max_duration > 0 and (now - t0) >= max_duration:
                end_reason = f"本地保护上限 {max_duration:.0f}s 到"
                break

            spare = sample_period - (time.time() - loop_t0)
            if spare > 0:
                global_stop.wait(spare)
    except KeyboardInterrupt:
        end_reason = "Ctrl-C 中断"
    finally:
        reporter.stop()
        reporter.join(timeout=3)

    wall_end = datetime.now()
    elapsed = time.time() - t0
    stopping = global_stop.is_set() and end_reason is None
    if end_reason is None:
        end_reason = "收到退出信号" if stopping else "未知"
    log.info("====== 测评结束 session=%s 原因=%s 时长=%.1fs 帧=%d 推理=%d ======",
             session_id, end_reason, elapsed, video_frames, infer_runs)

    # ---------- FINISH ----------
    if infer_runs == 0:
        # 不上传空数据是既有约定；会话保留在云端由手机/reaper 收尾
        log.error("本次会话没有任何有效推理（全程无人脸？），不上传空数据；"
                  "本地无留底，会话 %s 由云端收尾", session_id)
        return False

    meta = {
        "session_start": wall_start.isoformat(timespec="seconds"),
        "session_end": wall_end.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "video_frames": video_frames,
        "display_fps": round(video_frames / elapsed, 2) if elapsed > 0 else 0.0,
        "face_detected_frames": face_frames,
        "inference_runs": infer_runs,
    }
    payload = {"client_request_id": request_id, "session_meta": meta,
               "timeline": timeline}

    # 成功前不清空 timeline；本地留底（断网补传/人工核对用）
    out_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    audit_path = os.path.join(out_dir, f"cloud_upload_{wall_end.strftime('%Y%m%d_%H%M%S')}.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump({"session_id": session_id, **payload}, f,
                  ensure_ascii=False, indent=2)
    log.info("本地留底: %s（%d 条 timeline，request_id=%s）",
             audit_path, len(timeline), request_id)

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
            log.info("上传成功，云端已重算结果（会话自动收尾 completed）")
    except CloudValidationError as e:
        # 422 不重试：数据本身非法，明确记录，绝不静默丢弃
        log.error("上传被云端 422 拒绝（不重试）: %s；数据留底 %s", e, audit_path)
    except CloudAPIError as e:
        if e.status_code == 409:
            # 可能：云端尚未放开"结束后补传"或该会话已有其他结果
            log.error("上传被云端 409 拒绝（会话已结束且不允许补传/已有结果）: %s；"
                      "数据留底 %s，未静默丢失，请确认云端补传策略", e, audit_path)
        else:
            log.error("上传最终失败: %s；数据留底于 %s，网络恢复后可补传",
                      e, audit_path)

    print("\n========== 云端返回 ==========")
    print(f"session_id    : {session_id}")
    print(f"结束原因      : {end_reason}")
    print(f"request_id    : {request_id}（重试复用，幂等）")
    print(f"timeline 条数 : {len(timeline)}")
    if upload_resp is not None:
        print(json.dumps(upload_resp, ensure_ascii=False, indent=2))
        print(f"结果查询: GET {cfg.server_base_url}/api/v1/emotion/sessions/"
              f"{session_id}/result")
    else:
        print("上传失败/被拒，详见日志与本地留底文件")
    return upload_ok


def main():
    ap = argparse.ArgumentParser(description="Atlas 常驻表情测评（跟随手机会话）")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--duration", type=float, default=0,
                    help="单次测评本地保护上限秒数；0=完全跟随会话生命周期（默认）")
    ap.add_argument("--sample-fps", type=float, default=None)
    ap.add_argument("--infer-interval", type=float, default=None)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270])
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    cfg = AppConfig(args.config)
    log = setup_logging(cfg.log_file)

    sample_fps = args.sample_fps if args.sample_fps is not None else cfg.sample_fps
    infer_interval = (args.infer_interval if args.infer_interval is not None
                      else cfg.infer_interval)
    cam_args = {
        "sample_fps": sample_fps,
        "infer_interval": infer_interval,
        "rotate": args.rotate if args.rotate is not None else cfg.rotate,
        "mirror": args.mirror or cfg.mirror,
    }
    cam_idx = args.camera if args.camera is not None else cfg.camera_index
    width = args.width if args.width is not None else cfg.width
    height = args.height if args.height is not None else cfg.height

    global_stop = threading.Event()

    def _on_signal(signum, frame):
        log.info("收到信号 %s：测评中则先收尾上传再退出，空闲则直接退出", signum)
        global_stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # 1) 健康检查 + 注册/复用 token
    client = CloudClient(cfg.server_base_url, cfg.device_id,
                         timeout=cfg.http_timeout,
                         max_retries=cfg.upload_max_retries)
    try:
        log.info("云端健康检查: %s", client.health())
        ensure_registered(cfg, client, log)
    except CloudAPIError as e:
        log.error("无法连接/注册云端，终止: %s", e)
        return 1

    # 2) 心跳线程（常驻，贯穿整个进程生命周期）
    hb = HeartbeatThread(client, interval=cfg.heartbeat_interval,
                         firmware_version=cfg.firmware_version)
    hb.start()

    # 3) NPU 与摄像头只初始化一次，常驻期间反复服务多次测评
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

    # 4) 会话轮询线程（常驻，免认证，每 2s；与心跳/采集互不阻塞）
    watcher = SessionWatcher(client, interval=cfg.session_poll_interval,
                             timeout=cfg.session_poll_timeout)
    watcher.start()

    log.info("进入常驻模式：等待手机端创建测评会话（轮询 %.0fs / 心跳 %.0fs），Ctrl-C 退出",
             cfg.session_poll_interval, cfg.heartbeat_interval)

    # 5) 状态机外循环：IDLE → CLAIM/ASSESSING/FINISH → IDLE
    claimed_sid = None
    assessments = 0
    rc = 0
    try:
        while not global_stop.is_set():
            sid, status, reachable = watcher.snapshot()

            # IDLE：边沿——出现了一个"未认领"的活跃会话才开始
            if sid and sid != claimed_sid and status in ("created", "running"):
                assessments += 1
                ok = run_one_assessment(
                    cfg, client, log, recognizer, cap, watcher, hb,
                    sid, status, cam_args, global_stop,
                    max_duration=args.duration)
                # 旧会话完整收尾后才放开新会话，数据不串
                claimed_sid = sid
                log.info("第 %d 次测评收尾完成（上传%s），回到空闲等待",
                         assessments, "成功" if ok else "未成功，见日志")
                # 主动刷新一次视角，避免等下一节拍
                continue

            # 空闲低耗等待（可被信号立即唤醒）
            global_stop.wait(0.5)
    finally:
        log.info("进程退出：停止实时会话轮询与心跳（本次进程共服务 %d 次测评）",
                 assessments)
        watcher.stop()
        watcher.join(timeout=3)
        hb.stop()
        hb.join(timeout=3)
        cap.release()
    return rc


if __name__ == "__main__":
    sys.exit(main())
