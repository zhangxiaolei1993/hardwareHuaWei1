#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端接口自测脚本（验证用，不涉及摄像头/NPU）。

覆盖任务第 6 节的：
  1/2 注册幂等：重复注册返回同一 token，created=False
  7   重复 POST：同一 (session_id, client_request_id) 第二次返回 duplicate=true
  8   非法 expression / confidence=1.5 被云端 422 拒绝
  6   result 接口可查到云端重算结果

用法（板端，先 source activate.sh）：
  python cloud_selftest.py
"""
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud.device_config import AppConfig, DEFAULT_CONFIG
from cloud.cloud_client import CloudClient, CloudValidationError


def main():
    cfg = AppConfig(DEFAULT_CONFIG)
    client = CloudClient(cfg.server_base_url, cfg.device_id,
                         timeout=cfg.http_timeout,
                         max_retries=2)

    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    # ---- 健康检查 ----
    h = client.health()
    check("health 200", h.get("status") == "ok", str(h))

    # ---- 注册幂等（重复注册返回同一 token，不产生新设备）----
    r1 = client.register(cfg.register_payload())
    tok1 = r1["device_token"]
    r2 = client.register(cfg.register_payload())
    tok2 = r2["device_token"]
    cfg.save_token(tok2)
    client.set_token(tok2)
    check("注册幂等(同一token)", tok1 == tok2,
          f"created1={r1.get('created')} created2={r2.get('created')}")

    # ---- 建会话 ----
    sess = client.create_session(session_type="emotion")
    sid = sess["session_id"]
    check("创建会话", bool(sid), f"session_id={sid}")
    client.update_session_status(sid, "running")

    # ---- 正常上传 ----
    rid = str(uuid.uuid4())
    meta = {"elapsed_seconds": 3.0, "video_frames": 15,
            "display_fps": 5.0, "face_detected_frames": 10,
            "inference_runs": 6}
    tl = [
        {"relative_seconds": 0.0, "expression": "neutral", "confidence": 0.80},
        {"relative_seconds": 0.5, "expression": "happiness", "confidence": 0.60},
        {"relative_seconds": 1.0, "expression": "happiness", "confidence": 0.70},
        {"relative_seconds": 1.5, "expression": "neutral", "confidence": 0.90},
    ]
    up1 = client.upload_emotion_data(sid, rid, meta, tl)
    check("首次上传成功", up1.get("result") is not None,
          f"duplicate={up1.get('duplicate')}")

    # ---- 重复 POST：同 session + 同 request_id ----
    up2 = client.upload_emotion_data(sid, rid, meta, tl)
    check("重复提交 duplicate=true", up2.get("duplicate") is True,
          f"resp={json.dumps(up2, ensure_ascii=False)[:200]}")

    # ---- result 接口（云端按相邻 relative_seconds 差值重算）----
    # timeline: neutral@0, happiness@0.5, happiness@1.0, neutral@1.5
    # 云端加权：neutral 0.5s(33.33%) / happiness 1.0s(66.67%) → dominant=happiness
    res = client.get_result(sid)
    rdata = res.get("result") or {}
    check("result 可查且由云端重算",
          rdata.get("dominant_expression") == "happiness"
          and abs(rdata.get("face_coverage_percent", 0) - 66.67) < 0.01,
          json.dumps(rdata, ensure_ascii=False)[:300])

    # ---- 非法 expression -> 云端 422（绕过本地校验，直打服务端验证）----
    def raw_upload(timeline_bad):
        body = {"client_request_id": str(uuid.uuid4()),
                "session_meta": meta, "timeline": timeline_bad}
        return client._request(
            "POST", f"/api/v1/emotion/sessions/{sid}/data",
            json_body=body, max_retries=0, action="非法数据测试")

    bad1 = [{"relative_seconds": 0.0, "expression": "happy",
             "confidence": 0.9}]  # 拼错：应为 happiness
    try:
        raw_upload(bad1)
        check("非法expression被云端422", False, "云端竟然接受了")
    except CloudValidationError as e:
        check("非法expression被云端422", e.status_code == 422, str(e)[:160])

    # ---- confidence 越界 -> 422 ----
    bad2 = [{"relative_seconds": 0.0, "expression": "neutral",
             "confidence": 1.5}]
    try:
        raw_upload(bad2)
        check("confidence=1.5被云端422", False, "云端竟然接受了")
    except CloudValidationError as e:
        check("confidence=1.5被云端422", e.status_code == 422, str(e)[:160])

    client.update_session_status(sid, "completed")

    print("\n========== 自测汇总 ==========")
    for name, ok, _ in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n会话 {sid} 可在 GET "
          f"{cfg.server_base_url}/api/v1/emotion/sessions/{sid}/result 查看")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
