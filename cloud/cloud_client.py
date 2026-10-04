# -*- coding: utf-8 -*-
"""psychology-server 云端 HTTP 客户端（统一封装）。

职责：
- 所有云端接口调用收敛到本模块：注册 / 心跳 / 会话 / 数据上传 / 结果查询
- 统一请求头（X-Device-Id / X-Device-Token）、超时、异常处理
- 网络错误与 5xx / 408 / 429 做有限次指数退避重试
- 422（请求体不合法）立即抛出 CloudValidationError，不做重试
- 上传时由调用方传入 client_request_id；重试复用同一个 UUID，满足幂等要求
"""
import logging
import random
import time

import requests

log = logging.getLogger("cloud_client")

VALID_EXPRESSIONS = {
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
}


class CloudAPIError(Exception):
    """云端返回非预期状态码。"""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class CloudValidationError(CloudAPIError):
    """422 参数校验错误（非法 expression / confidence 越界等），不应重试。"""


# 可重试的状态码
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class CloudClient:
    def __init__(self, base_url, device_id, token=None, timeout=8, max_retries=5):
        self.base_url = base_url.rstrip("/")
        self.device_id = device_id
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.s = requests.Session()

    def set_token(self, token):
        self.token = token

    def _url(self, path):
        return self.base_url + path

    def _headers(self, auth=True):
        h = {"Content-Type": "application/json"}
        if auth:
            h["X-Device-Id"] = self.device_id
            h["X-Device-Token"] = self.token or ""
        return h

    def _request(self, method, path, auth=True, json_body=None,
                 max_retries=None, action="请求", timeout=None):
        """带重试的请求；返回解析后的 JSON dict。timeout 可覆盖默认值。"""
        retries = self.max_retries if max_retries is None else max_retries
        req_timeout = self.timeout if timeout is None else timeout
        url = self._url(path)
        last_exc = None
        for attempt in range(1, retries + 2):  # 首次 + retries 次重试
            try:
                resp = self.s.request(
                    method, url,
                    headers=self._headers(auth),
                    json=json_body,
                    timeout=req_timeout,
                )
            except requests.RequestException as e:
                last_exc = e
                if attempt > retries:
                    break
                delay = min(2 ** (attempt - 1), 16) + random.uniform(0, 0.3)
                log.warning("%s网络失败(第%d次): %s；%.1fs 后重试",
                            action, attempt, e, delay)
                time.sleep(delay)
                continue

            if resp.status_code in _RETRY_STATUS and attempt <= retries:
                delay = min(2 ** (attempt - 1), 16) + random.uniform(0, 0.3)
                log.warning("%s收到 %d(第%d次)，%.1fs 后重试",
                            action, resp.status_code, attempt, delay)
                time.sleep(delay)
                continue

            if resp.status_code == 422:
                raise CloudValidationError(
                    f"{action}被云端拒绝(422): {resp.text[:500]}",
                    status_code=422, body=_safe_json(resp))
            if not (200 <= resp.status_code < 300):
                raise CloudAPIError(
                    f"{action}失败 HTTP {resp.status_code}: {resp.text[:500]}",
                    status_code=resp.status_code, body=_safe_json(resp))
            return _safe_json(resp)

        raise CloudAPIError(f"{action}重试 {retries} 次后仍失败: {last_exc}")

    # ---------------- 业务接口 ----------------
    def health(self):
        return self._request("GET", "/api/v1/health", auth=False,
                             max_retries=1, action="健康检查")

    def register(self, payload):
        """注册（重复注册幂等，云端返回同一 token）。"""
        return self._request("POST", "/api/v1/devices/register",
                             auth=False, json_body=payload, action="设备注册")

    def get_device(self):
        return self._request("GET", f"/api/v1/devices/{self.device_id}",
                             action="查询设备")

    def heartbeat(self, firmware_version=None):
        body = {"firmware_version": firmware_version} if firmware_version else {}
        # 心跳只做 1 次重试：断网时单次调用尽快失败，等下一个 20s 节拍再报，
        # 避免长重试占住心跳线程（数据上传则使用独立的多重试策略）
        return self._request(
            "POST", f"/api/v1/devices/{self.device_id}/heartbeat",
            json_body=body, max_retries=1, action="心跳")

    def create_session(self, user_id=None, session_type="emotion"):
        return self._request("POST", "/api/v1/sessions", json_body={
            "device_id": self.device_id,
            "user_id": user_id,
            "session_type": session_type,
        }, action="创建会话")

    def update_session_status(self, session_id, status):
        return self._request(
            "POST", f"/api/v1/sessions/{session_id}/status",
            json_body={"status": status}, action=f"会话状态->{status}")

    def upload_emotion_data(self, session_id, client_request_id,
                            session_meta, timeline, max_retries=None):
        """批量上传一次测评的 timeline。同一次测评的网络重试复用同一 UUID。

        云端对 (session_id, client_request_id) 幂等；重复提交返回 duplicate=true。
        返回体示例: {session_id, session_status, duplicate, result:{...}}
        """
        validate_timeline(timeline)
        body = {
            "client_request_id": client_request_id,
            "session_meta": session_meta,
            "timeline": timeline,
        }
        return self._request(
            "POST", f"/api/v1/emotion/sessions/{session_id}/data",
            json_body=body, max_retries=max_retries, action="上传测评数据")

    def get_result(self, session_id):
        return self._request(
            "GET", f"/api/v1/emotion/sessions/{session_id}/result",
            action="查询测评结果")

    # ---------------- 实时表情状态（设备级，约 3s 一次，不重试） ----------------
    def put_emotion_status(self, face_detected, expression_detected,
                           current_expression=None, confidence=None,
                           timeout=None):
        """上报当前实时表情状态 PUT /emotion/devices/{id}/status。

        - 情况 A：无人脸 -> (False, False, None, None)
        - 情况 B：有人脸但暂无有效表情 -> (True, False, None, None)
        - 情况 C：有效识别 -> (True, True, 8类标签, 0~1)
        实时信号不做重试（max_retries=0）：失败由上报线程跳过本拍，等下一节拍。
        """
        face_detected = bool(face_detected)
        expression_detected = bool(expression_detected)
        if expression_detected:
            if current_expression not in VALID_EXPRESSIONS:
                raise ValueError(
                    f"current_expression 非法: {current_expression!r}，"
                    f"必须是 8 类之一: {sorted(VALID_EXPRESSIONS)}")
            if (not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not (0.0 <= float(confidence) <= 1.0)):
                raise ValueError(f"confidence 越界: {confidence!r}（应为 0~1）")
            body = {
                "face_detected": True,
                "expression_detected": True,
                "current_expression": current_expression,
                "confidence": round(float(confidence), 4),
            }
        else:
            # 云端规则：expression_detected=false 时即使本地带了值也会被清空
            body = {
                "face_detected": face_detected,
                "expression_detected": False,
                "current_expression": None,
                "confidence": None,
            }
        return self._request(
            "PUT", f"/api/v1/emotion/devices/{self.device_id}/status",
            json_body=body, max_retries=0,
            timeout=timeout, action="实时状态上报")

    def get_emotion_status(self):
        return self._request(
            "GET", f"/api/v1/emotion/devices/{self.device_id}/status",
            max_retries=1, action="查询实时状态")


def validate_timeline(timeline):
    """上传前本地校验：字段齐全、表情合法、置信度在 0~1、时间非负。"""
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("timeline 必须是非空 list")
    for i, item in enumerate(timeline):
        if not isinstance(item, dict):
            raise ValueError(f"timeline[{i}] 不是对象: {item!r}")
        expr = item.get("expression")
        if expr not in VALID_EXPRESSIONS:
            raise ValueError(
                f"timeline[{i}] expression 非法: {expr!r}，"
                f"必须是 8 类之一: {sorted(VALID_EXPRESSIONS)}")
        conf = item.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError(f"timeline[{i}] confidence 越界: {conf!r}（应为 0~1）")
        t = item.get("relative_seconds")
        if not isinstance(t, (int, float)) or t < 0:
            raise ValueError(f"timeline[{i}] relative_seconds 非法: {t!r}（应非负）")


def _safe_json(resp):
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text[:1000]}
