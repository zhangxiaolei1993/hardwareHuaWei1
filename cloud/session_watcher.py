# -*- coding: utf-8 -*-
"""活跃会话轮询线程：独立于心跳与采集循环，每约 2s 查询一次 active-session。

职责（只做"读"，绝不写会话状态、不做任何业务动作）：
- GET /devices/{id}/active-session（免认证、短超时、不重试）
- 把最近一次查询结果作为内存快照提供给主状态机：
    snapshot() -> (session_id, status, reachable)
      · 正常返回：     (云端 session_id 或 None, 云端 status 或 None, True)
      · 网络/服务异常：(上一次成功的 session_id, 上一次的 status, False)

为什么失败时保留旧值：
- 断网期间手机对会话做的任何操作设备都无从知晓，绝不能把"查询失败"
  误判成"会话结束"。主状态机只在 reachable=True 且会话确实消失/变化时
  才收尾；网络异常时继续采集，等恢复后同一节拍自然拿到最新状态。

与心跳完全独立：心跳 20s 决定 online/offline；本线程 2s 只感知会话。
任何异常只计数记日志，线程永不退出。
"""
import logging
import threading
import time

log = logging.getLogger("session_watcher")

ACTIVE_STATUSES = ("created", "running")


class SessionWatcher(threading.Thread):
    def __init__(self, client, interval=2.0, timeout=2.5):
        super().__init__(daemon=True, name="session-watcher")
        self.client = client
        self.interval = float(interval)
        self.timeout = float(timeout)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        # 最近一次"成功查询"的结果（None 表示无活跃会话）
        self._session_id = None
        self._status = None
        self._reachable = False
        self._last_ok_at = 0.0
        self.polls_ok = 0
        self.polls_fail = 0
        self.last_error = None

    def run(self):
        log.info("会话轮询线程启动（间隔 %.1fs，超时 %.1fs）",
                 self.interval, self.timeout)
        while not self._stop_evt.is_set():
            try:
                resp = self.client.get_active_session(timeout=self.timeout)
                sid = resp.get("session_id")
                status = resp.get("status")
                changed = False
                with self._lock:
                    changed = (sid != self._session_id
                               or status != self._status
                               or not self._reachable)
                    self._session_id = sid
                    self._status = status
                    self._reachable = True
                    self._last_ok_at = time.time()
                    self.polls_ok += 1
                if changed:
                    if sid:
                        log.info("观察到活跃会话: session_id=%s status=%s",
                                 sid, status)
                    else:
                        log.info("当前无活跃会话（空闲等待中）")
            except Exception as e:  # 任何异常：保留上次快照，只计数，不退出
                with self._lock:
                    self._reachable = False
                    self.polls_fail += 1
                    self.last_error = str(e)
                log.warning("会话轮询失败 #%d（保留上次状态，不影响采集/心跳）: %s",
                            self.polls_fail, e)
            self._stop_evt.wait(self.interval)
        log.info("会话轮询线程停止（成功 %d 次 / 失败 %d 次）",
                 self.polls_ok, self.polls_fail)

    def snapshot(self):
        """返回 (session_id, status, reachable)。主循环每帧调用，纯内存。"""
        with self._lock:
            return self._session_id, self._status, self._reachable

    def stop(self):
        self._stop_evt.set()
