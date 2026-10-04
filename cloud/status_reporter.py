# -*- coding: utf-8 -*-
"""独立实时表情状态上报线程：采集期间每约 3s 上报一次"当前帧状态"。

与心跳线程完全独立、互不替代：
- 心跳 20s 一次，决定设备 online/offline；
- 实时状态 3s 一次，只表示当下人脸/表情，设备级、不依赖 session。

非阻塞设计：
- 主采集/推理循环每帧只调用 update() 写入最新状态（加锁、纯内存）；
- 所有 HTTP 在本线程内完成（短超时），绝不拖慢摄像头/NPU；
- 网络失败只记日志并跳过本拍，等下一个 3s 节拍，线程不退出；
- stop() 可被立即唤醒，采集停止即停止上报。
"""
import logging
import threading
import time

log = logging.getLogger("status_reporter")


class StatusReporter(threading.Thread):
    def __init__(self, client, interval=3.0, timeout=2.5):
        super().__init__(daemon=True, name="emotion-status")
        self.client = client
        self.interval = float(interval)
        self.timeout = float(timeout)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        # (face_detected, expression_detected, expression, confidence)
        self._state = (False, False, None, None)
        self.sent_ok = 0
        self.sent_fail = 0
        self.last_error = None

    def update(self, face_detected, expression_detected,
               expression=None, confidence=None):
        """主循环每帧调用：仅更新内存最新状态，不做任何网络 IO。"""
        with self._lock:
            self._state = (bool(face_detected), bool(expression_detected),
                           expression, confidence)

    def run(self):
        log.info("实时状态上报线程启动（间隔 %.1fs，超时 %.1fs）",
                 self.interval, self.timeout)
        while not self._stop_evt.is_set():
            with self._lock:
                face, expr_ok, expr, conf = self._state
            try:
                resp = self.client.put_emotion_status(
                    face, expr_ok, expr, conf, timeout=self.timeout)
                self.sent_ok += 1
                if expr_ok and conf is not None:
                    tag = f"人脸+表情 {expr}({float(conf):.2f})"
                elif face:
                    tag = "仅人脸(无有效表情)"
                else:
                    tag = "无人脸"
                log.info("实时状态上报成功 #%d -> %s；云端 stale=%s updated_at=%s",
                         self.sent_ok, tag,
                         resp.get("stale"), resp.get("updated_at"))
            except Exception as e:  # 任何异常只跳过本拍，绝不能让线程退出
                self.sent_fail += 1
                self.last_error = str(e)
                log.warning("实时状态上报失败 #%d（跳过本拍，不影响采集/心跳）: %s",
                            self.sent_fail, e)
            # 可被 stop() 立即唤醒的等待
            self._stop_evt.wait(self.interval)
        log.info("实时状态上报线程停止（成功 %d 次 / 失败 %d 次）",
                 self.sent_ok, self.sent_fail)

    def stop(self):
        self._stop_evt.set()
