# -*- coding: utf-8 -*-
"""独立心跳线程：注册成功后启动，贯穿整个设备生命周期。

- 间隔默认 20s（云端 30s 无心跳判 offline，必须显著小于 30s）
- 立即先报一次，然后周期上报；任何异常只记录日志，绝不让线程退出
- 可通过 stop() 优雅停止；统计成功/失败次数便于自检
"""
import logging
import threading
import time

log = logging.getLogger("heartbeat")


class HeartbeatThread(threading.Thread):
    def __init__(self, client, interval=20.0, firmware_version=None):
        super().__init__(daemon=True, name="heartbeat")
        self.client = client
        self.interval = float(interval)
        self.firmware_version = firmware_version
        self._stop_evt = threading.Event()
        self.beats_ok = 0
        self.beats_fail = 0
        self.last_ok_at = None
        self.last_error = None

    def run(self):
        log.info("心跳线程启动（间隔 %.0fs，固件 %s）",
                 self.interval, self.firmware_version)
        while not self._stop_evt.is_set():
            try:
                resp = self.client.heartbeat(self.firmware_version)
                self.beats_ok += 1
                self.last_ok_at = time.time()
                log.info("心跳成功 #%d -> status=%s last_seen=%s",
                         self.beats_ok,
                         resp.get("status"),
                         resp.get("last_seen"))
            except Exception as e:  # 网络/服务异常：记录后继续，不能退出
                self.beats_fail += 1
                self.last_error = str(e)
                log.warning("心跳失败 #%d（已连续失败计数=%d）: %s",
                            self.beats_fail, self.beats_fail, e)
            # 可被 stop() 立即唤醒的等待
            self._stop_evt.wait(self.interval)
        log.info("心跳线程停止（成功 %d 次 / 失败 %d 次）",
                 self.beats_ok, self.beats_fail)

    def stop(self):
        self._stop_evt.set()

    def is_healthy(self):
        return self.beats_ok > 0 and (
            self.last_ok_at is None
            or time.time() - self.last_ok_at < self.interval * 2 + 5)
