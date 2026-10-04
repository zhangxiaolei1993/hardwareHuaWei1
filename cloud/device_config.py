# -*- coding: utf-8 -*-
"""设备配置与 token 持久化。

- 所有连接参数（服务器地址、device_id、心跳间隔、摄像头参数等）放在
  config/device_config.json，业务代码不硬编码。
- 注册返回的 device_token 落盘到 config/device_token.json（权限 600），
  断电/重启后直接复用，不重复注册。
"""
import json
import logging
import os
import stat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "device_config.json")


class AppConfig:
    """从 JSON 读取的设备配置；路径字段按项目根目录解析为绝对路径。"""

    def __init__(self, path=DEFAULT_CONFIG):
        self.config_path = path
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.server_base_url = cfg["server_base_url"].rstrip("/")
        self.device_id = cfg["device_id"]
        self.device_name = cfg["device_name"]
        self.device_type = cfg["device_type"]
        self.manufacturer = cfg.get("manufacturer")
        self.model = cfg.get("model")
        self.firmware_version = cfg.get("firmware_version")
        self.capabilities = cfg.get("capabilities", ["emotion"])
        self.heartbeat_interval = float(cfg.get("heartbeat_interval", 20))
        self.http_timeout = float(cfg.get("http_timeout", 8))
        self.upload_max_retries = int(cfg.get("upload_max_retries", 5))
        self.camera_index = int(cfg.get("camera_index", 0))
        self.width = int(cfg.get("width", 640))
        self.height = int(cfg.get("height", 480))
        self.rotate = int(cfg.get("rotate", 0))
        self.mirror = bool(cfg.get("mirror", False))
        self.sample_fps = float(cfg.get("sample_fps", 5.0))
        self.infer_interval = float(cfg.get("infer_interval", 0.5))
        self.token_file = self._abs(cfg.get("token_file", "config/device_token.json"))
        self.log_file = self._abs(cfg.get("log_file", "logs/cloud_device.log"))

    def _abs(self, p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    def register_payload(self):
        """注册接口请求体。"""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": self.firmware_version,
            "capabilities": self.capabilities,
        }

    # ---- token 落盘 / 读取 ----
    def load_token(self):
        """返回已保存的 token；不存在返回 None。"""
        if not os.path.exists(self.token_file):
            return None
        try:
            with open(self.token_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("device_token")
        except (ValueError, OSError):
            return None

    def save_token(self, token):
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        tmp = self.token_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"device_id": self.device_id, "device_token": token}, f,
                      ensure_ascii=False, indent=2)
        os.replace(tmp, self.token_file)
        try:
            os.chmod(self.token_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
        except OSError:
            pass


def setup_logging(log_file):
    """日志同时输出到终端和文件（UTF-8）。"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logging.getLogger("cloud_device")
