# 昇腾 NPU 表情识别 · 云端联动测评设备

华为 Atlas 200I DK A2（Ascend 310B4）+ USB 摄像头，人脸检测与 8 类表情识别全部跑在 NPU 上，
与云端服务联动完成测评：设备常驻运行，手机端创建/结束会话，设备自动采集、收尾并上传表情 timeline。

## 文档导航

| 文档 | 内容 | 适用对象 |
|---|---|---|
| [项目说明.md](项目说明.md) | 项目总览 + 单机统计工具使用说明 + 技术记录（踩坑/性能/历程） | 所有人 |
| [操作手册.md](操作手册.md) | 日常使用：开关机、启动、状态查看、常见问题排查、速查卡 | 测评操作人员 / 运维 |
| [硬件端部署手册.md](硬件端部署手册.md) | 从零部署板端环境、模型转换、网络自愈、验证清单 | 部署工程师 |

## 快速开始（板端已部署）

```bash
ssh root@192.168.137.2        # 密码 Mind@123（USB 通道，固定 IP）
cd /home/HwHiAiUser/expression_recognition
./start_assessment.sh         # 常驻运行；后台加 bg
```

手机端创建测评会话即自动开始，结束会话自动收尾上传。详见《操作手册.md》。

## 核心指标（Ascend 310B4，720p）

- FER+ 表情推理（NPU）：约 2.7ms
- YuNet 人脸检测（NPU，640 输入）：约 10.5ms
- 实拍整链路（检测 + 表情 + 上报）：约 20fps

## 仓库结构速览

```
├── run_cloud_assessment.py    # ★ 云端测评主程序（常驻状态机）
├── expression_recognizer.py   # 表情识别核心类（检测器调度 + NPU 推理）
├── yunet_detector.py          # NPU YuNet 人脸检测器
├── camera_util.py             # 摄像头打开工具
├── realtime_emotion_stats.py  # 单机时段统计工具（不依赖云端）
├── camera_demo.py             # 简版实时预览
├── cloud_selftest.py          # 云端接口自测
├── cloud/                     # 云端通信模块（client/心跳/轮询/状态上报/配置）
├── config/device_config.json  # 唯一配置文件
├── models/                    # ONNX 源模型（OM 在板端生成，不入库）
├── scripts/                   # net-keep 网络自愈三件套
└── start_assessment.sh        # 启动脚本（自动加载昇腾环境）
```
