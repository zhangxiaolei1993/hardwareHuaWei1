#!/bin/bash
# 表情测评常驻程序启动脚本（在 fish 或 bash 里直接执行均可）
# 用法: ./start_assessment.sh [后台]   带参数"bg"则后台运行
cd /home/HwHiAiUser/expression_recognition
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ "$1" = "bg" ]; then
    setsid ./venv/bin/python run_cloud_assessment.py </dev/null >/tmp/assess.out 2>&1 &
    echo "已后台启动 PID: $!  日志: tail -f /tmp/assess.out"
else
    exec ./venv/bin/python run_cloud_assessment.py
fi
