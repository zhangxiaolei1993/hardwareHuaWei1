#!/bin/bash
# net-keep: 每30秒检查默认路由，丢失则自动补回（不管谁清的）
GW=192.168.1.1
if ip link show wlan0 >/dev/null 2>&1; then
    if ! ip route show default 2>/dev/null | grep -q .; then
        ip route add default via $GW dev wlan0 2>/dev/null && logger -t net-keep "re-added default via $GW"
    fi
fi
# 顺带补 usb0 的管理IP
if ip link show usb0 >/dev/null 2>&1; then
    if ! ip -4 addr show usb0 | grep -q '192.168.137.2'; then
        ip addr add 192.168.137.2/24 dev usb0 2>/dev/null && logger -t net-keep "re-added 137.2 on usb0"
    fi
fi
