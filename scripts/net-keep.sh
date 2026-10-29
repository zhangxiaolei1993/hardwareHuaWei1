#!/bin/bash
# net-keep: 每 30 秒巡检一次网络
# 1) wlan0 掉线（无 IP / 无默认路由）时通过 connman 自动重连，连续失败则重启 connman
# 2) 默认路由丢失时补回
# 3) usb0 管理口保持 up，并确保 192.168.137.2 存在

GW=192.168.1.1
FAIL_FILE=/run/net-keep.wifi-failures
LOG_FILE=/var/log/net-keep.log

log() {
    echo "$(date '+%F %T') $*" >> "$LOG_FILE"
    logger -t net-keep "$*" 2>/dev/null
}

wifi_ok() {
    ip -4 addr show wlan0 2>/dev/null | grep 'inet ' | grep -qv '169\.254\.' || return 1
    ping -I wlan0 -c 1 -W 2 "$GW" >/dev/null 2>&1
}

wifi_watchdog() {
    ip link show wlan0 >/dev/null 2>&1 || return 0
    if wifi_ok; then
        rm -f "$FAIL_FILE"
        return 0
    fi

    local n svc i
    n=$(cat "$FAIL_FILE" 2>/dev/null || echo 0)
    n=$((n + 1))
    echo "$n" > "$FAIL_FILE"

    if [ "$n" -eq 2 ]; then
        svc=$(connmanctl services 2>/dev/null | awk '$1 ~ /^\*A/ {print $3; exit}')
        if [ -n "$svc" ]; then
            log "wlan0 offline, reconnect $svc"
            for i in 1 2 3; do
                connmanctl connect "$svc" >/dev/null 2>&1
                sleep 3
                wifi_ok && break
            done
        fi
    elif [ "$n" -ge 4 ]; then
        local now last
        now=$(date +%s)
        last=$(cat /run/net-keep.last-restart 2>/dev/null || echo 0)
        if [ $((now - last)) -ge 300 ]; then
            log "wlan0 still offline, restart connman"
            echo "$now" > /run/net-keep.last-restart
            systemctl restart connman
        fi
        rm -f "$FAIL_FILE"
    fi
}

route_keep() {
    ip link show wlan0 >/dev/null 2>&1 || return 0
    ip route show default 2>/dev/null | grep -q . && return 0
    ip route add default via "$GW" dev wlan0 2>/dev/null && log "re-added default via $GW"
}

usb_keep() {
    ip link show usb0 >/dev/null 2>&1 || return 0
    ip link set usb0 up 2>/dev/null
    if ! ip -4 addr show usb0 2>/dev/null | grep -q '192.168.137.2'; then
        ip addr add 192.168.137.2/24 dev usb0 2>/dev/null && log "re-added 137.2 on usb0"
    fi
}

wifi_watchdog
route_keep
usb_keep
