#!/usr/bin/env bash
# A display-only viewer: does not start sensors, a controller, or a ROS master.
set -e
_tb_view_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${DISPLAY:-}" ]]; then
    printf '%s\n' 'DISPLAY가 없습니다. Windows의 X 서버를 켜고 X11 forwarding을 사용한 SSH 세션에서 실행하세요.' >&2
    printf '%s\n' '일반 SSH 터미널에서는 RViz 창이 나타나지 않습니다. README_TUNNEL_TEST.md의 원격 보기 안내를 확인하세요.' >&2
    exit 1
fi

if [[ -n "${SSH_CONNECTION:-}" && ( "${DISPLAY}" == :0 || "${DISPLAY}" == :0.0 ) ]]; then
    printf '%s\n' 'DISPLAY=:0은 NUC의 로컬 화면입니다. Windows에서 X11 forwarding을 켠 새 SSH 세션으로 접속하세요.' >&2
    printf '%s\n' '새 연결에서 echo "$DISPLAY"가 localhost:10.0 같은 값인지 확인하세요. DISPLAY를 :0으로 덮어쓰지 마세요.' >&2
    exit 1
fi

source "${_tb_view_root}/scripts/nuc_env.sh"

# Keep DISPLAY and XAUTHORITY exactly as SSH supplied them. Software rendering
# helps forwarded X11 sessions; explicitly supplied user choices take priority.
if [[ -n "${SSH_CONNECTION:-}" && "${DISPLAY}" =~ ^[^:]+:([0-9]+)(\.[0-9]+)?$ ]] && (( ${BASH_REMATCH[1]} >= 10 )); then
    export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
    export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
fi

exec rviz -d "${_tb_view_root}/rviz/tunnel_sensors.rviz" __name:=tb_sensor_view "$@"
