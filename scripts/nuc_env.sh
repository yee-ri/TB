#!/usr/bin/env bash
# Shared sensor/direct-controller environment; optional source for trial tools.
_tb_nuc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || { return 1 2>/dev/null || exit 1; }
# This NUC's working sensor drivers are in the parent ParkCars workspace.
# Do not source the repository's historical catkin_ws build/devel artifacts.
_tb_setups=(/opt/ros/noetic/setup.bash "$(dirname -- "${_tb_nuc_root}")/devel/setup.bash")
# An explicitly prepared local compatibility overlay is optional on this PC.
if [[ -f "${_tb_nuc_root}/nuc_ws/devel/setup.bash" ]]; then
    _tb_setups+=("${_tb_nuc_root}/nuc_ws/devel/setup.bash")
fi
for _tb_setup in "${_tb_setups[@]}"; do
    if [[ ! -f "${_tb_setup}" ]]; then
        printf 'Missing required ROS setup: %s\n' "${_tb_setup}" >&2
        printf 'See %s/README_TUNNEL_TEST.md for the local workspace setup.\n' "${_tb_nuc_root}" >&2
        unset _tb_setup _tb_setups _tb_nuc_root
        return 1 2>/dev/null || exit 1
    fi
done
for _tb_setup in "${_tb_setups[@]}"; do
    source "${_tb_setup}" || { unset _tb_setup _tb_setups _tb_nuc_root; return 1 2>/dev/null || exit 1; }
done
# Official burger_noetic firmware uses the native Noetic BatteryState schema.
# Keep local turtlebot3_msgs/tf2_sensor_msgs, but prefer Noetic over the old
# sensor_msgs copy in nuc_ws for both Python imports and ROS package lookup.
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_PACKAGE_PATH="/opt/ros/noetic/share${ROS_PACKAGE_PATH:+:${ROS_PACKAGE_PATH}}"
# 연습 기본값: 외경 1820 mm / 포맥스 2 mm / 출입구 300 mm.
export TB_TUNNEL_CONFIG="${TB_TUNNEL_CONFIG:-${_tb_nuc_root}/codes/stepbystep/tunnel.real.yaml}"
# 대회: 안내 규격 1.8 m / 출입구 300 mm. 벽 두께·치수 기준 미확인으로 지도는 아직 미완성.
# 대회 실측 지도 확인 후 전환: 위 TB_TUNNEL_CONFIG 줄을 주석 처리하고 아래 줄의 #을 제거한다.
# 기존 센서를 종료하고 모든 터미널에서 다시 source한 뒤 센서부터 다시 시작한다.
# export TB_TUNNEL_CONFIG="${_tb_nuc_root}/codes/stepbystep/tunnel.competition.yaml"
# Explicit environment/ROS overrides remain available for comparisons.
export TB_TUNNEL_VERSION="${TB_TUNNEL_VERSION:-v2}"
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_HOSTNAME=127.0.0.1
unset ROS_IP
export ROS_LOG_DIR="${_tb_nuc_root}/records/ros"
mkdir -p "${ROS_LOG_DIR}" || { return 1 2>/dev/null || exit 1; }
export TB_NUC_ENV_ROOT="${_tb_nuc_root}"
unset _tb_setup _tb_setups _tb_nuc_root
