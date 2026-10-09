#!/usr/bin/env bash
# Source this file in each operational terminal; do not change shell startup files.
_tb_nuc_root=/home/sj/ParkCars/TB-tunnel-nuc
# 연습 기본값: 외경 1820 mm / 포맥스 2 mm / 출입구 300 mm.
export TB_TUNNEL_CONFIG="${_tb_nuc_root}/codes/stepbystep/tunnel.real.yaml"
# 대회: 안내 규격 1.8 m / 출입구 300 mm. 벽 두께·치수 기준 미확인으로 지도는 아직 미완성.
# 대회 실측 지도 확인 후 전환: 바로 위 줄을 주석 처리하고 아래 줄의 #을 제거한다.
# 기존 센서를 종료하고 모든 터미널에서 다시 source한 뒤 센서부터 다시 시작한다.
# export TB_TUNNEL_CONFIG="${_tb_nuc_root}/codes/stepbystep/tunnel.competition.yaml"
for _tb_setup in /opt/ros/noetic/setup.bash /home/sj/ParkCars/devel/setup.bash "${_tb_nuc_root}/nuc_ws/devel/setup.bash"; do
    if [[ ! -f "${_tb_setup}" ]]; then
        printf 'Missing required ROS setup: %s\n' "${_tb_setup}" >&2
        return 1 2>/dev/null || exit 1
    fi
    source "${_tb_setup}" || { return 1 2>/dev/null || exit 1; }
done
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_HOSTNAME=127.0.0.1
unset ROS_IP
export ROS_LOG_DIR="${_tb_nuc_root}/records/ros"
mkdir -p "${ROS_LOG_DIR}" || { return 1 2>/dev/null || exit 1; }
unset _tb_setup _tb_nuc_root
