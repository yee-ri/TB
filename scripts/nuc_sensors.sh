#!/usr/bin/env bash
set -e
TB_NUC_ROOT=/home/sj/ParkCars/TB-tunnel-nuc
source "$TB_NUC_ROOT/scripts/nuc_env.sh"
python3 - <<'PY'
import rosgraph
try:
    state = rosgraph.Master('/nuc_sensor_launch_check').getSystemState()
except Exception:
    state = [[], [], []]
publishers = dict(state[0]).get('/cmd_vel', [])
if publishers:
    raise SystemExit('Refusing sensor startup: /cmd_vel publishers already active: ' + ', '.join(publishers))
nodes = {node for entries in state for _, owners in entries for node in owners}
existing = nodes.intersection({'/turtlebot3_core', '/livox_lidar_publisher2', '/camera/realsense2_camera_manager', '/nuc_camera_mount', '/nuc_lidar_mount'})
if existing:
    raise SystemExit('Sensor/TF nodes already active: ' + ', '.join(sorted(existing)))
PY
TB_MOUNT_ARGS=()
if [ "${1:-}" = "--mounts" ]; then
  python3 "$TB_NUC_ROOT/scripts/nuc_mounts.py"
  TB_MOUNT_ARGS=(publish_measured_mounts:=true)
elif [ -n "${1:-}" ]; then
  printf 'Usage: %s [--mounts]\n' "$0" >&2
  exit 1
else
  printf 'Sensor check only: measured base-to-sensor TF is not published. After frame calibration, use --mounts.\n'
fi
mkdir -p "$TB_NUC_ROOT/records/ros"
set -o pipefail
roslaunch "$TB_NUC_ROOT/launch/nuc_sensors.launch" "${TB_MOUNT_ARGS[@]}" 2>&1 | tee -a "$TB_NUC_ROOT/records/sensors.log"
