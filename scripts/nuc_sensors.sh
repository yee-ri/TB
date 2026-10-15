#!/usr/bin/env bash
set -e
TB_NUC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TB_PUBLISH_MOUNTS=false
TB_ENABLE_DEPTH=false
for TB_SENSOR_OPTION in "$@"; do
  case "$TB_SENSOR_OPTION" in
    --mounts) TB_PUBLISH_MOUNTS=true ;;
    --depth) TB_ENABLE_DEPTH=true ;;
    *) printf 'Usage: %s [--mounts] [--depth]\n' "$0" >&2; exit 1 ;;
  esac
done
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
TB_SENSOR_ARGS=()
if [ "$TB_PUBLISH_MOUNTS" = true ]; then
  python3 "$TB_NUC_ROOT/scripts/nuc_mounts.py"
  TB_SENSOR_ARGS+=(publish_measured_mounts:=true)
else
  printf 'Sensor check only: measured base-to-sensor TF is not published. After frame calibration, use --mounts.\n'
fi
if [ "$TB_ENABLE_DEPTH" = true ]; then
  TB_SENSOR_ARGS+=(enable_depth:=true)
fi
mkdir -p "$TB_NUC_ROOT/records/ros"
set -o pipefail
roslaunch "$TB_NUC_ROOT/launch/nuc_sensors.launch" "${TB_SENSOR_ARGS[@]}" 2>&1 | tee -a "$TB_NUC_ROOT/records/sensors.log"
