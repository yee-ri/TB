#!/usr/bin/env python3
"""Explicit user-operated recording, gated tunnel start, and ordered stopping.

Creating/checking this program does not launch a controller or publish commands.
Use `source /home/sj/ParkCars/TB-tunnel-nuc/scripts/nuc_env.sh` first.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'records'
DEFAULT_CONFIG = Path(os.environ.get('TB_TUNNEL_CONFIG', str(ROOT / 'codes/stepbystep/tunnel.real.yaml')))
POINTER = RECORDS / 'current_trial.txt'
CONTROLLER = '/tb_controller'
SCOPE = ['codes/stepbystep/try_maze.py', 'codes/stepbystep/tunnel',
         'codes/stepbystep/tunnel.yaml', 'codes/stepbystep/tunnel.real.yaml',
         'codes/stepbystep/tunnel.competition.yaml', 'codes/stepbystep/maps/tunnel_practice_1820.geometry.json',
         'scripts', 'launch/nuc_sensors.launch', 'NUC_MEASUREMENTS.md',
         'NUC_MEASUREMENT_EVIDENCE.json', 'README.md', 'README_TUNNEL_TEST.md', 'DOCKER_NOETIC.md',
         'NUC_RUNBOOK.md', 'images/lturn1.png', 'images/rturn1.png']


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a measured finite number')
    if positive and value <= 0:
        raise ValueError(name + ' must be positive')
    return value


def vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError('{} must contain {} measured values'.format(name, length))
    return [number(item, name) for item in value]


def rectangle(value, name):
    xmin, ymin, xmax, ymax = vector(value, 4, name)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError(name + ' must be [xmin,ymin,xmax,ymax] with increasing bounds')
    return xmin, ymin, xmax, ymax


def read_config(path):
    import yaml
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('YAML root must be a mapping')
    return config


def check_config(path):
    import cv2
    import numpy as np
    config = read_config(path)
    def check_finite(value, name):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number(value, name)
        elif isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, name + '.' + key)
        elif isinstance(value, list):
            for item in value:
                check_finite(item, name)
    for section in ('goal', 'exit_connector', 'footprint', 'sensors', 'grid', 'entry', 'planner', 'control'):
        check_finite(config[section], section)
    for name in ('front', 'rear', 'half_width'):
        number(config['footprint'][name], 'footprint.' + name, positive=True)
    number(config['footprint']['padding'], 'footprint.padding')
    if config['footprint']['padding'] < 0:
        raise ValueError('footprint.padding cannot be negative')
    goals = [vector(config['goal'][key], 3, 'goal.' + key)
             for key in ('pose', 'planning_pose')]
    bounds = rectangle(config['grid']['bounds'], 'grid.bounds')
    keep_in = config['planner']['keep_in_rectangles']
    if not isinstance(keep_in, list) or not keep_in:
        raise ValueError('planner.keep_in_rectangles needs measured nonempty regions')
    regions = [rectangle(region, 'planner.keep_in_rectangles') for region in keep_in]
    for region in regions:
        if region[0] < bounds[0] or region[1] < bounds[1] or region[2] > bounds[2] or region[3] > bounds[3]:
            raise ValueError('keep-in regions must lie inside grid.bounds')
    for x, y in [(0., 0.)] + [(goal[0], goal[1]) for goal in goals]:
        if not any(xmin <= x <= xmax and ymin <= y <= ymax for xmin, ymin, xmax, ymax in regions):
            raise ValueError('the activation origin and both goals must lie in keep-in regions')
    for key in ('distance', 'obstacle_clearance'):
        number(config['entry'][key], 'entry.' + key, positive=True)
    number(config['grid']['resolution'], 'grid.resolution', positive=True)
    if config['grid']['resolution'] != 0.02:
        raise ValueError('first physical trial requires the unchanged normal 0.02m map resolution')
    static_map = config['grid']['static_map']
    origin = vector(static_map['origin'], 2, 'grid.static_map.origin')
    turns = number(static_map['quarter_turns'], 'grid.static_map.quarter_turns')
    if int(turns) != turns:
        raise ValueError('grid.static_map.quarter_turns must be an integer')
    if not isinstance(static_map['image'], str) or not static_map['image'].strip():
        raise ValueError('grid.static_map.image must name the surveyed physical wall map')
    map_path = (path.parent / static_map['image']).resolve()
    if not map_path.is_file() or map_path.suffix.lower() != '.pgm':
        raise ValueError('the surveyed physical wall map must be an existing PGM file')
    gazebo_map = ROOT / 'codes/stepbystep/maps/autorace_gazebo.pgm'
    if map_path == gazebo_map.resolve() or digest(map_path) == digest(gazebo_map):
        raise ValueError('the original Gazebo wall map cannot be used for a physical layout')
    pixels = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if pixels is None or not np.any(pixels == 0) or not np.any(pixels != 0):
        raise ValueError('physical wall map must contain readable black walls and free cells')
    rotated = np.rot90(np.flipud(pixels), int(turns))
    height, width = rotated.shape
    upper = [origin[0] + width * 0.02, origin[1] + height * 0.02]
    if bounds[0] < origin[0] - 1e-8 or bounds[1] < origin[1] - 1e-8 or bounds[2] > upper[0] + 1e-8 or bounds[3] > upper[1] + 1e-8:
        raise ValueError('rotated physical map extent must cover grid.bounds')
    if config.get('profiles') != {'normal': {}}:
        raise ValueError('physical trial supports only profiles: {normal: {}}')
    if config['sensors']['base_frame'] != 'base_footprint':
        raise ValueError('this physical launch and diagnostics require sensors.base_frame=base_footprint')
    # Speed limits are configured in YAML, without a second numeric ceiling here.
    number(config['control']['cruise_velocity'], 'control.cruise_velocity', positive=True)
    number(config['entry']['cruise_velocity'], 'entry.cruise_velocity', positive=True)
    # Validate runtime constructor arguments without any ROS node or publisher.
    sys.path.insert(0, str(ROOT / 'codes/stepbystep'))
    from tunnel.mission import TunnelMission, load_config
    from tunnel.planner import OccupancyGrid, Pose2D
    mission = TunnelMission(load_config(path, 'normal'))
    # Use the runtime's exact black-cell map semantics and padded rectangle.
    grid = OccupancyGrid((rotated == 0).astype(np.int8) * 100,
                         config['grid']['resolution'], origin[0], origin[1],
                         occupied_threshold=config['grid']['static_occupied_threshold'])
    for pose, label in [(Pose2D(0., 0., 0.), 'activation origin')] + [
            (Pose2D(goal[0], goal[1], math.radians(goal[2])), label)
            for goal, label in zip(goals, ('outside exit', 'internal goal'))]:
        if not mission.planner.pose_is_collision_free(grid, pose):
            raise ValueError('Padded physical footprint collides with static walls/bounds at ' + label)
    if not mission.planner.primitive_is_collision_free(grid, Pose2D(0., 0., 0.),
                                                       0., config['entry']['distance']):
        raise ValueError('Configured straight entry sweep collides with measured static walls/bounds')
    mission.close()
    return config, map_path


def require_environment():
    if os.environ.get('ROS_MASTER_URI') != 'http://127.0.0.1:11311' or os.environ.get('ROS_HOSTNAME') != '127.0.0.1' or os.environ.get('ROS_IP'):
        raise RuntimeError('Source scripts/nuc_env.sh first; this CLI uses only the NUC local master')


def graph():
    import rosgraph
    master = rosgraph.Master('/tb_trial_operator')
    pubs, subs, _ = master.getSystemState()
    return {'published_topics': dict(master.getPublishedTopics('/')),
            'topic_types': dict(master.getTopicTypes()),
            'publishers': dict(pubs), 'subscribers': dict(subs)}


def current_trial():
    if not POINTER.is_file():
        raise RuntimeError('No current trial; run record first')
    trial = Path(POINTER.read_text().strip()).resolve()
    if trial.parent != RECORDS.resolve() or not (trial / 'trial.json').is_file():
        raise RuntimeError('Invalid current trial pointer')
    return trial, json.loads((trial / 'trial.json').read_text())


def process_identity(pid):
    try:
        raw = Path('/proc/{}/stat'.format(pid)).read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_process(metadata, key, marker):
    expected = metadata.get(key)
    if not expected or process_identity(expected['pid']) != expected:
        return None
    command = Path('/proc/{}/cmdline'.format(expected['pid'])).read_bytes().split(b'\0')
    if marker.encode() not in command:
        raise RuntimeError('PID command identity mismatch; refusing to signal it')
    return expected['pid']


def can_stop_record():
    state = graph()
    publishers = state['publishers'].get('/cmd_vel', [])
    if publishers:
        raise RuntimeError('Stop controller first; /cmd_vel publishers remain: ' + ', '.join(publishers))
    if any(CONTROLLER in nodes for nodes in state['subscribers'].values()):
        raise RuntimeError('Stop TB controller first; it is still registered')


def signal_group(pid, timeout=20):
    if os.getpgid(pid) != pid:
        raise RuntimeError('Refusing to signal a process without its own process group')
    os.killpg(pid, signal.SIGINT)
    deadline = time.monotonic() + timeout
    while process_identity(pid) is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_identity(pid) is not None:
        raise RuntimeError('Process did not exit normally; inspect its log and use hardware stop')


def interrupt_foreground(_signal, _frame):
    raise KeyboardInterrupt()


def snapshot(trial, config_path):
    config = read_config(config_path)
    shutil.copy2(config_path, trial / 'tunnel.yaml')
    static_map = config.get('grid', {}).get('static_map')
    if static_map and isinstance(static_map.get('image'), str):
        map_path = (config_path.parent / static_map['image']).resolve()
        if map_path.is_file():
            shutil.copy2(map_path, trial / 'walls.pgm')
    geometry = config.get('localization', {}).get('geometry')
    geometry_path = (config_path.parent / geometry).resolve() if geometry else None
    if geometry_path is not None:
        shutil.copy2(geometry_path, trial / 'wall_geometry.json')
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD', '--'] + SCOPE, text=True)
    (trial / 'source.diff').write_text(diff, encoding='utf-8')
    hashes = {}
    source_root = trial / 'source'
    for relative in SCOPE:
        path = ROOT / relative
        candidates = path.rglob('*') if path.is_dir() else [path]
        for source in candidates:
            if source.is_file() and '__pycache__' not in source.parts and source.suffix in ('.py', '.yaml', '.launch', '.sh', '.md', '.json', '.png'):
                hashes[str(source.relative_to(ROOT))] = digest(source)
                destination = source_root / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    driver_config = Path('/home/sj/ParkCars/src/livox_ros_driver2/config/MID360_config.json')
    if driver_config.is_file():
        destination = source_root / 'driverconfig/MID360_config.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(driver_config, destination)
    dependencies = {}
    for name, repository, scoped_path in (
            ('geometry2', ROOT / 'deps/geometry2', 'tf2_sensor_msgs'),
            ('common_msgs', ROOT / 'deps/common_msgs', 'sensor_msgs'),
            ('turtlebot3_msgs', Path('/home/sj/Desktop/ARAICodes/catkin_ws/src/turtlebot3_msgs'), '.'),
            ('livox_ros_driver2', Path('/home/sj/ParkCars/src/livox_ros_driver2'), 'config/MID360_config.json')):
        if not repository.is_dir():
            continue
        commit = subprocess.run(['git', '-C', str(repository), 'rev-parse', 'HEAD'],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        delta = subprocess.run(['git', '-C', str(repository), 'diff', 'HEAD', '--', scoped_path],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        dependencies[name] = {'path': str(repository), 'commit': commit.stdout.strip() if commit.returncode == 0 else None}
        destination = source_root / 'dependencies' / name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'scoped.diff').write_text(delta.stdout, encoding='utf-8')
        if name == 'common_msgs':
            dependencies[name]['local_change_purpose'] = 'Host BatteryState definition matches deployed firmware MD5; firmware itself was not modified.'
        if name in ('geometry2', 'common_msgs', 'turtlebot3_msgs'):
            selected = repository / scoped_path
            for source in selected.rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and '.git' not in source.parts and source.suffix in ('.py', '.cpp', '.h', '.hpp', '.xml', '.msg', '.srv', '.txt', '.cmake'):
                    target = destination / source.relative_to(repository)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    save_json(trial / 'source_sha256.json', hashes)
    save_json(trial / 'dependencies.json', dependencies)
    return {'git_commit': revision, 'config_path': str(config_path),
            'config_sha256': digest(config_path), 'source_digest_file': 'source_sha256.json',
            'wall_map_recorded': (trial / 'walls.pgm').is_file(),
            'wall_map_sha256': digest(trial / 'walls.pgm') if (trial / 'walls.pgm').is_file() else None,
            'wall_geometry_sha256': digest(geometry_path) if geometry_path is not None else None,
            'sensor_ros_log_directory': os.environ.get('ROS_LOG_DIR'),
            'sensor_log_offset_bytes': (RECORDS / 'sensors.log').stat().st_size if (RECORDS / 'sensors.log').is_file() else 0,
            'firmware_source_status': 'Installed firmware source/version, RC100 behavior, watchdog, and motor torque effect remain unverified unless separately evidenced.'}


def save_sensor_log(trial, metadata):
    source = RECORDS / 'sensors.log'
    if source.is_file():
        with source.open('rb') as stream, (trial / 'sensors.log').open('wb') as output:
            stream.seek(min(metadata.get('sensor_log_offset_bytes', 0), source.stat().st_size))
            shutil.copyfileobj(stream, output)


def record(config_path):
    if POINTER.exists():
        previous, old = current_trial()
        if owned_process(old, 'rosbag_process', 'record'):
            raise RuntimeError('Recording already active in ' + str(previous))
        if old.get('recording_active'):
            raise RuntimeError('Previous recording state is stale; run stop-record to inspect/finalize it first')
    state = graph()
    if state['publishers'].get('/cmd_vel'):
        raise RuntimeError('Start recording before driving; active /cmd_vel publishers already exist')
    actual = state['published_topics']
    required = ['/camera/color/image_raw', '/livox/lidar', '/odom', '/tf', '/tf_static', '/rosout']
    missing = [topic for topic in required if topic not in actual]
    if missing:
        raise RuntimeError('Required recording inputs are not published: ' + ', '.join(missing))
    if actual['/livox/lidar'] != 'sensor_msgs/PointCloud2':
        raise RuntimeError('/livox/lidar must be PointCloud2, not Livox CustomMsg')
    topics = {'/cmd_vel', '/motor_power', '/cmd_vel_rc100', '/tf', '/tf_static', '/rosout', '/rosout_agg', '/step', '/odom', '/livox/lidar'}
    topics.update(topic for topic, kind in actual.items()
                  if (topic.startswith('/camera/') and kind in ('sensor_msgs/Image', 'sensor_msgs/CameraInfo'))
                  or (topic.startswith('/livox/') and kind == 'sensor_msgs/Imu')
                  or topic in ('/version_info', '/firmware_version', '/sensor_state', '/battery_state', '/imu'))
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f_KST')
    trial = RECORDS / stamp
    trial.mkdir(parents=True)
    metadata = snapshot(trial, config_path)
    save_json(trial / 'topics.json', {'recorded_topics': sorted(topics), 'graph_before': state})
    metadata.update({'trial': str(trial), 'created_kst': stamp, 'recording_active': True,
                     'controller_status': 'not_started', 'recorded_topics': sorted(topics)})
    command = ['rosbag', 'record', '__name:=tb_trial_recorder', '--lz4', '-O', str(trial / 'sensors.bag')] + sorted(topics)
    logfile = (trial / 'rosbag.log').open('w', encoding='utf-8')
    process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True,
                               env=dict(os.environ, ROS_LOG_DIR=str(trial / 'ros')))
    metadata['rosbag_process'] = process_identity(process.pid)
    metadata['rosbag_command'] = command
    save_json(trial / 'trial.json', metadata)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    POINTER.write_text(str(trial) + '\n')
    print('Recording: ' + str(trial), flush=True)
    print('Start tunnel in another sourced terminal. Stop controller before stop-record. Rosbag log: ' + str(trial / 'rosbag.log'), flush=True)
    stop_requested = threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
    try:
        while process.poll() is None:
            if stop_requested.is_set():
                stop_requested.clear()
                try:
                    can_stop_record()
                    os.killpg(process.pid, signal.SIGINT)
                except Exception as error:
                    print(str(error) + '; recording continues. Run stop in another terminal.', file=sys.stderr, flush=True)
            time.sleep(.1)
        process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        logfile.close()
    metadata = json.loads((trial / 'trial.json').read_text())
    active_files = list(trial.glob('*.bag.active'))
    metadata['recording_active'] = False
    metadata['recording_exit_code'] = process.returncode
    metadata['bag_finalized'] = (trial / 'sensors.bag').is_file() and not active_files
    save_sensor_log(trial, metadata)
    save_json(trial / 'trial.json', metadata)
    if not metadata['bag_finalized']:
        raise RuntimeError('Bag did not finalize normally; inspect rosbag.log and .bag.active')
    print('Bag finalized: ' + str(trial / 'sensors.bag'))


def check_remote(trial):
    import rospy
    from geometry_msgs.msg import Twist
    rospy.init_node('tb_start_guard', anonymous=True, disable_signals=True, disable_rosout=True)
    messages = []
    def callback(message):
        messages.append([message.linear.x, message.linear.y, message.linear.z,
                         message.angular.x, message.angular.y, message.angular.z])
    subscriber = rospy.Subscriber('/cmd_vel_rc100', Twist, callback, queue_size=10)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        time.sleep(.05)
    subscriber.unregister()
    rospy.signal_shutdown('Read-only remote preflight complete')
    nonzero = any(any(not math.isfinite(value) or abs(value) > 1e-9 for value in message) for message in messages)
    save_json(trial / 'remote_prestart.json', {'topic': '/cmd_vel_rc100', 'duration_seconds': 3,
              'messages_received': len(messages), 'nonzero_or_invalid_input': nonzero,
              'disabled_state_basis': 'User confirmed physical preparation complete on 2026-10-10; this check only observes current remote input.'})
    if nonzero:
        raise RuntimeError('Nonzero/invalid RC100 remote input observed; disable the firmware remote input before trial')


def start(config_path, start_step=9):
    if type(start_step) is not int or start_step not in (8, 9):
        raise ValueError('Physical trial start_step must be 8 or 9')
    config, map_path = check_config(config_path)
    trial, metadata = current_trial()
    if not metadata.get('recording_active') or not owned_process(metadata, 'rosbag_process', 'record'):
        raise RuntimeError('An active owned rosbag recording is required before tunnel start')
    if metadata['config_sha256'] != digest(config_path):
        raise RuntimeError('YAML changed after recording began; end this recording and create a new trial')
    if metadata.get('wall_map_sha256') != digest(map_path):
        raise RuntimeError('Wall map changed after recording began; create a new trial')
    geometry = config.get('localization', {}).get('geometry')
    if geometry and metadata.get('wall_geometry_sha256') != digest((config_path.parent / geometry).resolve()):
        raise RuntimeError('Alignment wall geometry changed after recording began; create a new trial')
    hashes = json.loads((trial / 'source_sha256.json').read_text())
    for relative, expected in hashes.items():
        if relative == 'codes/stepbystep/try_maze.py' or relative.startswith('codes/stepbystep/tunnel/'):
            if digest(ROOT / relative) != expected:
                raise RuntimeError('Runtime source changed after recording began; create a new trial')
    check_remote(trial)
    print('Checking fresh real sensor data and acquisition-time TF before controller start...', flush=True)
    diagnostic = [sys.executable, str(ROOT / 'scripts/nuc_diagnostics.py'), '--duration', '6',
                  '--output', str(trial / 'prestart_diagnostics.json')]
    with (trial / 'prestart_diagnostics.log').open('w') as log:
        completed = subprocess.run(diagnostic, stdout=log, stderr=subprocess.STDOUT, timeout=20)
    if completed.returncode != 0:
        raise RuntimeError('Sensor/TF check failed: ' + str(trial / 'prestart_diagnostics.json'))
    report = json.loads((trial / 'prestart_diagnostics.json').read_text())
    odom = report['topics']['/odom']
    limit = config['control']['safety_reaction_time']
    if odom['age_at_receipt_seconds']['max'] > limit or odom['age_of_last_message_now_seconds'] > limit:
        raise RuntimeError('Actual odometry age exceeds the unchanged control safety_reaction_time; inspect prestart diagnostics')
    state = graph()
    if state['publishers'].get('/cmd_vel'):
        raise RuntimeError('Existing /cmd_vel publisher blocks start: ' + ', '.join(state['publishers']['/cmd_vel']))
    if '/turtlebot3_core' not in state['subscribers'].get('/cmd_vel', []):
        raise RuntimeError('Expected real /turtlebot3_core drivetrain subscriber is absent')
    if not owned_process(metadata, 'rosbag_process', 'record'):
        raise RuntimeError('Recording stopped during diagnostics; tunnel start refused')
    command = [sys.executable, str(ROOT / 'codes/stepbystep/try_maze.py'),
               '__name:=tb_controller', '_start_step:=' + str(start_step), '_tunnel_profile:=normal',
               '_enable_motor_torque:=true',
               '_tunnel_config:=' + str(config_path), '_lidar_type:=pointcloud2', '_lidar_topic:=/livox/lidar']
    metadata['controller_command'] = command
    metadata['start_step'] = start_step
    metadata['start_condition'] = 'gate_already_open' if start_step == 8 else 'tunnel_entry'
    metadata['lane_controller'] = 'upstream_speedup_e0b970f'
    metadata['controller_status'] = 'starting'
    start_message = 'Controller /tb_controller, normal step{}. Completion continues into step11 lane driving.'.format(start_step)
    start_message += ' Lane tracking: upstream speedup e0b970f (default 0.1045 m/s).'
    if start_step == 8:
        start_message += ' Gate must already be open; step8 does not check the gate.'
    previous_term = signal.signal(signal.SIGTERM, interrupt_foreground)
    with (trial / 'controller.log').open('w') as logfile:
        logfile.write(start_message + '\n')
        logfile.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True, text=True, bufsize=1,
                                   env=dict(os.environ, PYTHONUNBUFFERED='1', ROS_LOG_DIR=str(trial / 'ros')))
        metadata['controller_process'] = process_identity(process.pid)
        metadata['controller_status'] = 'running'
        save_json(trial / 'trial.json', metadata)
        print(start_message, flush=True)
        try:
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end='', flush=True)
            process.wait()
        except KeyboardInterrupt:
            signal_group(process.pid, timeout=10)
            process.wait(timeout=2)
        finally:
            if process.poll() is None:
                signal_group(process.pid, timeout=10)
                process.wait(timeout=2)
            signal.signal(signal.SIGTERM, previous_term)
    metadata = json.loads((trial / 'trial.json').read_text())
    metadata['controller_status'] = 'exited'
    metadata['controller_exit_code'] = process.returncode
    save_json(trial / 'trial.json', metadata)
    print('Controller exited. Keep the PC on: run stop, observe that the robot has stopped, then run stop-record. Power off the PC only after Bag finalized.')


def request_torque_disable(state):
    """Use the existing OpenCR input; feedback confirms its reported state only."""
    if state['topic_types'].get('/motor_power') != 'std_msgs/Bool' or '/turtlebot3_core' not in state['subscribers'].get('/motor_power', []):
        raise RuntimeError('Existing /turtlebot3_core /motor_power Bool subscriber unavailable; torque disable was NOT requested')

    import rospy
    from std_msgs.msg import Bool
    from turtlebot3_msgs.msg import SensorState

    rospy.init_node('tb_operator_stop', anonymous=True, disable_signals=True)
    response = threading.Event()
    request_time = [None]
    feedback_available = (
        state['topic_types'].get('/sensor_state') == 'turtlebot3_msgs/SensorState'
        and '/turtlebot3_core' in state['publishers'].get('/sensor_state', []))

    def reported_state(message):
        if request_time[0] is None or (getattr(message, '_connection_header', None) or {}).get('callerid') != '/turtlebot3_core':
            return
        stamp = message.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        # Reject old queued reports and stale/future-dated sensor data.
        if math.isfinite(stamp) and stamp > 0 and stamp >= request_time[0] and -.05 <= age <= .5:
            if message.torque:
                response.clear()
            else:
                response.set()

    subscriber = None
    publisher = None
    try:
        if feedback_available:
            subscriber = rospy.Subscriber('/sensor_state', SensorState, reported_state, queue_size=10)
        publisher = rospy.Publisher('/motor_power', Bool, queue_size=1, latch=False)
        deadline = time.monotonic() + 2
        while publisher.get_num_connections() == 0 and time.monotonic() < deadline:
            time.sleep(.05)
        if publisher.get_num_connections() == 0:
            raise RuntimeError('Motor-power subscriber did not connect; torque disable was NOT requested')
        request_time[0] = rospy.Time.now().to_sec()
        publisher.publish(Bool(data=False))
        # Allow the existing serial subscriber to receive the request even if
        # telemetry is unavailable. A publish call alone is not confirmation.
        time.sleep(.2)
        if not feedback_available:
            raise RuntimeError('/motor_power=false requested, but /turtlebot3_core SensorState feedback is unavailable; software stop is unconfirmed')
        if not response.wait(3):
            raise RuntimeError('/motor_power=false requested, but no fresh torque=false report from /turtlebot3_core within 3 seconds; software stop is unconfirmed')
    finally:
        if publisher is not None:
            publisher.unregister()
        if subscriber is not None:
            subscriber.unregister()
        rospy.signal_shutdown('Operator torque-disable request finished')
    print('OpenCR reports torque=false after the stop request. This confirms reported state, not physical braking or motor-power isolation.')


def stop():
    import rosnode
    state = graph()
    nodes = {node for group in state['publishers'].values() for node in group}
    nodes.update(node for group in state['subscribers'].values() for node in group)
    if CONTROLLER in nodes:
        succeeded, failed = rosnode.kill_nodes([CONTROLLER])
        if failed or CONTROLLER not in succeeded:
            raise RuntimeError('Controller shutdown failed; software stop is unconfirmed')
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = graph()
        if CONTROLLER not in {node for group in state['publishers'].values() for node in group} and CONTROLLER not in {node for group in state['subscribers'].values() for node in group}:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Controller remains registered; software stop is unconfirmed')
    if state['publishers'].get('/cmd_vel'):
        raise RuntimeError('Other /cmd_vel publishers remain; software stop is unconfirmed: ' + ', '.join(state['publishers']['/cmd_vel']))
    request_torque_disable(state)
    print('Controller stopped. Keep the PC on, observe that the robot has stopped, then run stop-record. Power off the PC only after Bag finalized.')


def stop_record():
    can_stop_record()
    trial, metadata = current_trial()
    pid = owned_process(metadata, 'rosbag_process', 'record')
    if pid:
        signal_group(pid)
    deadline = time.monotonic() + 5
    while list(trial.glob('*.bag.active')) and time.monotonic() < deadline:
        time.sleep(.1)
    finalized = (trial / 'sensors.bag').is_file() and not list(trial.glob('*.bag.active'))
    metadata = json.loads((trial / 'trial.json').read_text())
    metadata['recording_active'] = False
    metadata['bag_finalized'] = finalized
    save_sensor_log(trial, metadata)
    save_json(trial / 'trial.json', metadata)
    if not finalized:
        raise RuntimeError('No finalized bag; preserve .bag.active and inspect ' + str(trial / 'rosbag.log'))
    print('Bag finalized: ' + str(trial / 'sensors.bag'))
    print('Once the robot is physically stopped, stop sensors with Ctrl+C; the PC can then be shut down.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('record', 'start', 'start8', 'stop', 'stop-record', 'check-config', 'check-layout'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    path = args.config.expanduser().resolve()
    operation_lock = None
    try:
        if args.command in ('check-config', 'check-layout'):
            check_config(path)
            print('Static layout/footprint checks passed: ' + str(path))
            print('Live sensor/TF, remote input, recording and publisher checks run when start is requested.')
            return 0
        require_environment()
        if args.command in ('record', 'start', 'start8'):
            RECORDS.mkdir(parents=True, exist_ok=True)
            lock_name = 'record' if args.command == 'record' else 'start'
            operation_lock = (RECORDS / (lock_name + '.lock')).open('a')
            try:
                fcntl.flock(operation_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another ' + lock_name + ' command is already active')
        {'record': lambda: record(path), 'start': lambda: start(path),
         'start8': lambda: start(path, start_step=8),
         'stop': stop, 'stop-record': stop_record}[args.command]()
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print('REFUSED/FAILED: {}: {}'.format(type(error).__name__, error), file=sys.stderr)
        if args.command in ('start', 'start8', 'stop'):
            print('If the robot does not stop, disconnect actual drive power; the PC-power switch does not guarantee motor stop. Keep the PC on for stop-record when possible, and wait for Bag finalized before PC shutdown.', file=sys.stderr)
        return 1
    finally:
        if operation_lock is not None:
            operation_lock.close()


if __name__ == '__main__':
    sys.exit(main())
