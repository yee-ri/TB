#!/usr/bin/env python3
"""Passive ROS 1 Gazebo observer; never publishes commands or changes models.

Run inside the isolated Noetic container, after its ROS master starts. It exits
only itself after STEP 11 plus the requested observation time, or a wall-time
limit. ModelStates/Twist have no header: their times are ROS receipt times.
Planar clearance is evaluated after observation, not in control callbacks.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


def yaw_of(q):
    return math.atan2(2.*(q.w*q.z+q.x*q.y), 1.-2.*(q.y*q.y+q.z*q.z))


def world_shapes(path):
    shapes = {}
    root = ET.parse(path).getroot()
    for model in root.findall('.//world/model'):
        if model.get('name') not in ('tunnel_wall', 'tunnel_obstacle'):
            continue
        for link in model.findall('link'):
            for collision in link.findall('collision'):
                pose = [float(x) for x in collision.findtext('pose', '0 0 0 0 0 0').split()]
                geometry = collision.find('geometry')
                box, cylinder = geometry.find('box'), geometry.find('cylinder')
                record = {'local_pose': pose, 'name': collision.get('name')}
                if box is not None:
                    record.update(kind='wall', size=[float(x) for x in box.findtext('size').split()])
                elif cylinder is not None:
                    record.update(kind='cylinder', radius=float(cylinder.findtext('radius')))
                else:
                    continue
                shapes[model.get('name')+'::'+link.get('name')] = record
    if len(shapes) != 7:
        raise ValueError('expected four tunnel wall links and three cylinder links')
    return shapes


def vertices(poses, front, rear, half_width):
    local = np.array([[front, half_width], [front, -half_width],
                      [-rear, -half_width], [-rear, half_width]])
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    x = poses[:, 0, None]+c[:, None]*local[:, 0]-s[:, None]*local[:, 1]
    y = poses[:, 1, None]+s[:, None]*local[:, 0]+c[:, None]*local[:, 1]
    return np.stack((x, y), axis=-1)


def point_segment_distance(points, start, end):
    direction = end-start
    fraction = np.sum((points-start)*direction, axis=-1)/np.sum(direction*direction, axis=-1)
    closest = start+np.clip(fraction, 0., 1.)[..., None]*direction
    return np.linalg.norm(points-closest, axis=-1)


def clearances(poses, shapes, link_poses):
    """Exact separated planar distances; negative box values are SAT overlap."""
    robot = vertices(poses, .067645, .118073, .0903)
    wall_min = np.full(len(poses), np.inf)
    cylinder_min = wall_min.copy()
    for name, shape in shapes.items():
        if name not in link_poses:
            raise ValueError('missing live link pose: '+name)
        lx, ly, yaw = link_poses[name]
        local = shape['local_pose']
        cx = lx+math.cos(yaw)*local[0]-math.sin(yaw)*local[1]
        cy = ly+math.sin(yaw)*local[0]+math.cos(yaw)*local[1]
        angle = yaw+local[5]
        if shape['kind'] == 'cylinder':
            dx, dy = cx-poses[:, 0], cy-poses[:, 1]
            c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
            px, py = c*dx+s*dy, -s*dx+c*dy
            center = (.067645-.118073)/2.
            qx, qy = np.abs(px-center)-(.067645+.118073)/2., np.abs(py)-.0903
            distance = np.hypot(np.maximum(qx, 0.), np.maximum(qy, 0.))
            distance += np.minimum(np.maximum(qx, qy), 0.)
            cylinder_min = np.minimum(cylinder_min, distance-shape['radius'])
            continue
        length, width = shape['size'][:2]
        wall = vertices(np.array([[cx, cy, angle]]), length/2., length/2., width/2.)[0]
        axes = np.stack((np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2]))),
                         np.column_stack((-np.sin(poses[:, 2]), np.cos(poses[:, 2]))),
                         np.tile([math.cos(angle), math.sin(angle)], (len(poses), 1)),
                         np.tile([-math.sin(angle), math.cos(angle)], (len(poses), 1))), axis=1)
        rp = np.einsum('nvi,nai->nva', robot, axes)
        wp = np.einsum('vi,nai->nva', wall, axes)
        overlap = np.minimum(rp.max(axis=1), wp.max(axis=1))-np.maximum(rp.min(axis=1), wp.min(axis=1))
        intersecting = np.all(overlap >= 0., axis=1)
        distance = np.full(len(poses), np.inf)
        for edge in range(4):
            distance = np.minimum(distance, point_segment_distance(robot, wall[edge], wall[(edge+1)%4]).min(axis=1))
            for vertex in wall:
                distance = np.minimum(distance, point_segment_distance(
                    vertex, robot[:, edge], robot[:, (edge+1)%4]))
        distance[intersecting] = -overlap[intersecting].min(axis=1)
        wall_min = np.minimum(wall_min, distance)
    return wall_min, cylinder_min


def summarize_observations(samples, commands, logs, links, shapes, controller='/tb_controller'):
    result = {'model_samples': len(samples), 'command_samples': len(commands)}
    state_logs = [item for item in logs if item.get('node') == controller]
    handoff = next((item['sim_receipt'] for item in state_logs if re.search(r'STEP:\s*11\b', item['text'])), None)
    complete = next((item['sim_receipt'] for item in state_logs if 'TUNNEL: COMPLETE ' in item['text']), None)
    first_motion = next((row[0] for row in commands if abs(row[1]) > .005 or abs(row[2]) > .005), None)
    extension_reports, last_report = [], None
    for item in state_logs:
        if 'local path extended in ' in item['text']:
            report = item['text'].split('local path extended in ', 1)[1]
            if report != last_report:
                extension_reports.append(item['sim_receipt'])
                last_report = report
    result.update(first_motion_sim=first_motion, tunnel_complete_log_sim=complete,
                  first_step11_camera_log_sim=handoff,
                  observed_entry_log=any('TUNNEL: ENTRY ' in item['text'] for item in state_logs),
                  observed_extension_log_count_lower_bound=len(extension_reports),
                  observed_extension_report_sim_times=extension_reports,
                  zero_command_threshold=.005,
                  command_publishers=sorted(set(row[3] for row in commands)))
    global_reports = {}
    for item in state_logs:
        match = re.search(r'global route (\d+) updated in ([\d.]+)s', item['text'])
        if match:
            global_reports.setdefault(int(match[1]), {'sim_receipt': item['sim_receipt'],
                                                      'planning_wall_s': float(match[2])})
    result['observed_global_route_update_count_lower_bound'] = max(global_reports, default=0)
    result['global_route_reports'] = global_reports
    result['first_tunnel_log_sim'] = next((item['sim_receipt'] for item in state_logs if 'TUNNEL:' in item['text']), None)
    result['latest_observed_step_log'] = next((item for item in reversed(state_logs) if 'STEP:' in item['text']), None)
    result['latest_tunnel_log'] = next((item for item in reversed(state_logs) if 'TUNNEL:' in item['text']), None)
    if first_motion is not None and handoff is not None:
        result['first_motion_to_step11_sim_s'] = handoff-first_motion
    zero_intervals, start = [], None
    for stamp, linear, angular, _caller in commands:
        zero = abs(linear) <= .005 and abs(angular) <= .005
        if zero and start is None:
            start = stamp
        if not zero and start is not None:
            zero_intervals.append([start, stamp])
            start = None
    if start is not None and commands:
        zero_intervals.append([start, commands[-1][0]])
    result['zero_command_intervals_sim'] = zero_intervals
    end = handoff if handoff is not None else (commands[-1][0] if commands else 0.)
    result['zero_command_time_first_motion_to_handoff_sim_s'] = (
        sum(max(0., min(b, end)-max(a, first_motion)) for a, b in zero_intervals)
        if first_motion is not None else None)
    if samples:
        values = np.asarray(samples, dtype=float)
        result['first_world_pose'] = values[0, 1:4].tolist()
        result['last_world_pose'] = values[-1, 1:4].tolist()
        result['maximum_pose_receipt_gap_sim_s'] = float(np.diff(values[:, 0]).max()) if len(values)>1 else None
        result['maximum_body_forward_velocity_mps'] = float(values[:, 4].max())
        if handoff is not None:
            tail = values[values[:, 0] >= handoff]
            result['lane_tail_sim_s'] = float(tail[-1, 0]-handoff) if len(tail) else 0.
            result['lane_tail_distance_m'] = float(np.linalg.norm(np.diff(tail[:, 1:3], axis=0), axis=1).sum()) if len(tail)>1 else 0.
            result['positive_lane_commands'] = sum(t >= handoff and v > .005 for t, v, _w, _c in commands)
        if len(links) == len(shapes):
            first_tunnel = result['first_tunnel_log_sim']
            observed = values[values[:, 0] >= first_tunnel] if first_tunnel is not None else np.empty((0, 6))
            if not len(observed):
                result['geometry'] = {'not_checked': 'no tunnel phase observed'}
                return result
            wall, cylinder = clearances(observed[:, 1:4], shapes, links)
            result['geometry'] = {'minimum_wall_clearance_m': float(wall.min()),
                                  'minimum_cylinder_clearance_m': float(cylinder.min()),
                                  'wall_overlap_samples': int(np.count_nonzero(wall < -1e-9)),
                                  'cylinder_overlap_samples': int(np.count_nonzero(cylinder < -1e-9)),
                                  'samples': len(observed), 'window_start_sim': first_tunnel}
        else:
            result['geometry'] = {'not_checked': 'missing live tunnel link states',
                                  'missing_links': sorted(set(shapes)-set(links))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--world', default='/workspace/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_autorace_2020.world')
    parser.add_argument('--robot', default='custom_autorace')
    parser.add_argument('--controller', default='/tb_controller')
    parser.add_argument('--scope', choices=('official-start', 'full-start', 'tunnel-only'), default='tunnel-only')
    parser.add_argument('--start-step', type=int, default=None)
    parser.add_argument('--after-handoff', type=float, default=4.)
    parser.add_argument('--wall-timeout', type=float, default=180.)
    args = parser.parse_args()
    import rospy
    from gazebo_msgs.msg import ModelStates, LinkStates
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Log
    from sensor_msgs.msg import LaserScan

    rospy.init_node('passive_tunnel_v2_observer', anonymous=True, disable_signals=True)
    if rospy.get_param('/use_sim_time', False) is not True:
        raise RuntimeError('observer requires an isolated Gazebo ROS master with use_sim_time=true')
    shapes = world_shapes(args.world)
    samples, commands, logs, links, counters = [], [], [], {}, {}
    lock = threading.RLock()
    handoff = [None]
    max_link_change = [0.]
    start_wall = time.monotonic()

    def model_callback(message):
        if args.robot not in message.name:
            return
        index = message.name.index(args.robot)
        pose, twist = message.pose[index], message.twist[index]
        yaw = yaw_of(pose.orientation)
        with lock:
            samples.append((rospy.Time.now().to_sec(), pose.position.x, pose.position.y, yaw,
                            math.cos(yaw)*twist.linear.x+math.sin(yaw)*twist.linear.y, twist.angular.z))

    def link_callback(message):
        with lock:
            for name in shapes:
                if name not in message.name:
                    continue
                pose = message.pose[message.name.index(name)]
                value = (pose.position.x, pose.position.y, yaw_of(pose.orientation))
                if name in links:
                    max_link_change[0] = max(max_link_change[0], max(abs(a-b) for a, b in zip(value, links[name])))
                else:
                    links[name] = value

    def command_callback(message):
        with lock:
            commands.append((rospy.Time.now().to_sec(), message.linear.x, message.angular.z,
                             getattr(message, '_connection_header', {}).get('callerid', 'unknown')))

    def log_callback(message):
        if not ('TUNNEL:' in message.msg or 'STEP' in message.msg or 'Tunnel version:' in message.msg
                or 'GREEN LIGHT' in message.msg or 'GATE ' in message.msg or message.level >= 4):
            return
        stamp = rospy.Time.now().to_sec()
        with lock:
            if message.name == args.controller and re.search(r'STEP:\s*11\b', message.msg) and handoff[0] is None:
                handoff[0] = stamp
            # Camera step logs repeat at image rate; keep the first transition.
            if 'STEP:' in message.msg and any(item['text'] == message.msg for item in logs[-3:]):
                return
            logs.append({'sim_receipt': stamp, 'source_stamp': message.header.stamp.to_sec(),
                         'node': message.name, 'text': message.msg})

    def sensor_callback(message, topic):
        now, source = rospy.Time.now().to_sec(), message.header.stamp.to_sec()
        sensor_count(topic, now, source)

    def sensor_count(topic, now, source):
        with lock:
            record = counters.setdefault(topic, {'count': 0, 'maximum_age_sim_s': 0., 'maximum_source_gap_s': 0., 'first_source': source, 'last_source': None})
            record['count'] += 1
            record['maximum_age_sim_s'] = max(record['maximum_age_sim_s'], now-source)
            if record['last_source'] is not None:
                record['maximum_source_gap_s'] = max(record['maximum_source_gap_s'], source-record['last_source'])
            record['last_source'] = source

    def camera_callback(message):
        # Header's seq/secs/nsecs are fixed at the start of serialized Image.
        # No cv_bridge, pixel processing, visualization or image storage.
        if len(message._buff) >= 12:
            _seq, seconds, nanoseconds = struct.unpack_from('<III', message._buff)
            sensor_count('/camera/color/image_raw', rospy.Time.now().to_sec(), seconds+nanoseconds*1e-9)

    subscribers = [rospy.Subscriber('/gazebo/model_states', ModelStates, model_callback, queue_size=2000),
                   rospy.Subscriber('/gazebo/link_states', LinkStates, link_callback, queue_size=100),
                   rospy.Subscriber('/cmd_vel', Twist, command_callback, queue_size=2000),
                   rospy.Subscriber('/rosout_agg', Log, log_callback, queue_size=2000),
                   rospy.Subscriber('/odometry/filtered', Odometry, sensor_callback, callback_args='/odometry/filtered', queue_size=100),
                   rospy.Subscriber('/scan_mid360_raw', LaserScan, sensor_callback, callback_args='/scan_mid360_raw', queue_size=100),
                   rospy.Subscriber('/camera/color/image_raw', rospy.AnyMsg, camera_callback, queue_size=2, buff_size=2**24)]
    reason = 'wall timeout'
    try:
        while not rospy.is_shutdown() and time.monotonic()-start_wall < args.wall_timeout:
            if handoff[0] is not None and rospy.Time.now().to_sec()-handoff[0] >= args.after_handoff:
                reason = 'STEP11 observation window complete'
                break
            time.sleep(.05)
        if rospy.is_shutdown():
            reason = 'ROS shutdown observed'
    except KeyboardInterrupt:
        reason = 'observer interrupted'
    finally:
        for subscriber in subscribers:
            subscriber.unregister()
    with lock:
        for record in counters.values():
            duration = (record['last_source'] or 0.)-record['first_source']
            record['source_rate_hz'] = (record['count']-1)/duration if duration > 0. else None
        result = summarize_observations(samples, commands, logs, links, shapes, args.controller)
        result.update(requested_scope=args.scope, requested_start_step=args.start_step,
                      expected_controller_node=args.controller,
                      scope=('Official-start observation requested; actual reached steps are reported separately; STEP11 is not a finish-line test'
                             if args.scope in ('official-start', 'full-start') else
                             'Tunnel-entry observation; preceding missions and official-start course are outside scope'),
                      observer_end_reason=reason, observer_wall_seconds=time.monotonic()-start_wall,
                      world_path=args.world, world_sha256=hashlib.sha256(Path(args.world).read_bytes()).hexdigest(),
                      live_link_poses=links, maximum_static_link_pose_change=max_link_change[0], sensors=counters,
                      logs=logs, limitations=[
                          'Read-only observer; no cmd_vel/step publication, teleport, ROS/Gazebo shutdown or robot service calls.',
                          'Command and ground-truth timestamps are ROS receipt time; reported intervals are simulated time.',
                          'Zero Twist intervals are commanded stops, not proof of physically stopped wheels.',
                          'Extension logs provide observed updates, not exact worker dispatch/completion counts or pending-worker intervals.',
                          'Clearance uses every received planar nominal-body sample and live link poses with source collision dimensions; no interpolation or contact-sensor proof.',
                          'Static geometry results require unchanged live link poses; counts do not prove no subscriber drops.',
                          'STEP11 plus positive commands and ground-truth displacement shows lane-control continuation, not lane-boundary compliance.'])
        if max_link_change[0] > 1e-6:
            result['geometry']['validity'] = 'invalid: tunnel links moved after the geometry snapshot'
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        raw = output.with_suffix('.samples.jsonl')
        with raw.open('w', encoding='utf-8') as stream:
            for row in samples:
                stream.write(json.dumps({'type': 'model', 'values': row})+'\n')
            for row in commands:
                stream.write(json.dumps({'type': 'command', 'values': row})+'\n')
    print(json.dumps({'output': str(output), 'raw_samples': str(raw),
                      'end_reason': reason, 'summary': {key: result.get(key) for key in
                      ('first_motion_to_step11_sim_s', 'lane_tail_distance_m', 'geometry')}}), flush=True)


if __name__ == '__main__':
    main()
