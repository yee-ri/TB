"""Bounded physical-start handshake using the controller's sole cmd_vel publisher."""
from contextlib import contextmanager
import math
import threading
import time


CORE = '/turtlebot3_core'


def _connected(publisher, topic):
    # A recorder also subscribes to cmd_vel, so a nonzero connection count alone
    # cannot establish that OpenCR's rosserial bridge receives this publisher.
    return any(row[1] == CORE and row[4] == topic and row[5]
               for row in publisher.impl.get_stats_info())


def _check_graph(ros, deadline):
    import rosgraph
    result = []
    def read():
        try:
            master = rosgraph.Master(ros.get_name())
            result.append((master.getSystemState(), master.getTopicTypes(),
                           ros.get_param('/use_sim_time', False)))
        except Exception as error:
            result.append(error)
    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(min(.25, max(0., deadline-time.monotonic())))
    if not result or isinstance(result[0], Exception):
        raise RuntimeError('Motor start: ROS graph unavailable within the startup deadline')
    (pubs, subs, _), topic_types, sim_time = result[0]
    pubs, subs, topic_types = dict(pubs), dict(subs), dict(topic_types)
    node = ros.get_name()
    expected_types = {'/cmd_vel': 'geometry_msgs/Twist', '/motor_power': 'std_msgs/Bool',
                      '/sensor_state': 'turtlebot3_msgs/SensorState', '/cmd_vel_rc100': 'geometry_msgs/Twist'}
    if sim_time is not False:
        raise RuntimeError('Motor start requires use_sim_time=false')
    if any(topic_types.get(topic) != kind for topic, kind in expected_types.items()):
        raise RuntimeError('Motor start: unexpected or missing drivetrain topic type')
    if set(pubs.get('/cmd_vel', [])) != {node} or set(pubs.get('/motor_power', [])) != {node}:
        raise RuntimeError('Motor start: another command or motor-power publisher is active')
    if any(CORE not in subs.get(topic, []) for topic in ('/cmd_vel', '/motor_power')):
        raise RuntimeError('Motor start: expected OpenCR subscriptions are missing')
    if any(set(pubs.get(topic, [])) != {CORE} for topic in ('/sensor_state', '/cmd_vel_rc100')):
        raise RuntimeError('Motor start: feedback must come only from turtlebot3_core')


@contextmanager
def arm_motor_torque(command_publisher, zero_message, cancelled, abort, command_lock):
    """Yield only after fresh, post-request OpenCR torque feedback.

    Caller activates its movement gate and timer inside the context. Any error,
    shutdown or interruption before leaving the context rolls back to repeated
    zero commands plus torque-off. This function never creates cmd_vel.
    """
    import rospy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool
    from turtlebot3_msgs.msg import SensorState

    started = time.monotonic()
    deadline = started + 2.5
    feedback = {'sensor': None, 'remote': None, 'error': None}
    subscribers = []
    power = None
    succeeded = False

    def source(message):
        if (getattr(message, '_connection_header', {}) or {}).get('callerid') != CORE:
            feedback['error'] = 'unexpected feedback publisher'

    def sensor(message):
        source(message)
        if message.button != 0:
            feedback['error'] = 'OpenCR button input is active'
        feedback['sensor'] = (time.monotonic(), message.header.stamp.to_sec(),
                              bool(message.torque), message.button)

    def remote(message):
        source(message)
        values = (message.linear.x, message.linear.y, message.linear.z,
                  message.angular.x, message.angular.y, message.angular.z)
        if any(not math.isfinite(value) or abs(value) > 1e-9 for value in values):
            feedback['error'] = 'RC100 remote input is active or invalid'
        feedback['remote'] = time.monotonic()

    def check_cancelled():
        if rospy.is_shutdown() or cancelled():
            raise RuntimeError('Motor start cancelled before controller activation')
        if feedback['error']:
            raise RuntimeError('Motor start: ' + feedback['error'])

    def fresh_state():
        state, remote_at = feedback['sensor'], feedback['remote']
        now = time.monotonic()
        ros_now = rospy.Time.now().to_sec()
        if (state is not None and remote_at is not None
                and 0. <= now-state[0] <= .20 and 0. <= now-remote_at <= .20
                and -.02 <= ros_now-state[1] <= .20):
            return state
        return None

    def pause():
        time.sleep(min(.05, max(0., deadline-time.monotonic())))

    try:
        power = rospy.Publisher('/motor_power', Bool, queue_size=1, latch=False)
        subscribers.append(rospy.Subscriber('/sensor_state', SensorState, sensor, queue_size=1))
        subscribers.append(rospy.Subscriber('/cmd_vel_rc100', Twist, remote, queue_size=1))
        _check_graph(rospy, deadline)
        connected_since = None
        requested = None
        last_ack = None
        acknowledgements = 0
        while time.monotonic() < deadline:
            check_cancelled()
            now = time.monotonic()
            command_publisher.publish(zero_message())
            connected = _connected(command_publisher, '/cmd_vel') and _connected(power, '/motor_power')
            if not connected:
                connected_since = None
                if requested is not None or now-started >= 1.:
                    raise RuntimeError('Motor start: no confirmed command connection to OpenCR')
                pause()
                continue
            if connected_since is None:
                connected_since = now
            state = fresh_state()
            if state is None:
                if requested is not None:
                    raise RuntimeError('Motor start: OpenCR feedback became stale')
                pause()
                continue
            if requested is None and now-connected_since >= .30:
                _check_graph(rospy, deadline)
                with command_lock:
                    check_cancelled()
                    if fresh_state() is None or time.monotonic() >= deadline:
                        raise RuntimeError('Motor start: feedback expired before torque request')
                    requested = rospy.Time.now().to_sec()
                    power.publish(Bool(data=True))
            elif requested is not None and state[1] > requested:
                if state[1] != last_ack:
                    last_ack = state[1]
                    acknowledgements = acknowledgements+1 if state[2] else 0
                if acknowledgements >= 2:
                    _check_graph(rospy, deadline)
                    with command_lock:
                        check_cancelled()
                        state = fresh_state()
                        if (state is None or not state[2] or state[1] <= requested
                                or time.monotonic() >= deadline):
                            raise RuntimeError('Motor start: torque acknowledgement expired before activation')
                        rospy.loginfo('OpenCR torque enabled and fresh feedback confirmed; starting controller')
                        try:
                            yield
                            check_cancelled()
                            succeeded = True
                        except BaseException:
                            # Revoke movement while still holding the same
                            # lock as shutdown/move, before rollback publishes.
                            abort()
                            raise
                    return
            pause()
        raise RuntimeError('Motor start: fresh OpenCR torque acknowledgement not received within 2.5 seconds')
    finally:
        try:
            if not succeeded:
                try:
                    abort()
                finally:
                    # shutdown may itself fail after lowering the movement
                    # gate; that must not skip the torque-off rollback.
                    rollback_end = time.monotonic() + .30
                    while time.monotonic() < rollback_end:
                        try:
                            command_publisher.publish(zero_message())
                        except Exception:
                            pass
                        try:
                            if power is not None:
                                power.publish(Bool(data=False))
                        except Exception:
                            pass
                        time.sleep(min(.05, max(0., rollback_end-time.monotonic())))
        finally:
            for subscriber in subscribers:
                try:
                    subscriber.unregister()
                except Exception:
                    pass
            if power is not None:
                try:
                    power.unregister()
                except Exception:
                    pass
