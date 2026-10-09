#!/usr/bin/env python3
"""Observe duplicate sensor mount TF authorities; never publish or write config."""
import threading
import time

MOUNT_FRAMES = {'lidar': 'livox_frame', 'camera': 'camera_link'}


def check_existing_mount_authorities():
    """Observe existing TF for one wall-clock second; never publish a message.

    Latched /tf_static messages must be inspected even if their stamp is zero
    or old. A stamp does not make an existing child authority safe to replace.
    """
    import rosgraph
    import rospy
    from tf2_msgs.msg import TFMessage

    if not rosgraph.is_master_online():
        print('No local ROS master is online; no existing TF publishers to inspect.')
        return
    rospy.init_node('nuc_mount_authority_check', anonymous=True,
                    disable_signals=True, disable_rosout=True)
    master = rosgraph.Master(rospy.get_name())
    lock = threading.Lock()
    seen = set()
    existing = set()

    def tf_callback(message, topic):
        caller = getattr(message, '_connection_header', {}).get('callerid', 'unknown')
        with lock:
            seen.add((topic, caller))
            for transform in message.transforms:
                child = transform.child_frame_id.lstrip('/')
                if child in MOUNT_FRAMES.values():
                    existing.add((child, transform.header.frame_id.lstrip('/'),
                                  caller, topic, transform.header.stamp.to_sec()))

    def tf_publishers():
        publishers = dict(master.getSystemState()[0])
        return {(topic, caller) for topic in ('/tf', '/tf_static')
                for caller in publishers.get(topic, [])}

    subscriptions = []
    try:
        before = tf_publishers()
        subscriptions = [rospy.Subscriber(topic, TFMessage, tf_callback,
                                         callback_args=topic, queue_size=100)
                         for topic in ('/tf', '/tf_static')]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        after = tf_publishers()
        with lock:
            conflicts = sorted(existing)
            unseen = (before | after) - seen
        if conflicts:
            description = '; '.join('{} <- {} from {} on {} (stamp={})'.format(*entry)
                                    for entry in conflicts)
            raise ValueError('existing sensor mount TF authority blocks --mounts: '
                             + description + '. Stop its publisher or use the existing calibrated TF.')
        if unseen:
            raise ValueError('TF authority check is inconclusive: no TF message received '
                             'within 1s from ' + ', '.join('{} ({})'.format(caller, topic)
                                                          for topic, caller in sorted(unseen)))
        print('One-second TF authority check passed: camera_link/livox_frame have no observed publisher.')
    finally:
        for subscription in subscriptions:
            subscription.unregister()
        rospy.signal_shutdown('Read-only mount authority check finished')



if __name__ == '__main__':
    try:
        check_existing_mount_authorities()
    except (ValueError, OSError) as error:
        raise SystemExit('Sensor TF conflict: ' + str(error))
