"""Run ONLY inside an isolated --network none Noetic container.

Starts a private ROS master and the actual TB node without sensors. Checks
imports, initialization, a single cmd_vel publisher and stopped waiting.
This is transport smoke testing, not navigation or Gazebo verification.
"""
import os
from pathlib import Path
import signal
import subprocess
import time

os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:11311'
os.environ['ROS_HOSTNAME'] = '127.0.0.1'

import rosgraph
import rospy
from geometry_msgs.msg import Twist


def main():
    processes = []
    root = Path(__file__).resolve().parents[1]
    try:
        processes.append(subprocess.Popen(['roscore'], stdout=subprocess.DEVNULL,
                                          stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 15.
        master = rosgraph.Master('/tunnel_port_smoke')
        while time.monotonic() < deadline:
            try:
                master.getPid()
                break
            except Exception:
                time.sleep(.1)
        else:
            raise RuntimeError('private ROS master did not start')
        rospy.init_node('tunnel_port_smoke', disable_signals=True)
        commands = []
        subscriber = rospy.Subscriber('/cmd_vel', Twist, commands.append, queue_size=20)
        for profile in ('normal', 'fast'):
            commands.clear()
            process = subprocess.Popen(
                ['python3', str(root / 'codes/stepbystep/try_maze.py'),
                 '_start_step:=9', '_tunnel_profile:='+profile],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            processes.append(process)
            deadline = time.monotonic() + 10.
            while len(commands) < 10 and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read().decode())
                time.sleep(.05)
            assert len(commands) >= 10, 'actual node did not publish waiting commands'
            publishers = dict(master.getSystemState()[0]).get('/cmd_vel', [])
            assert len(publishers) == 1, publishers
            assert all(m.linear.x == 0. and m.angular.z == 0. for m in commands)
            print('ROS_SMOKE_PASS: '+profile+' actual TB node, one cmd_vel owner, sensorless wait is stopped')
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=8.)
            deadline = time.monotonic()+5.
            while dict(master.getSystemState()[0]).get('/cmd_vel', []) and time.monotonic() < deadline:
                time.sleep(.05)
            assert not dict(master.getSystemState()[0]).get('/cmd_vel', []), 'publisher did not shut down'
        subscriber.unregister()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=8.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5.)


if __name__ == '__main__':
    main()
