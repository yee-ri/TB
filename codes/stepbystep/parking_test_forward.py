#!/usr/bin/env python3
"""ROS1 전진 회전 진입 후 바로 후진 회전하는 단독 테스트. 시작할 때 로봇을 주차 통로 진행 방향으로 놓으세요.

예: python3 parking_test_forward.py --side right
    python3 parking_test_forward.py --side left

전체 코스 controller 등 /cmd_vel을 보내는 다른 노드는 종료하고 실행하세요.
이 파일은 모터 토크를 활성화하지 않습니다. /odom 및 구동 노드가 필요합니다.
"""
import argparse
import math
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ParkingTuner:
    def __init__(self, args):
        self.args = args
        self.yaw = None
        self.odom_time = None
        self.pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)
        self.sub = rospy.Subscriber(args.odom_topic, Odometry, self.odom, queue_size=1)
        rospy.on_shutdown(self.stop)

    def odom(self, msg):
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w*q.z + q.x*q.y),
                              1 - 2 * (q.y*q.y + q.z*q.z))
        self.odom_time = time.monotonic()

    def move(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.pub.publish(msg)

    def stop(self):
        self.move(0.0, 0.0)

    def check_odom(self):
        if self.odom_time is None or time.monotonic() - self.odom_time > 0.5:
            raise RuntimeError('odom이 0.5초 이상 갱신되지 않아 정지합니다.')

    def timed_move(self, linear, duration, target_yaw=None):
        deadline = time.monotonic() + duration
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.check_odom()
            angular = 0.0
            if target_yaw is not None:
                err = wrap(target_yaw - self.yaw)
                angular = max(-0.15, min(0.15, err * 1.5))
            self.move(linear, angular)
            time.sleep(1/30)
        self.stop()
        if rospy.is_shutdown():
            raise RuntimeError('테스트 중단')

    def turn(self, target, direction, reverse=False):
        a = self.args
        deadline = time.monotonic() + a.timeout
        while not rospy.is_shutdown():
            self.check_odom()
            err = wrap(target - self.yaw)
            rospy.loginfo_throttle(0.3, 'yaw=%.1f target=%.1f error=%.1f deg',
                                   math.degrees(self.yaw), math.degrees(target),
                                   math.degrees(err))
            if abs(err) <= math.radians(a.tolerance):
                self.stop()
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('회전 제한시간 초과')
            if direction * err < 0:
                raise RuntimeError('목표 방향을 지나쳤습니다. 회전속도를 낮춰보세요.')
            limit = a.max_angular if reverse else a.entry_angular
            angular = direction * min(limit, max(0.06, abs(err) * a.gain))
            linear = (-a.reverse_speed if reverse else a.entry_speed) * min(1.0, abs(err)/0.3)
            self.move(linear, angular)
            time.sleep(1/30)
        raise RuntimeError('테스트 중단')

    def run(self):
        deadline = time.monotonic() + 10
        while self.yaw is None or self.pub.get_num_connections() == 0:
            if rospy.is_shutdown() or time.monotonic() > deadline:
                raise RuntimeError('10초 안에 odom 또는 속도 명령 수신 노드가 준비되지 않았습니다.')
            time.sleep(0.05)
        self.stop()
        rospy.loginfo('%.1f초 후 주차 시작. Ctrl+C로 중단합니다.', self.args.delay)
        deadline = time.monotonic() + self.args.delay
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.stop()
            time.sleep(0.05)
        if rospy.is_shutdown():
            return
        self.check_odom()
        # 주차 동작 전 직진: 속도(m/s), 시간(초)을 여기서 수정하세요.
        # self.timed_move(0.06, 0.5)
        self.check_odom()
        base = self.yaw
        direction = 1.0 if self.args.side == 'right' else -1.0
        # 기존 통로 yaw=-pi/2 기준 right=0, left=-pi, 출구=+pi/2를
        # 테스트 시작 시점 yaw에 대한 상대각으로 옮깁니다.
        entry = wrap(base + direction * math.radians(self.args.entry_angle))
        exit_yaw = wrap(base + direction * math.pi)
        rospy.loginfo('기준=%.1f 주차=%.1f 출구=%.1f deg',
                      math.degrees(base), math.degrees(entry), math.degrees(exit_yaw))
        # left: 오른쪽으로 전진 회전 → 같은 회전 방향으로 후진 회전
        # right: 왼쪽으로 전진 회전 → 같은 회전 방향으로 후진 회전
        self.turn(entry, direction)

        # 회전 없이 직선 후진: 속도(m/s), 시간(초)
        # self.timed_move(-0.06, 1)

        # 이후 후진하면서 출구 방향으로 회전
        self.turn(exit_yaw, direction, reverse=True)
        self.stop()
        rospy.loginfo('테스트 완료: 시작 방향의 반대 방향에서 정지합니다.')



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--side', choices=['right', 'left'], required=True)
    a = p.parse_args(rospy.myargv()[1:])

    # 아래 값을 코드에서 직접 수정하세요. 속도는 m/s, 각속도는 rad/s입니다.
    a.entry_speed = 0.06    # 주차 방향으로 회전하는 동안 전진 속도
    a.entry_angular = 0.35    # 진입 회전 속도 상한
    a.entry_angle = 90        # 진입 회전각 (도)
    a.reverse_speed = 0.06   # 후진 회전 속력
    a.max_angular = 0.35       # 후진 회전 속도 상한
    a.gain = 1.2
    a.tolerance = 2           # 방향 허용 오차 (도)
    a.timeout = 15            # 각 회전 제한시간 (초)
    a.delay = 3               # 시작 전 대기 (초)
    a.odom_topic = '/odom'
    a.cmd_topic = '/cmd_vel'
    for key in ('entry_speed', 'entry_angular',
                'reverse_speed', 'max_angular', 'gain', 'tolerance', 'timeout'):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(key + ' 값은 유한한 양수여야 합니다.')
    for key in ('entry_angle',):
        if not math.isfinite(getattr(a, key)) or not 0 < getattr(a, key) < 180:
            p.error(key + ' 값은 0도 초과, 180도 미만이어야 합니다.')
    if not math.isfinite(a.delay) or a.delay < 0:
        p.error('delay는 유한한 0 이상의 값이어야 합니다.')
    return a


if __name__ == '__main__':
    args = parse_args()
    rospy.init_node('parking_tuner', anonymous=True)
    tuner = ParkingTuner(args)
    try:
        tuner.run()
    except (RuntimeError, KeyboardInterrupt) as exc:
        rospy.logwarn('주차 테스트 중단: %s', exc)
    finally:
        tuner.stop()
    