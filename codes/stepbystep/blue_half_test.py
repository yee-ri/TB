#!/usr/bin/env python3

from threading import Lock

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


def sign_blue_mask(image):
    # parking_step5_updated.py의 sign_blue_mask와 동일한 조건입니다.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([96, 255, 9]), np.array([108, 255, 21]))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


class BlueMaskMonitor:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest = None
        self.show_window = rospy.get_param('~show_window', True)
        topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.subscriber = rospy.Subscriber(
            topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        rospy.loginfo('BLUE MASK MONITOR: %s', topic)

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        except CvBridgeError as error:
            rospy.logwarn_throttle(1.0, 'IMAGE ERROR: %s', error)
            return

        mask = sign_blue_mask(image)
        height, width = mask.shape
        half = width // 2
        total = cv2.countNonZero(mask)
        left = cv2.countNonZero(mask[:, :half])
        right = cv2.countNonZero(mask[:, half:])
        direction = 'LEFT' if left > right else 'RIGHT' if right > left else 'TIE'
        threshold = 'OVER 3000' if total > 3000 else '3000 OR LESS'

        rospy.loginfo_throttle(
            0.2, 'TOTAL: %d | LEFT: %d | RIGHT: %d | %s | %s',
            total, left, right, direction, threshold
        )

        if not self.show_window:
            return

        color = (0, 0, 255) if total > 3000 else (0, 255, 0)
        camera_view = image.copy()
        mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for view in (camera_view, mask_view):
            cv2.line(view, (half, 0), (half, height - 1), (0, 255, 255), 2)

        # 검은 띠 위에 수치를 표시해 배경과 무관하게 읽을 수 있게 합니다.
        banner = np.zeros((100, width, 3), dtype=np.uint8)
        cv2.putText(
            banner, 'TOTAL: {}   LEFT: {}   RIGHT: {}'.format(total, left, right),
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            banner, '{}   {}   |   Q: quit'.format(direction, threshold),
            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )
        display = cv2.vconcat([banner, camera_view, mask_view])
        with self.lock:
            self.latest = display

    def run(self):
        if not self.show_window:
            rospy.spin()
            return

        name = 'Blue Mask Monitor'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 848, 800)
        rate = rospy.Rate(30)
        try:
            while not rospy.is_shutdown():
                with self.lock:
                    display = self.latest
                if display is not None:
                    cv2.imshow(name, display)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    rospy.signal_shutdown('Monitor closed')
                    break
                rate.sleep()
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('blue_mask_monitor', anonymous=True)
    try:
        BlueMaskMonitor().run()
    except rospy.ROSInterruptException:
        pass
