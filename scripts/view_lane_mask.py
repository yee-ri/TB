#!/usr/bin/env python3
"""Camera-only preview of try_maze's default lane colour mask."""

import cv2
import numpy as np


def lane_mask(image):
    """Match the default lane_tracking crop, HSV bounds and side selection."""
    crop = image[300:, :]
    if crop.size == 0:
        raise ValueError('Camera image must be taller than 300 pixels')
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([179, 55, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([50, 255, 255]))
    # Mirrors the default (no special turn/parking mode) lane_tracking branch.
    white[:, :crop.shape[1] // 2] = 0
    return cv2.bitwise_or(white, yellow)


def main():
    import rospy
    from cv_bridge import CvBridge, CvBridgeError
    from sensor_msgs.msg import Image

    rospy.init_node('lane_mask_view', anonymous=True)
    bridge = CvBridge()
    latest = [None]

    def receive(message):
        try:
            latest[0] = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as error:
            rospy.logwarn_throttle(2., 'Cannot read camera image: %s', error)

    subscriber = rospy.Subscriber('/camera/color/image_raw', Image, receive,
                                  queue_size=1, buff_size=2**24)
    window = 'LANE MASK - normal mode (Q / Esc to close)'
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        waiting = np.zeros((180, 640), dtype=np.uint8)
        cv2.putText(waiting, 'Waiting for camera...', (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 1)
        cv2.imshow(window, waiting)
        rospy.loginfo('Camera-only lane mask; no motor command publisher. Q / Esc or Ctrl+C to close.')
        while not rospy.is_shutdown():
            frame = latest[0]
            if frame is not None:
                try:
                    cv2.imshow(window, lane_mask(frame))
                except ValueError as error:
                    rospy.logwarn_throttle(2., '%s', error)
            if cv2.waitKey(30) & 0xff in (27, ord('q')):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        subscriber.unregister()
        cv2.destroyAllWindows()
        rospy.signal_shutdown('Lane mask window closed')


if __name__ == '__main__':
    main()
