#!/usr/bin/env python3
"""Show the live camera image and its LUT color-recognition result."""

from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class LutRecognizer:
    def __init__(self):
        rospy.init_node("lut_recognize", anonymous=True)

        lut_path = Path(__file__).resolve().with_name("current.lut")
        if not lut_path.is_file():
            raise FileNotFoundError(f"LUT file not found: {lut_path}")
        if lut_path.stat().st_size != 256 ** 3:
            raise ValueError(f"LUT must contain exactly 16777216 bytes: {lut_path}")

        self.table = np.fromfile(str(lut_path), dtype=np.uint8)
        allowed_labels = {0, 1, 2, 4, 8, 16}
        found_labels = set(np.unique(self.table).tolist())
        if not found_labels.issubset(allowed_labels):
            raise ValueError(f"Unexpected LUT labels: {sorted(found_labels - allowed_labels)}")

        self.bridge = CvBridge()
        camera_topic = rospy.get_param("~camera_topic", "/camera/color/image_raw")
        self.subscriber = rospy.Subscriber(
            camera_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24
        )
        rospy.on_shutdown(self.close_windows)
        rospy.loginfo("Showing camera and LUT result from %s", camera_topic)

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            rospy.logerr("Could not convert camera image: %s", error)
            return

        pixels = image.astype(np.uint32)
        indices = (pixels[:, :, 2] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 0]
        labels = self.table[indices]

        # Display colors match the LUT debug palette used by the main program.
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0] = (26, 26, 26)
        palette[1] = (255, 255, 255)
        palette[2] = (0, 160, 255)
        palette[4] = (255, 0, 0)
        palette[8] = (0, 0, 255)
        palette[16] = (0, 255, 0)
        recognized = palette[labels]

        cv2.imshow("Original Image", image)
        cv2.imshow("LUT Recognized Image", recognized)
        cv2.waitKey(1)

    @staticmethod
    def close_windows():
        cv2.destroyAllWindows()


if __name__ == "__main__":
    LutRecognizer()
    rospy.spin()
