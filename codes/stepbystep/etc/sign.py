#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class TemplateMatch:
    def __init__(self):
        rospy.init_node('template_match', anonymous=True)
        self.bridge = CvBridge()

        self.size = 100
        self.min_score = 0.50       # 최소 유사도: 실제 영상으로 조절
        self.min_gap = 0.08         # 좌우 점수 차이
        self.confirm_frames = 3    # 같은 방향 연속 검출 횟수
        self.last_direction = None
        self.direction_count = 0

        self.left_template = self.load_template('/home/sj/Desktop/TB/images/left.png')
        self.right_template = self.load_template('/home/sj/Desktop/TB/images/right.png')

        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)

    def blue_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([85, 70, 40]), np.array([135, 255, 255]))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def arrow_mask(self, image):
        image = cv2.resize(image, (self.size, self.size))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([179, 100, 255]))

        inner = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(inner, (self.size // 2, self.size // 2), int(self.size * 0.40), 255, -1)
        return cv2.bitwise_and(mask, inner)

    def load_template(self, path):
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError("템플릿 이미지를 읽을 수 없습니다: " + path)

        contours, _ = cv2.findContours(self.blue_mask(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("템플릿에서 파란 표지판을 찾지 못했습니다: " + path)

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        template = self.arrow_mask(image[y:y+h, x:x+w])
        if np.std(template) < 1:
            raise RuntimeError("템플릿에서 흰색 화살표를 찾지 못했습니다: " + path)
        return template

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        blue = self.blue_mask(image)
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        direction = None
        best_score = -1.0
        best_box = None
        best_arrow = np.zeros((self.size, self.size), dtype=np.uint8)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 400:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if not 0.65 <= w / float(h) <= 1.5:
                continue
            if area / float(w * h) < 0.45:
                continue

            arrow = self.arrow_mask(image[y:y+h, x:x+w])
            white_ratio = cv2.countNonZero(arrow) / float(self.size * self.size)
            if not 0.03 <= white_ratio <= 0.40:
                continue

            left = float(cv2.matchTemplate(arrow, self.left_template, cv2.TM_CCOEFF_NORMED)[0, 0])
            right = float(cv2.matchTemplate(arrow, self.right_template, cv2.TM_CCOEFF_NORMED)[0, 0])
            if not np.isfinite(left) or not np.isfinite(right):
                continue

            score = max(left, right)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(image, "L:%.2f R:%.2f" % (left, right), (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if score >= self.min_score and abs(left - right) >= self.min_gap and score > best_score:
                best_score = score
                direction = 'LEFT' if left > right else 'RIGHT'
                best_box = (x, y, w, h)
                best_arrow = arrow

        if direction is None:
            self.last_direction = None
            self.direction_count = 0
        elif direction == self.last_direction:
            self.direction_count += 1
        else:
            self.last_direction = direction
            self.direction_count = 1

        confirmed = direction if self.direction_count >= self.confirm_frames else 'NONE'

        if best_box is not None:
            x, y, w, h = best_box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        rospy.loginfo_throttle(0.3, "SIGN: %s / candidate: %s / score: %.2f", confirmed, direction or 'NONE', max(0.0, best_score))
        cv2.putText(image, confirmed, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('SIGN CAMERA', image)
        cv2.imshow('BLUE MASK', blue)
        cv2.imshow('ARROW MASK', best_arrow)
        cv2.waitKey(1)


if __name__ == '__main__':
    node = TemplateMatch()
    try:
        rospy.spin()
    finally:
        cv2.destroyAllWindows()