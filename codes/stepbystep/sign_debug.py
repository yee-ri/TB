#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os


def sign_blue_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([101, 255, 18]), np.array([108, 255, 26]))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def sign_arrow_mask(image):
    image = cv2.resize(image, (100, 100))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([66, 64, 30]), np.array([90, 71, 43]))
    inner = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(inner, (50, 50), 40, 255, -1)
    return cv2.bitwise_and(mask, inner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='png/jpg image path')
    parser.add_argument('--save', action='store_true', help='save result images')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError('Cannot read image: ' + args.image_path)

    blue_mask = sign_blue_mask(image)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    crop_view = np.zeros((100, 100, 3), dtype=np.uint8)
    arrow_mask = np.zeros((100, 100), dtype=np.uint8)
    arrow_overlay = np.zeros((100, 100, 3), dtype=np.uint8)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        sign_crop = image[y:y + h, x:x + w]
        crop_view = cv2.resize(sign_crop, (100, 100))
        arrow_mask = sign_arrow_mask(sign_crop)

        arrow_overlay = crop_view.copy()
        arrow_overlay[arrow_mask > 0] = (0, 0, 255)
    else:
        print('No blue sign region detected.')

    h, w = image.shape[:2]
    display_w = 400
    display_h = int(h * display_w / w)

    original_view = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
    result_view = cv2.resize(result, (display_w, display_h), interpolation=cv2.INTER_AREA)
    blue_view = cv2.resize(blue_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
    crop_view_big = cv2.resize(crop_view, (250, 250), interpolation=cv2.INTER_AREA)
    arrow_mask_big = cv2.resize(arrow_mask, (250, 250), interpolation=cv2.INTER_NEAREST)
    arrow_overlay_big = cv2.resize(arrow_overlay, (250, 250), interpolation=cv2.INTER_AREA)

    cv2.imshow('ORIGINAL', original_view)
    cv2.imshow('BLUE HSV MASK', blue_view)
    cv2.imshow('SIGN DETECTION', result_view)
    cv2.imshow('SIGN CROP', crop_view_big)
    cv2.imshow('ARROW HSV MASK', arrow_mask_big)
    cv2.imshow('ARROW OVERLAY', arrow_overlay_big)

    if args.save:
        base = os.path.splitext(os.path.basename(args.image_path))[0]
        out_dir = os.path.join(os.path.dirname(args.image_path), base + '_hsv_debug')
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, '1_original.png'), image)
        cv2.imwrite(os.path.join(out_dir, '2_blue_mask.png'), blue_mask)
        cv2.imwrite(os.path.join(out_dir, '3_detection.png'), result)
        cv2.imwrite(os.path.join(out_dir, '4_sign_crop.png'), crop_view)
        cv2.imwrite(os.path.join(out_dir, '5_arrow_mask.png'), arrow_mask)
        cv2.imwrite(os.path.join(out_dir, '6_arrow_overlay.png'), arrow_overlay)
        print('Saved to:', out_dir)

    print('Press any key on an image window to close.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
