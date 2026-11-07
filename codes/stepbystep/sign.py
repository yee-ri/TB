from pathlib import Path
import argparse
import cv2
import numpy as np

class _ColorLUTAddition:

    def __init__(self, path):
        self.load_lut(path)

    def load_lut(self, path):
        path = Path(path)
        if path.stat().st_size != 256 ** 3:
            raise ValueError('LUT must contain exactly 16777216 bytes: ' + str(path))
        table = np.fromfile(str(path), dtype=np.uint8)
        if table.size != 256 ** 3:
            raise ValueError('LUT read was incomplete')
        values, counts = np.unique(table, return_counts=True)
        if not set(values.tolist()).issubset({0, 1, 2, 4, 8, 16}):
            raise ValueError('Unexpected LUT class values')
        self.table = table
        self.counts = dict(zip(values.tolist(), counts.tolist()))

    def rgb_to_label(self, image, bgr=False):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected a uint8 H x W x 3 image')
        pixels = image.astype(np.uint32)
        if bgr:
            r, g, b = (pixels[:, :, 2], pixels[:, :, 1], pixels[:, :, 0])
        else:
            r, g, b = (pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2])
        index = r << 16 | g << 8 | b
        return self.table[index]

    @staticmethod
    def downsample_label(label):
        if label.dtype != np.uint8 or label.ndim != 2:
            raise ValueError('Expected a uint8 H x W label image')
        h, w = label.shape
        if h % 2 or w % 2:
            raise ValueError('OR downsampling requires even width and height')
        return label[0::2, 0::2] | label[0::2, 1::2] | label[1::2, 0::2] | label[1::2, 1::2]

    @staticmethod
    def label_to_rgb(label):
        if label.dtype != np.uint8 or label.ndim != 2:
            raise ValueError('Expected a uint8 H x W label image')
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0] = [26, 26, 26]
        palette[1] = [255, 255, 255]
        palette[2] = [255, 160, 0]
        palette[4] = [0, 0, 255]
        palette[8] = [255, 0, 0]
        palette[16] = [0, 255, 0]
        return palette[label]

    def mask(self, bgr, label):
        return (self.rgb_to_label(bgr, bgr=True) == label).astype(np.uint8) * 255

class SignTest:
    def sign_blue_mask(self, image):
        mask = self._color_lut.mask(image, 4)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def __init__(self, lut):
        self._color_lut = _ColorLUTAddition(lut)

    def arrow(self, image, mode):
        image = cv2.resize(image, (100, 100))
        if mode == 'lut':
            mask = self._color_lut.mask(image, 1)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, int(mode)]), np.array([179, 100, 255]))
        inner = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(inner, (50, 50), 40, 255, -1)
        return cv2.bitwise_and(mask, inner)

    def template(self, path, mode, output):
        image = read_image(path)
        blue = self.sign_blue_mask(image)
        save(output / (path.stem + '_template_blue.png'), blue)
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(f'{path}: no LUT blue region; see template_blue.png')
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        crop = image[y:y+h, x:x+w]
        arrow = self.arrow(crop, mode)
        save(output / (path.stem + '_template_crop.png'), crop)
        save(output / (path.stem + '_template_arrow.png'), arrow)
        if np.std(arrow) == 0:
            raise ValueError(f'{path}: empty/constant template arrow; try --template-mode 80 or 40')
        return arrow


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def save(path, image):
    if not cv2.imwrite(str(path), image):
        raise IOError(f'Cannot save: {path}')


def main():
    parser = argparse.ArgumentParser(description='Offline arrow comparison. No ROS, camera, or motor commands.')
    parser.add_argument('--images', type=Path, default=Path.home() / 'Desktop/TB/images')
    parser.add_argument('--lut', type=Path, default=Path.home() / 'Desktop/TB/codes/current.lut')
    parser.add_argument('--input', type=Path, help='Photo or folder; defaults to --images')
    parser.add_argument('--template-mode', choices=['lut', '80', '40'], default='80', help='Template white mask: LUT or HSV minimum V')
    parser.add_argument('--output', type=Path, default=Path('sign_test_results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tester = SignTest(args.lut)
    templates = [tester.template(args.images / name, args.template_mode, args.output) for name in ('left.png', 'right.png')]
    source = args.input or args.images
    paths = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'})
    if not paths:
        raise ValueError(f'No images: {source}')
    lines = ['image,candidate,mode,area,aspect,fill,white_ratio,left_score,right_score,result']
    for index, path in enumerate(paths):
        image = read_image(path)
        blue = tester.sign_blue_mask(image)
        stem = f'{index:03d}_{path.stem}'
        save(args.output / f'{stem}_blue.png', blue)
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = image.copy()
        best = {mode: None for mode in ('lut', '80', '40')}
        for number, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect, fill = w / float(h), area / float(w * h)
            geometry = 'area<400' if area < 400 else 'aspect' if not 0.65 <= aspect <= 1.5 else 'fill<0.45' if fill < 0.45 else ''
            if area >= 400:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(annotated, str(number), (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for mode in best:
                arrow = tester.arrow(image[y:y+h, x:x+w], mode)
                ratio = cv2.countNonZero(arrow) / 10000.0
                scores = [float(cv2.matchTemplate(arrow, t, cv2.TM_CCOEFF_NORMED)[0, 0]) if np.std(arrow) else float('nan') for t in templates]
                reason = geometry or ('white_ratio' if not 0.03 <= ratio <= 0.40 else 'nonfinite' if not np.isfinite(scores).all() else 'PASS')
                lines.append(f'{path.name},{number},{mode},{area:.1f},{aspect:.3f},{fill:.3f},{ratio:.4f},{scores[0]:.4f},{scores[1]:.4f},{reason}')
                if area >= 400:
                    save(args.output / f'{stem}_c{number}_{mode}_arrow.png', arrow)
                    save(args.output / f'{stem}_c{number}_crop.png', image[y:y+h, x:x+w])
                if reason == 'PASS' and (best[mode] is None or max(scores) > max(best[mode])):
                    best[mode] = scores
        save(args.output / f'{stem}_boxes.png', annotated)
        print(path.name)
        for mode, scores in best.items():
            print(f'  {mode:>3}: LEFT={scores[0]:.3f} RIGHT={scores[1]:.3f}' if scores is not None else f'  {mode:>3}: no candidate passed filters (see scores.csv)')
        if not contours:
            lines.append(f'{path.name},,,,,,,,,no_blue_contours')
    (args.output / 'scores.csv').write_text('\n'.join(lines) + '\n')
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
