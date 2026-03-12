"""Simple demo showing SDK usage and saving annotated image."""
from sdk.detector import Detector
import cv2
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='examples/test.jpg', help='Input image path')
    p.add_argument('--out', default='examples/out.jpg', help='Output image path')
    p.add_argument('--weights', default='yolov8n.pt', help='Weights path')
    args = p.parse_args()

    det = Detector(model_path=args.weights)
    img = cv2.imread(args.image)
    if img is None:
        print('Image not found:', args.image)
        return
    dets = det.detect(img)
    ann = det.annotate(img, dets)
    cv2.imwrite(args.out, ann)
    print('Detections:', dets)
    print('Saved annotated image to', args.out)


if __name__ == '__main__':
    main()
