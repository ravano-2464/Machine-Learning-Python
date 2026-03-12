"""Export a trained YOLO model to ONNX / TFLite / CoreML for mobile use.

Example:
    python export_model.py --weights runs/train/exp/weights/best.pt --format onnx
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to .pt weights')
    p.add_argument('--format', default='onnx', choices=['onnx', 'tflite', 'coreml'])
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.weights)
    print('Exporting to', args.format)
    model.export(format=args.format, imgsz=args.imgsz)


if __name__ == '__main__':
    main()
