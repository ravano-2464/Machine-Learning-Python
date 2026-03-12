"""Train a YOLO model on a custom dataset using ultralytics.

Dataset should be in YOLO/COCO format. Provide a YAML file describing train/val paths.

Example:
    python train.py --data data.yaml --model yolov8n.pt --epochs 50
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to data YAML (train/val)')
    p.add_argument('--model', default='yolov8n.pt', help='Backbone or weights')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch)


if __name__ == '__main__':
    main()
