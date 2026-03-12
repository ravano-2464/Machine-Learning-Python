import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path ke file .pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "tflite", "coreml"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    print("Exporting to", args.format)
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        nms=args.nms,
        simplify=args.simplify,
    )


if __name__ == "__main__":
    main()
