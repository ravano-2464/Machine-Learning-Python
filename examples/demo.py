import argparse
import json
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdk.detector import ObjectDetectorSDK


def resolve_image_path(image_arg: str | None) -> Path:
    if image_arg:
        image_path = Path(image_arg)
        if image_path.exists():
            return image_path
        raise SystemExit(
            f"Image tidak ditemukan: {image_path}\n"
            f"Contoh pakai:\n"
            f"python examples\\demo.py --image C:\\path\\ke\\gambar.jpg --out-dir examples\\output --weights yolov8n.pt"
        )

    input_dir = PROJECT_ROOT / "examples" / "input"
    supported_files = []
    if input_dir.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            supported_files.extend(sorted(input_dir.glob(pattern)))
    if supported_files:
        return supported_files[0]

    raise SystemExit(
        "Argumen --image belum diisi dan tidak ada gambar di folder examples\\input.\n"
        "Solusi:\n"
        "1. Jalankan dengan --image C:\\path\\ke\\gambar.jpg\n"
        "2. Atau simpan satu gambar ke examples\\input lalu tekan Run lagi"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path gambar input")
    parser.add_argument("--out-dir", default="examples/output", help="Folder hasil output")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path model weights")
    parser.add_argument("--target-label", default=None, help="Label target seperti single_finger atau palm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--padding-ratio", type=float, default=0.12)
    parser.add_argument("--reference-width-mm", type=float, default=None)
    parser.add_argument("--reference-height-mm", type=float, default=None)
    args = parser.parse_args()

    image_path = resolve_image_path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Gagal membaca image: {image_path}")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdk = ObjectDetectorSDK(model_path=args.weights, device=args.device, conf=args.conf)
    result = sdk.scan(
        image=image,
        target_label=args.target_label,
        size=args.imgsz,
        padding_ratio=args.padding_ratio,
        reference_width_mm=args.reference_width_mm,
        reference_height_mm=args.reference_height_mm,
    )
    annotated = sdk.annotate(image, result.detections)
    annotated_path = output_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    exported = sdk.export_scan(result, output_dir=output_dir, prefix="scan")

    payload = result.to_dict()
    payload["annotated"] = str(annotated_path)
    payload["exports"] = exported
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
