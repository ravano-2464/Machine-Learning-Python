from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdk.detector import ObjectDetectorSDK

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WEIGHTS = "yolov8n.pt"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


def normalize_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.exists():
        return str(candidate.resolve())
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return str(project_candidate.resolve())
    return model_path


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def parse_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Gambar tidak bisa dibaca. Gunakan JPG, PNG, BMP, atau WEBP.")
    return image


def encode_image_to_data_url(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@lru_cache(maxsize=6)
def get_sdk(model_path: str, device: str, conf: float) -> ObjectDetectorSDK:
    return ObjectDetectorSDK(model_path=model_path, device=device, conf=conf)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
    default_config = {
        "weights": DEFAULT_WEIGHTS,
        "device": "cpu",
        "conf": 0.25,
        "imgsz": 640,
        "padding_ratio": 0.12,
    }

    @app.get("/")
    def index():
        return render_template("index.html", config=default_config)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/scan")
    def scan():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Pilih file gambar terlebih dahulu."}), 400

        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Format file belum didukung. Gunakan JPG, PNG, BMP, atau WEBP."}), 400

        try:
            file_bytes = uploaded.read()
            image = decode_uploaded_image(file_bytes)
            weights = normalize_model_path(request.form.get("weights", DEFAULT_WEIGHTS).strip() or DEFAULT_WEIGHTS)
            device = request.form.get("device", "cpu").strip() or "cpu"
            conf = round(clamp(float(request.form.get("conf", 0.25)), 0.01, 0.99), 4)
            imgsz = max(128, min(1280, int(request.form.get("imgsz", 640))))
            padding_ratio = clamp(float(request.form.get("padding_ratio", 0.12)), 0.0, 0.5)
            target_label = request.form.get("target_label", "").strip() or None
            reference_width_mm = parse_optional_float(request.form.get("reference_width_mm"))
            reference_height_mm = parse_optional_float(request.form.get("reference_height_mm"))

            sdk = get_sdk(weights, device, conf)
            result = sdk.scan(
                image=image,
                target_label=target_label,
                size=imgsz,
                padding_ratio=padding_ratio,
                reference_width_mm=reference_width_mm,
                reference_height_mm=reference_height_mm,
            )
            annotated = sdk.annotate(image, result.detections)

            payload = {
                "summary": {
                    "image_shape": result.to_dict()["image_shape"],
                    "detections_count": len(result.detections),
                    "target_label": result.target.label if result.target else None,
                    "target_confidence": round(result.target.confidence, 4) if result.target else None,
                    "quality": result.quality,
                    "status": "Target ditemukan dan berhasil diproses." if result.target else "Belum ada target yang cocok pada gambar ini.",
                },
                "detections": [item.to_dict() for item in result.detections],
                "measurements": result.measurements,
                "scan": result.to_dict(),
                "images": {
                    "original": encode_image_to_data_url(image),
                    "annotated": encode_image_to_data_url(annotated),
                    "crop": encode_image_to_data_url(result.crop),
                    "flattened": encode_image_to_data_url(result.flattened),
                    "enhanced": encode_image_to_data_url(result.enhanced),
                },
            }
            return jsonify(payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify({"error": "Ukuran file terlalu besar. Maksimal 10 MB."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
