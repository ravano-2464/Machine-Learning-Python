from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "public"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdk.detector import ObjectDetectorSDK

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WEIGHTS = "yolov8n.pt"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MODEL_PRESETS = (
    {
        "value": "yolov8n.pt",
        "label": "YOLOv8 Nano",
        "description": "Paling ringan untuk demo cepat dan device CPU.",
    },
    {
        "value": "yolov8s.pt",
        "label": "YOLOv8 Small",
        "description": "Sedikit lebih akurat dengan beban inferensi masih ringan.",
    },
    {
        "value": "yolov8m.pt",
        "label": "YOLOv8 Medium",
        "description": "Seimbang untuk kebutuhan akurasi dan performa.",
    },
    {
        "value": "yolov8l.pt",
        "label": "YOLOv8 Large",
        "description": "Model besar untuk detail deteksi yang lebih kuat.",
    },
    {
        "value": "yolov8x.pt",
        "label": "YOLOv8 XLarge",
        "description": "Varian terbesar YOLOv8 untuk akurasi maksimum.",
    },
    {
        "value": "yolo11n.pt",
        "label": "YOLO11 Nano",
        "description": "Preset generasi YOLO11 paling ringan untuk inferensi cepat.",
    },
    {
        "value": "yolo11s.pt",
        "label": "YOLO11 Small",
        "description": "YOLO11 small untuk akurasi lebih baik dengan beban tetap efisien.",
    },
    {
        "value": "yolo11m.pt",
        "label": "YOLO11 Medium",
        "description": "Varian menengah YOLO11 untuk skenario inference yang seimbang.",
    },
    {
        "value": "yolo11l.pt",
        "label": "YOLO11 Large",
        "description": "YOLO11 large saat butuh detail deteksi yang lebih kuat.",
    },
    {
        "value": "yolo11x.pt",
        "label": "YOLO11 XLarge",
        "description": "Preset YOLO11 terbesar untuk fokus ke akurasi maksimum.",
    },
)


def normalize_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.exists():
        return str(candidate.resolve())
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return str(project_candidate.resolve())
    return model_path


def model_exists_locally(model_path: str) -> bool:
    candidate = Path(model_path)
    if candidate.is_file():
        return True
    return (PROJECT_ROOT / candidate).is_file()


def is_builtin_downloadable_model(model_path: str) -> bool:
    model_path = model_path.strip()
    return Path(model_path).name == model_path and model_path in GITHUB_ASSETS_NAMES


def build_model_presets() -> list[dict[str, str | bool]]:
    presets: list[dict[str, str | bool]] = []
    for preset in MODEL_PRESETS:
        is_local = model_exists_locally(preset["value"])
        availability = "Tersedia lokal" if is_local else "Akan diunduh otomatis saat pertama kali dipakai"
        presets.append(
            {
                **preset,
                "is_local": is_local,
                "availability": availability,
            }
        )
    return presets


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


@lru_cache(maxsize=12)
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

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/")
    def index():
        return render_template("index.html", config=default_config, model_presets=build_model_presets())

    @app.get("/public/<path:filename>")
    def public_file(filename: str):
        return send_from_directory(PUBLIC_ROOT, filename)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/scan", methods=["POST", "OPTIONS"])
    def scan():
        if request.method == "OPTIONS":
            return ("", 204)

        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Pilih file gambar terlebih dahulu."}), 400

        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Format file belum didukung. Gunakan JPG, PNG, BMP, atau WEBP."}), 400

        try:
            file_bytes = uploaded.read()
            image = decode_uploaded_image(file_bytes)
            raw_weights = request.form.get("weights", DEFAULT_WEIGHTS).strip() or DEFAULT_WEIGHTS
            if not model_exists_locally(raw_weights) and not is_builtin_downloadable_model(raw_weights):
                supported_presets = ", ".join(preset["value"] for preset in MODEL_PRESETS)
                raise FileNotFoundError(
                    f"Weights '{raw_weights}' tidak ditemukan. Isi path `.pt` yang valid atau gunakan salah satu preset ini: {supported_presets}."
                )
            weights = normalize_model_path(raw_weights)
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
            scan_payload = result.to_dict()
            scan_payload["runtime"] = {
                "weights": raw_weights,
                "resolved_weights": weights,
                "device": device,
                "conf": conf,
                "imgsz": imgsz,
                "padding_ratio": padding_ratio,
                "target_label": target_label,
                "reference_width_mm": reference_width_mm,
                "reference_height_mm": reference_height_mm,
            }

            payload = {
                "summary": {
                    "image_shape": result.to_dict()["image_shape"],
                    "detections_count": len(result.detections),
                    "target_label": result.target.label if result.target else None,
                    "target_confidence": round(result.target.confidence, 4) if result.target else None,
                    "quality": result.quality,
                    "weights": raw_weights,
                    "status": "Target ditemukan dan berhasil diproses." if result.target else "Belum ada target yang cocok pada gambar ini.",
                },
                "detections": [item.to_dict() for item in result.detections],
                "measurements": result.measurements,
                "scan": scan_payload,
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

    @app.errorhandler(Exception)
    def unexpected_error(error):
        return jsonify({"error": f"Server error: {error}"}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
