from __future__ import annotations

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from .image_ops import annotate_detections, crop_image, enhance_image, load_image, quality_score, rectify_image
from .types import Detection, ImageInput, ScanResult


class ObjectDetectorSDK:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", conf: float = 0.25):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            pass

    def _predict(self, image, size: int, max_det: int):
        kwargs = {
            "source": image,
            "imgsz": size,
            "conf": self.conf,
            "verbose": False,
            "max_det": max_det,
        }
        if self.device:
            kwargs["device"] = self.device
        return self.model.predict(**kwargs)

    def _label_for_class(self, class_id: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def detect(self, image: ImageInput, size: int = 640, max_det: int = 25) -> list[Detection]:
        source = load_image(image)
        results = self._predict(source, size=size, max_det=max_det)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=confidence,
                        class_id=class_id,
                        label=self._label_for_class(class_id),
                    )
                )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def detect_as_dict(self, image: ImageInput, size: int = 640, max_det: int = 25) -> list[dict]:
        return [item.to_dict() for item in self.detect(image=image, size=size, max_det=max_det)]

    def select_target(self, detections: list[Detection], target_label: str | int | None = None) -> Detection | None:
        if not detections:
            return None
        if target_label is None:
            return detections[0]
        for detection in detections:
            if detection.label == str(target_label):
                return detection
            if str(detection.class_id) == str(target_label):
                return detection
        return None

    def measure(self, detection: Detection, reference_width_mm: float | None = None, reference_height_mm: float | None = None) -> dict[str, float]:
        x1, y1, x2, y2 = detection.bbox
        width_px = max(0.0, float(x2 - x1))
        height_px = max(0.0, float(y2 - y1))
        measurements = {
            "width_px": width_px,
            "height_px": height_px,
        }
        if reference_width_mm and reference_width_mm > 0:
            measurements["pixels_per_mm_width"] = width_px / reference_width_mm
        if reference_height_mm and reference_height_mm > 0:
            measurements["pixels_per_mm_height"] = height_px / reference_height_mm
        return measurements

    def scan(
        self,
        image: ImageInput,
        target_label: str | int | None = None,
        size: int = 640,
        padding_ratio: float = 0.12,
        flatten: bool = True,
        enhance: bool = True,
        reference_width_mm: float | None = None,
        reference_height_mm: float | None = None,
    ) -> ScanResult:
        source = load_image(image)
        detections = self.detect(source, size=size)
        target = self.select_target(detections, target_label=target_label)
        measurements = self.measure(target, reference_width_mm, reference_height_mm) if target else {}
        if target is None:
            return ScanResult(
                detections=detections,
                image_shape=(int(source.shape[0]), int(source.shape[1])),
                target=None,
                crop=None,
                flattened=None,
                enhanced=None,
                quality=0.0,
                measurements=measurements,
            )
        crop = crop_image(source, target.bbox, padding_ratio=padding_ratio)
        if crop.size == 0:
            return ScanResult(
                detections=detections,
                image_shape=(int(source.shape[0]), int(source.shape[1])),
                target=target,
                crop=None,
                flattened=None,
                enhanced=None,
                quality=0.0,
                measurements=measurements,
            )
        flattened = rectify_image(crop) if flatten else crop.copy()
        enhanced_image = enhance_image(flattened) if enhance else None
        processed = enhanced_image if enhanced_image is not None else flattened
        return ScanResult(
            detections=detections,
            image_shape=(int(source.shape[0]), int(source.shape[1])),
            target=target,
            crop=crop,
            flattened=flattened,
            enhanced=enhanced_image,
            quality=quality_score(source, target.bbox, processed),
            measurements=measurements,
        )

    def annotate(self, image: ImageInput, detections: list[Detection], box_color=(0, 255, 0), thickness: int = 2):
        source = load_image(image)
        return annotate_detections(source, detections, box_color=box_color, thickness=thickness)

    def export_scan(self, result: ScanResult, output_dir: str | Path, prefix: str = "scan") -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        exported: dict[str, str] = {}
        if result.crop is not None:
            crop_path = output_path / f"{prefix}_crop.jpg"
            cv2.imwrite(str(crop_path), result.crop)
            exported["crop"] = str(crop_path)
        if result.flattened is not None:
            flattened_path = output_path / f"{prefix}_flattened.jpg"
            cv2.imwrite(str(flattened_path), result.flattened)
            exported["flattened"] = str(flattened_path)
        if result.enhanced is not None:
            enhanced_path = output_path / f"{prefix}_enhanced.jpg"
            cv2.imwrite(str(enhanced_path), result.enhanced)
            exported["enhanced"] = str(enhanced_path)
        metadata_path = output_path / f"{prefix}_metadata.json"
        metadata_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        exported["metadata"] = str(metadata_path)
        return exported


Detector = ObjectDetectorSDK
