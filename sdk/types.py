from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ImageInput = str | Path | np.ndarray


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": float(self.confidence),
            "class": int(self.class_id),
            "label": self.label,
        }


@dataclass(slots=True)
class ScanResult:
    detections: list[Detection]
    image_shape: tuple[int, int]
    target: Detection | None = None
    crop: np.ndarray | None = None
    flattened: np.ndarray | None = None
    enhanced: np.ndarray | None = None
    quality: float | None = None
    measurements: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_shape": [int(self.image_shape[0]), int(self.image_shape[1])],
            "detections": [item.to_dict() for item in self.detections],
            "target": self.target.to_dict() if self.target else None,
            "crop_shape": list(self.crop.shape[:2]) if self.crop is not None else None,
            "flattened_shape": list(self.flattened.shape[:2]) if self.flattened is not None else None,
            "enhanced_shape": list(self.enhanced.shape[:2]) if self.enhanced is not None else None,
            "quality": self.quality,
            "measurements": self.measurements,
        }
