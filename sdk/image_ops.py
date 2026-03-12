from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import Detection, ImageInput


def load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image))
        if loaded is None:
            raise FileNotFoundError(f"Image not found: {image}")
        return loaded
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a file path or numpy.ndarray")
    return image.copy()


def crop_image(image: np.ndarray, bbox: tuple[float, float, float, float], padding_ratio: float = 0.0) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    return image[top:bottom, left:right].copy()


def order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sums)]
    ordered[2] = points[np.argmax(point_sums)]
    ordered[1] = points[np.argmin(point_diffs)]
    ordered[3] = points[np.argmax(point_diffs)]
    return ordered


def find_surface_quad(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image.shape[0] * image.shape[1] * 0.12
    best_quad = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area
    if best_quad is not None:
        return order_points(best_quad)
    if contours:
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        return order_points(cv2.boxPoints(rect).astype(np.float32))
    height, width = image.shape[:2]
    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


def warp_from_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = order_points(quad)
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_width = max(int(round(max(width_a, width_b))), 1)
    max_height = max(int(round(max(height_a, height_b))), 1)
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(order_points(quad), destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def rectify_image(image: np.ndarray) -> np.ndarray:
    quad = find_surface_quad(image)
    return warp_from_quad(image, quad)


def enhance_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    denoised = cv2.GaussianBlur(normalized, (3, 3), 0)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    value = cv2.Laplacian(gray, cv2.CV_64F).var() / 1500.0
    return float(np.clip(value, 0.0, 1.0))


def center_score(image_shape: tuple[int, int], bbox: tuple[float, float, float, float]) -> float:
    image_height, image_width = image_shape
    x1, y1, x2, y2 = bbox
    bbox_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    image_center = np.array([image_width / 2.0, image_height / 2.0], dtype=np.float32)
    max_distance = np.linalg.norm(np.array([image_width / 2.0, image_height / 2.0], dtype=np.float32))
    distance = np.linalg.norm(bbox_center - image_center)
    if max_distance == 0:
        return 0.0
    return float(np.clip(1.0 - (distance / max_distance), 0.0, 1.0))


def area_score(image_shape: tuple[int, int], bbox: tuple[float, float, float, float]) -> float:
    image_height, image_width = image_shape
    x1, y1, x2, y2 = bbox
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = max(1.0, float(image_height * image_width))
    return float(np.clip((bbox_area / image_area) / 0.35, 0.0, 1.0))


def quality_score(source: np.ndarray, bbox: tuple[float, float, float, float], processed: np.ndarray) -> float:
    image_shape = (int(source.shape[0]), int(source.shape[1]))
    score = (0.45 * sharpness_score(processed)) + (0.35 * center_score(image_shape, bbox)) + (0.20 * area_score(image_shape, bbox))
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def annotate_detections(
    image: np.ndarray,
    detections: list[Detection],
    box_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        label = f"{detection.label}:{detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        cv2.putText(
            annotated,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            1,
            cv2.LINE_AA,
        )
    return annotated
