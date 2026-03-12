from ultralytics import YOLO
import cv2
import numpy as np


class Detector:
    """Simple object detection SDK wrapper using Ultralytics YOLO.

    Usage:
        det = Detector(model_path='yolov8n.pt', device='cpu', conf=0.25)
        results = det.detect(image)  # image = numpy array or path
        annotated = det.annotate(image, results)
    """

    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu', conf: float = 0.25):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            # some environments handle device automatically
            pass

    def detect(self, image, size: int = 640):
        """Run detection on an image (ndarray) or image path.

        Returns list of detections: {bbox:[x1,y1,x2,y2], conf:float, class:int, label:str}
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image}")
        else:
            img = image

        results = self.model.predict(source=img, imgsz=size, conf=self.conf, verbose=False)
        out = []
        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                # box.xyxy, box.conf, box.cls
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = self.model.names.get(cls, str(cls))
                out.append({'bbox': [float(x) for x in xyxy], 'conf': conf, 'class': cls, 'label': label})
        return out

    def annotate(self, image, detections, box_color=(0, 255, 0), thickness: int = 2):
        """Draw detections onto the image and return annotated image (copy)."""
        img = image.copy() if isinstance(image, np.ndarray) else cv2.imread(image)
        for d in detections:
            x1, y1, x2, y2 = map(int, d['bbox'])
            label = f"{d['label']}:{d['conf']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
            cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        return img
