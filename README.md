# 🚀 Touchless Object Detector SDK

Python SDK for a touchless scanner-style object detection workflow, bundled with a local Flask web interface for uploading images and previewing `annotated`, `crop`, `flattened`, `enhanced`, and metadata outputs directly in the browser.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8%20%2F%20YOLO11-0F172A?style=for-the-badge&labelColor=1E293B" alt="Ultralytics YOLO">
  <img src="https://img.shields.io/badge/OpenCV-Scanner%20Pipeline-0EA5E9?style=for-the-badge&labelColor=0369A1" alt="OpenCV">
  <img src="https://img.shields.io/badge/Flask-Local%20Web%20UI-111827?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
</div>

## 📖 Overview

This repository is built for workflows that feel closer to a touchless scanning app than a plain detection demo. The pipeline can:

- detect objects with Ultralytics YOLO
- select the highest-confidence result or a specific target label
- crop the detected area with optional padding
- rectify perspective to create a flatter surface view
- enhance the image for downstream usage
- compute a simple quality score and pixel-based measurements
- preview everything through CLI output or a browser-based interface

This project is a good fit for:

- rapid prototyping of detection + preprocessing pipelines
- validating custom YOLO weights locally
- showing scan outputs to teammates without browsing output folders manually
- building a starting point for future API or camera integrations

## ✨ Features

| Area | Description |
| --- | --- |
| Detection SDK | Load a YOLO model, run inference, and convert detections into Python-friendly structures |
| Scanner Pipeline | Crop, perspective rectification, enhancement, and quality scoring |
| Measurements | Estimate width/height in pixels and optional pixels-per-mm values |
| CLI Demo | Run a full scan from the terminal and export processed images plus metadata |
| Web Frontend | Upload images in the browser and inspect every generated output visually |
| Training | Train custom models with Ultralytics YOLO |
| Export | Export `.pt` weights to `onnx`, `tflite`, or `coreml` |

## 🧰 Requirements

- Python 3.10+
- pip
- A valid YOLO `.pt` model file or one of the built-in Ultralytics preset models

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```text
📦 Touchless Object Detector SDK
├── 📁 configs/                            # Example dataset configuration files for training
│   └── 📄 biometric_data.example.yaml     # Sample Ultralytics dataset YAML for biometric-style data
├── 📁 examples/                           # Example scripts and optional runtime input/output folders
│   └── 📄 demo.py                         # CLI demo that runs a scan and exports processed results
├── 📁 public/                             # Public assets served by the Flask frontend
│   └── 📁 Icon/                           # Browser icon assets
│       └── 🖼️ icon.webp                  # Favicon used by the web interface
├── 📁 sdk/                                # Core SDK package for detection and image processing
│   ├── 📄 __init__.py                     # Package entry point for SDK imports
│   ├── 📄 detector.py                     # ObjectDetectorSDK: inference, target selection, scan pipeline, and export helpers
│   ├── 📄 image_ops.py                    # Image loading, cropping, rectification, enhancement, annotation, and quality scoring
│   └── 📄 types.py                        # Shared dataclasses and type definitions for detections and scan results
├── 📁 webapp/                             # Flask-based web application and API endpoints
│   ├── 📄 __init__.py                     # Marks the frontend folder as a Python package
│   ├── 📄 app.py                          # Flask app factory, `/scan` endpoint, `/health` check, and model preset handling
│   ├── 📁 static/                         # Frontend JavaScript and CSS assets
│   │   ├── 📄 app.js                      # Upload flow, form interactions, API calls, and client-side result rendering
│   │   └── 🎨 styles.css                  # Layout, theme, animations, and custom scrollbar styling
│   └── 📁 templates/                      # HTML templates rendered by Flask
│       └── 📄 index.html                  # Main browser dashboard for uploads, controls, previews, and metadata
├── 📄 export_model.py                     # CLI script for exporting YOLO weights to ONNX, TFLite, or CoreML
├── 📄 train.py                            # CLI script for training Ultralytics YOLO models
├── ⚙️ requirements.txt                   # Python dependencies for the SDK, scripts, and web app
└── 📝 README.md                           # Project documentation
```

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare model weights

The default setup uses `yolov8n.pt`.

- In the web app, you can choose from preset YOLOv8 and YOLO11 models.
- If a selected preset is not available locally, Ultralytics will download it on first use.
- In CLI scripts or Python code, you can also point to your own local `.pt` file.

Example custom weight path:

```text
runs/train/scanner_sdk/weights/best.pt
```

### 3. Run the web frontend

```bash
python -m webapp.app
```

Then open:

```text
http://127.0.0.1:5000
```

From the browser UI, you can:

- upload an image
- choose model preset, device, confidence, image size, and padding ratio
- optionally filter by `target_label`
- optionally provide reference width/height in millimeters
- inspect visual outputs and JSON metadata on a single page

### 4. Run the CLI demo

```bash
python examples/demo.py --image path/to/input.jpg --out-dir examples/output --weights yolov8n.pt
```

If `--image` is omitted, the script tries to use the first supported image inside `examples/input/`.

## 🌐 Web Frontend

The frontend lives in `webapp/` and uses Flask as a lightweight local server.

### Browser outputs

The UI displays:

- original input
- annotated detection
- target crop
- flattened surface
- enhanced output
- detection table
- JSON scan metadata
- measurement cards and quality meter

### Main endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Render the frontend page |
| `/scan` | `POST` | Run inference and return the scan payload as JSON |
| `/health` | `GET` | Simple health check |
| `/public/<path>` | `GET` | Serve public frontend assets |

### Frontend form fields

| Field | Purpose |
| --- | --- |
| `image` | Input image file |
| `weights` | Selected preset model value sent to the backend |
| `device` | `cpu` or `cuda` |
| `conf` | Confidence threshold |
| `imgsz` | Inference image size |
| `padding_ratio` | Extra crop padding around the target |
| `target_label` | Optional label or class ID filter |
| `reference_width_mm` | Optional physical width reference for pixel/mm estimation |
| `reference_height_mm` | Optional physical height reference for pixel/mm estimation |

## ⌨️ SDK and CLI Usage

### Python SDK example

```python
from sdk.detector import ObjectDetectorSDK

sdk = ObjectDetectorSDK(model_path="yolov8n.pt", device="cpu", conf=0.25)
result = sdk.scan(
    image="input.jpg",
    target_label=None,
    size=640,
    padding_ratio=0.12,
    reference_width_mm=18.0,
)

print(result.to_dict())
```

### CLI example

```bash
python examples/demo.py --image path/to/input.jpg --out-dir examples/output --weights yolov8n.pt --device cpu --conf 0.25 --imgsz 640
```

### CLI export files

The demo generates files such as:

- `annotated.jpg`
- `scan_crop.jpg`
- `scan_flattened.jpg`
- `scan_enhanced.jpg`
- `scan_metadata.json`

## 🏋️ Training a Model

Dataset template:

```text
configs/biometric_data.example.yaml
```

Example training command:

```bash
python train.py --data configs/biometric_data.example.yaml --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640 --device cpu
```

Important arguments:

| Argument | Purpose |
| --- | --- |
| `--data` | Dataset YAML file |
| `--model` | Backbone or starting weights |
| `--epochs` | Number of training epochs |
| `--batch` | Batch size |
| `--imgsz` | Training image size |
| `--device` | Training device |
| `--project` | Training output directory |
| `--name` | Experiment name |
| `--patience` | Early stopping patience |
| `--workers` | Data loader worker count |
| `--pretrained` / `--no-pretrained` | Enable or disable pretrained initialization |
| `--single-cls` | Train as a single-class task |

## 📤 Export a Model

Example ONNX export:

```bash
python export_model.py --weights runs/train/scanner_sdk/weights/best.pt --format onnx --imgsz 640 --device cpu
```

Supported export formats:

- `onnx`
- `tflite`
- `coreml`

Optional export flags:

- `--half`
- `--nms`
- `--simplify`

## 🔁 Typical Workflow

```text
Collect dataset -> Label data -> Train detector -> Run scan pipeline -> Review outputs in the web UI -> Export model -> Integrate into your app
```

Example flow:

```bash
python train.py --data configs/biometric_data.example.yaml --model yolov8n.pt --epochs 50
python -m webapp.app
python export_model.py --weights runs/train/scanner_sdk/weights/best.pt --format onnx
```

## 🧠 Output Schema

`ScanResult.to_dict()` returns a compact structure like this:

```json
{
  "image_shape": [720, 1280],
  "detections": [],
  "target": null,
  "crop_shape": null,
  "flattened_shape": null,
  "enhanced_shape": null,
  "quality": 0.0,
  "measurements": {}
}
```

The web frontend wraps that result with additional `summary`, `images`, `measurements`, and `runtime` data to make browser rendering easier.

## 📝 Notes

- Model quality still depends on your dataset and training process.
- The web frontend is designed for local demos and validation, not production deployment.
- If you want to use `cuda`, make sure your Torch environment actually supports GPU inference.
- The Flask app currently limits uploads to 10 MB per image.

## 🩹 Troubleshooting

### Model weights not found

Make sure the `weights` value points to a valid `.pt` file, or choose one of the supported preset model names such as `yolov8n.pt` or `yolo11n.pt`.

### Unsupported or unreadable image

Use one of the supported formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### Frontend is not loading correctly

Start the server from the project root:

```bash
python -m webapp.app
```

Then open `http://127.0.0.1:5000`.

## 🌱 Possible Next Improvements

- add scan history persistence
- expose a separate API endpoint for external integrations
- add evaluation metrics such as mAP and confusion matrix reporting
- add authentication if the web app will be used by multiple users

## 📜 License

Add the license that best matches how you want to distribute this project.
