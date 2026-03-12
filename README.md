# <div align="center">Object Detection SDK (Python)</div>

<div align="center">

Simple YOLOv8-based object detection project with a reusable SDK, training script, model export script, and demo example.

</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8-0F172A?style=for-the-badge&labelColor=1E293B" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-0EA5E9?style=for-the-badge&labelColor=0369A1" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyTorch-Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</div>

---

## Overview

Project ini menyediakan fondasi sederhana untuk workflow object detection:

- SDK Python untuk menjalankan inference dari kode aplikasi
- Script training untuk dataset custom format YOLO/COCO YAML
- Script export model ke format deployment seperti ONNX, TFLite, dan CoreML
- Demo end-to-end untuk membaca gambar, menjalankan deteksi, lalu menyimpan hasil anotasi

README ini dibuat agar cepat dipakai, jadi setiap bagian langsung mengarah ke file dan command yang relevan.

---

## Features

| Area | Keterangan |
| --- | --- |
| Inference SDK | Wrapper `Detector` untuk load model, detect object, dan annotate hasil |
| Training | Training model YOLOv8 custom lewat `train.py` |
| Export | Konversi `.pt` ke `onnx`, `tflite`, atau `coreml` |
| Demo | Contoh pemakaian sederhana lewat `examples/demo.py` |

---

## Tech Stack

- `ultralytics`
- `opencv-python`
- `numpy`
- `torch`

Dependency saat ini ada di `requirements.txt`.

---

## Project Structure

```text
Machine Learning/
|-- examples/
|   `-- demo.py             # Demo inference + simpan gambar hasil anotasi
|-- sdk/
|   `-- detector.py         # Wrapper SDK untuk load model, detect, annotate
|-- export_model.py         # Export model .pt ke onnx / tflite / coreml
|-- train.py                # Training model YOLOv8 untuk dataset custom
|-- requirements.txt        # Daftar dependency Python
`-- README.md               # Dokumentasi proyek
```

---

## Quick Start

### 1. Install dependency

```bash
pip install -r requirements.txt
```

Jika environment Anda belum punya `torch` yang cocok, pasang versi CPU/GPU sesuai perangkat yang dipakai.

### 2. Siapkan model

Gunakan salah satu dari berikut:

- model bawaan seperti `yolov8n.pt`
- hasil training sendiri, misalnya `runs/train/exp/weights/best.pt`

### 3. Jalankan demo

```bash
python examples/demo.py --image path/to/input.jpg --out path/to/output.jpg --weights yolov8n.pt
```

Hasil:

- objek akan dideteksi
- bounding box dan label akan digambar ke gambar output
- daftar deteksi akan dicetak ke terminal

---

## SDK Usage

Class utama ada di `sdk/detector.py`.

### Contoh penggunaan

```python
from sdk.detector import Detector
import cv2

det = Detector(model_path="yolov8n.pt", device="cpu", conf=0.25)

image = cv2.imread("input.jpg")
detections = det.detect(image)
annotated = det.annotate(image, detections)

cv2.imwrite("output.jpg", annotated)
print(detections)
```

### Format output deteksi

Setiap hasil deteksi berbentuk dictionary seperti ini:

```python
{
    "bbox": [x1, y1, x2, y2],
    "conf": 0.91,
    "class": 0,
    "label": "person"
}
```

---

## Training Model

Gunakan `train.py` untuk training pada dataset custom.

### Format dataset

Script mengharapkan file YAML dataset yang menunjuk ke data `train` dan `val`.

### Command

```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 50 --batch 16
```

### Argument utama

| Argument | Fungsi |
| --- | --- |
| `--data` | Path ke file dataset YAML |
| `--model` | Model awal / weight awal |
| `--epochs` | Jumlah epoch training |
| `--batch` | Batch size |

---

## Export Model

Gunakan `export_model.py` setelah training selesai untuk deployment ke platform lain.

### Contoh export ke ONNX

```bash
python export_model.py --weights runs/train/exp/weights/best.pt --format onnx --imgsz 640
```

### Format yang didukung

- `onnx`
- `tflite`
- `coreml`

### Argument utama

| Argument | Fungsi |
| --- | --- |
| `--weights` | Path ke file `.pt` |
| `--format` | Format export |
| `--imgsz` | Ukuran input saat export |

---

## Typical Workflow

```text
Prepare dataset -> Train model -> Test inference -> Export model -> Integrate to app
```

Command ringkas:

```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 50
python examples/demo.py --image sample.jpg --out result.jpg --weights runs/train/exp/weights/best.pt
python export_model.py --weights runs/train/exp/weights/best.pt --format onnx
```

---

## Main Files

| File | Peran |
| --- | --- |
| `sdk/detector.py` | Inti SDK untuk inference dan anotasi |
| `examples/demo.py` | Contoh pemakaian SDK |
| `train.py` | Entry point training |
| `export_model.py` | Entry point export model |
| `requirements.txt` | Dependency proyek |

---

## Notes

- `Detector` bisa menerima image path atau `numpy.ndarray`
- Device default saat ini adalah `cpu`
- Jika device tertentu tidak tersedia, model akan fallback ke perilaku default environment
- Demo mengharapkan file image valid pada path yang diberikan

---

## Next Improvements

- Tambahkan sample `data.yaml`
- Tambahkan `__init__.py` agar struktur package lebih eksplisit
- Tambahkan notebook evaluasi / benchmarking
- Tambahkan contoh integrasi ke Android atau API service

---

## License

Silakan sesuaikan lisensi proyek sesuai kebutuhan distribusi Anda.
