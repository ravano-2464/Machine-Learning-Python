# 🚀 Touchless Object Detector SDK

Python SDK untuk workflow object detection bergaya scanner, lengkap dengan web frontend untuk upload gambar dan melihat hasil `annotated`, `crop`, `flattened`, `enhanced`, serta metadata scan langsung di browser.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8-0F172A?style=for-the-badge&labelColor=1E293B" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Scanner%20Pipeline-0EA5E9?style=for-the-badge&labelColor=0369A1" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyTorch-Custom%20Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</div>

## 📖 Overview

Project ini berfokus pada alur yang lebih dekat ke aplikasi scanner touchless:

- deteksi objek target dengan YOLOv8
- pilih target utama berdasarkan confidence
- crop area target dengan padding opsional
- flatten perspective agar permukaan lebih rapi
- enhance image untuk hasil yang lebih siap dipakai downstream
- hitung quality score dan measurement sederhana
- tampilkan output lewat CLI atau frontend web

Repo ini cocok untuk:

- prototyping alur deteksi + preprocessing gambar
- validasi hasil model custom
- demo lokal ke user atau stakeholder tanpa buka folder output manual
- dasar integrasi ke API atau aplikasi kamera

## ✨ Features

| Area | Keterangan |
| --- | --- |
| Detection SDK | Load model, jalankan inference, dan ubah hasil ke format Python dict |
| Scanner Pipeline | Crop, perspective rectification, enhancement, quality scoring |
| Measurements | Hitung lebar/tinggi pixel dan estimasi pixel per mm |
| CLI Demo | Jalankan scan dari terminal dan export file hasil |
| Web Frontend | Upload gambar via browser dan lihat semua output scan |
| Training | Train model custom berbasis Ultralytics YOLO |
| Export | Export weights `.pt` ke `onnx`, `tflite`, atau `coreml` |

Install dependency:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```text
Machine Learning/
|-- configs/
|   `-- biometric_data.example.yaml
|-- examples/
|   `-- demo.py
|-- sdk/
|   |-- __init__.py
|   |-- detector.py
|   |-- image_ops.py
|   `-- types.py
|-- webapp/
|   |-- __init__.py
|   |-- app.py
|   |-- static/
|   |   |-- app.js
|   |   `-- styles.css
|   `-- templates/
|       `-- index.html
|-- export_model.py
|-- train.py
|-- requirements.txt
|-- README.md
`-- yolov8n.pt
```

## ⚡ Quick Start

### 1. 📦 Install dependency

```bash
pip install -r requirements.txt
```

### 2. 🤖 Siapkan model weights

Default repo ini memakai `yolov8n.pt`, tetapi Anda juga bisa memakai hasil training sendiri, misalnya:

```text
runs/train/scanner_sdk/weights/best.pt
```

### 3. 🌐 Jalankan lewat frontend web

```bash
python -m webapp.app
```

Setelah server aktif, buka:

```text
http://127.0.0.1:5000
```

Yang bisa dilakukan dari frontend:

- upload gambar melalui browser
- atur `weights`, `device`, `confidence`, `image size`, dan `padding ratio`
- isi `target label` jika hanya ingin label tertentu
- isi `reference width/height` untuk menghitung estimasi pixel per mm
- lihat output visual dan metadata langsung di halaman

### 4. 💻 Jalankan lewat CLI

```bash
python examples/demo.py --image path/to/input.jpg --out-dir examples/output --weights yolov8n.pt
```

Jika Anda menjalankan tanpa `--image`, script akan mencoba mengambil gambar pertama dari folder `examples/input/`.

## 🎨 Web Frontend

Frontend ada di folder `webapp/` dan memakai Flask sebagai backend ringan.

### 🖼️ Tampilan hasil di browser

Web UI akan menampilkan:

- original input
- annotated detection
- crop target
- flattened surface
- enhanced output
- tabel deteksi
- JSON metadata hasil scan
- measurement cards dan quality meter

### 🔌 Endpoint utama

| Endpoint | Method | Fungsi |
| --- | --- | --- |
| `/` | `GET` | Render halaman frontend |
| `/scan` | `POST` | Jalankan inference dan mengembalikan JSON hasil scan |
| `/health` | `GET` | Health check sederhana |

### 🧾 Parameter form frontend

| Field | Fungsi |
| --- | --- |
| `image` | File gambar input |
| `weights` | Path weights model |
| `device` | `cpu` atau `cuda` |
| `conf` | Confidence threshold |
| `imgsz` | Ukuran input inference |
| `padding_ratio` | Padding tambahan saat crop |
| `target_label` | Filter target berdasarkan label atau class id |
| `reference_width_mm` | Referensi lebar fisik untuk estimasi pixel/mm |
| `reference_height_mm` | Referensi tinggi fisik untuk estimasi pixel/mm |

## ⌨️ CLI Usage

Contoh pemakaian SDK di Python:

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

Contoh CLI lengkap:

```bash
python examples/demo.py --image path/to/input.jpg --out-dir examples/output --weights yolov8n.pt --device cpu --conf 0.25 --imgsz 640
```

Output file CLI:

- `annotated.jpg`
- `scan_crop.jpg`
- `scan_flattened.jpg`
- `scan_enhanced.jpg`
- `scan_metadata.json`

## 🏋️ Training Model

Template dataset ada di:

`configs/biometric_data.example.yaml`

Contoh training:

```bash
python train.py --data configs/biometric_data.example.yaml --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640 --device cpu
```

Argumen penting:

| Argument | Fungsi |
| --- | --- |
| `--data` | File dataset YAML |
| `--model` | Backbone atau weight awal |
| `--epochs` | Jumlah epoch training |
| `--batch` | Batch size |
| `--imgsz` | Ukuran input training |
| `--device` | Device training |
| `--project` | Folder output training |
| `--name` | Nama eksperimen |

## 📤 Export Model

Contoh export ke ONNX:

```bash
python export_model.py --weights runs/train/scanner_sdk/weights/best.pt --format onnx --imgsz 640 --device cpu
```

Format export yang didukung:

- `onnx`
- `tflite`
- `coreml`

## 🔁 Typical Workflow

```text
Collect dataset -> Label dataset -> Train detector -> Run scan pipeline -> Review output in frontend -> Export model -> Integrate to app
```

Contoh alur ringkas:

```bash
python train.py --data configs/biometric_data.example.yaml --model yolov8n.pt --epochs 50
python -m webapp.app
python export_model.py --weights runs/train/scanner_sdk/weights/best.pt --format onnx
```

## 🧠 Output Schema

`ScanResult.to_dict()` mengembalikan struktur ringkas seperti berikut:

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

Frontend akan membungkus hasil tersebut dengan `summary`, `images`, dan `detections` agar mudah dirender di browser.

## 📝 Notes

- Akurasi model tetap bergantung pada dataset dan training Anda
- Frontend ini dirancang untuk demo lokal dan validasi hasil, belum untuk deployment production
- Jika memakai `cuda`, pastikan environment Torch Anda memang mendukung GPU
- Default maksimal ukuran upload di web app adalah 10 MB per gambar

## 🩹 Troubleshooting

### ❌ Model tidak ditemukan

Pastikan nilai `weights` mengarah ke file `.pt` yang valid, misalnya `yolov8n.pt` atau `runs/train/scanner_sdk/weights/best.pt`.

### 🖼️ Gambar gagal dibaca

Gunakan salah satu format yang didukung:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### 🌐 Frontend tidak tampil

Pastikan server dijalankan dari root project:

```bash
python -m webapp.app
```

Lalu buka `http://127.0.0.1:5000`.

## 🌱 Next Improvements

- tambahkan penyimpanan history scan
- tambahkan endpoint API terpisah untuk integrasi eksternal
- tambahkan evaluasi metrik seperti mAP dan confusion matrix
- tambahkan autentikasi jika web app dipakai multi-user

## 📜 License

Silakan tambahkan lisensi sesuai kebutuhan distribusi project Anda.
