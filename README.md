# <div align="center">🚀 Touchless Object Detector SDK</div>

<div align="center">

Python SDK untuk workflow object detection bergaya scanner: deteksi target, crop area, flatten perspective, enhance image, lalu export hasil.

</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8-0F172A?style=for-the-badge&labelColor=1E293B" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Scanner%20Pipeline-0EA5E9?style=for-the-badge&labelColor=0369A1" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyTorch-Custom%20Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</div>

---

## 🎯 Overview

Repo ini sekarang diarahkan untuk use case SDK yang lebih dekat ke aplikasi scanner touchless:

- deteksi objek target dengan YOLO
- pilih target terbaik
- crop area target
- flatten perspective agar area lebih rapi
- enhance hasil crop agar siap diproses lebih lanjut
- export metadata dan image hasil scan

Fokusnya bukan sekadar bounding box, tetapi pipeline yang lebih cocok untuk integrasi aplikasi scanning berbasis kamera.

---

## ✨ Features

| Area | Keterangan |
| --- | --- |
| Detection SDK | Load model, detect object, pilih target utama |
| Scanner Pipeline | Crop, perspective rectification, enhancement, quality scoring |
| Measurements | Hitung ukuran pixel dan estimasi pixel per mm |
| Training | Training custom detector dengan dataset sendiri |
| Export | Export model `.pt` ke `onnx`, `tflite`, atau `coreml` |
| Demo | CLI demo untuk inference dan export hasil scan |

---

## 🛠️ Tech Stack

- `ultralytics`
- `opencv-python`
- `numpy`
- `torch`

Install dependency:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```text
Machine Learning/
|-- configs/
|   `-- biometric_data.example.yaml     # Template dataset YAML untuk training custom
|-- examples/
|   `-- demo.py                         # Demo CLI untuk detect, scan, dan export hasil
|-- sdk/
|   |-- __init__.py                     # Public export untuk package SDK
|   |-- detector.py                     # Class utama ObjectDetectorSDK
|   |-- image_ops.py                    # Utility crop, rectify, enhance, annotate, scoring
|   `-- types.py                        # Dataclass Detection dan ScanResult
|-- export_model.py                     # Script export model ke onnx / tflite / coreml
|-- train.py                            # Script training model YOLO custom
|-- requirements.txt                    # Dependency Python project
|-- .gitignore                          # Ignore cache, weights, runs, dan output lokal
`-- README.md                           # Dokumentasi utama project
```

### 🗂️ Breakdown Struktur

| Path | Isi |
| --- | --- |
| `configs/` | Menyimpan template atau konfigurasi dataset/training |
| `examples/` | Contoh entry point untuk menjalankan SDK dari CLI |
| `sdk/` | Package inti yang berisi logic object detection scanner-style |
| `sdk/detector.py` | Orkestrasi inference, target selection, measurement, scan pipeline, dan export |
| `sdk/image_ops.py` | Operasi citra level rendah untuk preprocessing dan postprocessing |
| `sdk/types.py` | Struktur data hasil deteksi dan hasil scan agar output konsisten |
| `train.py` | Menjalankan training model custom berbasis Ultralytics YOLO |
| `export_model.py` | Menyiapkan model hasil training untuk deployment lintas platform |
| `.gitignore` | Menjaga file artefak lokal tidak ikut ke version control |

### 📄 Main Files

| File | Fungsi |
| --- | --- |
| `sdk/detector.py` | SDK utama untuk detect, scan, annotate, dan export |
| `sdk/image_ops.py` | Utility image pipeline untuk crop, flatten, enhancement, scoring |
| `sdk/types.py` | Tipe data hasil deteksi dan scan |
| `examples/demo.py` | Contoh pemakaian CLI |
| `train.py` | Training custom detector |
| `export_model.py` | Export model ke format deployment |
| `configs/biometric_data.example.yaml` | Template dataset YAML |

---

## ⚡ Quick Start

### 1. 📦 Siapkan dependency

```bash
pip install -r requirements.txt
```

Jika `torch` belum cocok dengan device Anda, pasang versi CPU/GPU yang sesuai.

### 2. 🤖 Siapkan weights

Gunakan salah satu dari berikut:

- `yolov8n.pt`
- hasil training sendiri, misalnya `runs/train/scanner_sdk/weights/best.pt`

### 3. ▶️ Jalankan demo scanner

```bash
python examples/demo.py --image path/to/input.jpg --out-dir examples/output --weights yolov8n.pt
```

Jika Anda menjalankan script lewat tombol Run di editor tanpa argumen, simpan dulu satu gambar di folder `examples/input/`. Script akan otomatis memakai gambar pertama dari folder tersebut.

Output yang dihasilkan:

- `annotated.jpg`
- `scan_crop.jpg`
- `scan_flattened.jpg`
- `scan_enhanced.jpg`
- `scan_metadata.json`

---

## 🧠 SDK Usage

Class utama ada di `sdk.detector.ObjectDetectorSDK`.

### 💡 Contoh penggunaan

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

### 🧾 Hasil scan

Result akan berisi:

- daftar semua deteksi
- target utama yang dipilih
- ukuran image sumber
- crop hasil target
- image yang sudah di-flatten
- image yang sudah di-enhance
- quality score
- measurement metadata

---

## 🏋️ Training Model

Repo ini sudah menyiapkan pipeline training untuk model object detector custom, tetapi model final tetap harus dilatih memakai dataset target Anda sendiri.

### 🗂️ Template dataset

Template awal ada di:

```text
configs/biometric_data.example.yaml
```

Isi kelas default saat ini:

- `single_finger`
- `multi_finger`
- `palm`

Anda bisa ubah nama kelas sesuai objek target Anda.

### 💻 Command training

```bash
python train.py --data configs/biometric_data.example.yaml --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640 --device cpu
```

### 📌 Argument penting

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

---

## 📤 Export Model

Setelah training selesai, export model ke format deployment.

### 🔄 Contoh export ke ONNX

```bash
python export_model.py --weights runs/train/scanner_sdk/weights/best.pt --format onnx --imgsz 640 --device cpu
```

### 📚 Format export

- `onnx`
- `tflite`
- `coreml`

---

## 🔁 Typical Workflow

```text
Collect dataset -> Label dataset -> Train detector -> Run scanner pipeline -> Export model -> Integrate to app
```

Command ringkas:

```bash
python train.py --data configs/biometric_data.example.yaml --model yolov8n.pt --epochs 50
python examples/demo.py --image sample.jpg --out-dir examples/output --weights runs/train/scanner_sdk/weights/best.pt
python export_model.py --weights runs/train/scanner_sdk/weights/best.pt --format onnx
```

---

## 📝 Notes

- SDK ini sudah punya alur yang lebih dekat ke aplikasi scanner touchless, tetapi belum bisa menghasilkan model akurat tanpa dataset training yang sesuai domain Anda
- Jika target Anda spesifik seperti jari, telapak, dokumen, atau komponen tertentu, dataset harus mengikuti objek tersebut
- `scan_metadata.json` menyimpan ringkasan hasil scan untuk integrasi downstream
- Semua file Python di repo ini sudah dibersihkan dari komentar dan docstring

---

## 🌱 Next Improvements

- Tambahkan dataset nyata untuk domain target
- Tambahkan evaluasi metrik seperti mAP dan confusion matrix
- Tambahkan API service dengan FastAPI
- Tambahkan paket pip agar SDK mudah dipasang di proyek lain

---

## 📜 License

Silakan tambahkan lisensi sesuai kebutuhan distribusi proyek Anda.
