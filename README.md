# Object Detection SDK (Python)

Proyek ini menyediakan contoh SDK deteksi objek sederhana menggunakan Ultralytics YOLO (yolov8). Tujuan: bikin SDK Python yang bisa dipakai untuk inference, dilengkapi skrip pelatihan, ekspor model untuk mobile, dan contoh penggunaan.

**Persiapan**

- Install dependency:

```bash
pip install -r requirements.txt
```

Catatan: `torch` diperlukan — pasang versi yang sesuai (CPU/GPU) jika perlu.

**Struktur proyek**

- `sdk/detector.py`: kelas `Detector` untuk memuat model dan melakukan inferensi.
- `train.py`: skrip latihan untuk dataset custom (YOLO/COCO YAML).
- `export_model.py`: ekspor `.pt` ke `onnx`, `tflite`, atau `coreml`.
- `examples/demo.py`: contoh pemakaian SDK, menyimpan gambar teranotasi.

**Contoh penggunaan SDK (inference)**

1. Siapkan model, misal `yolov8n.pt` atau hasil pelatihan `runs/train/exp/weights/best.pt`.
2. Jalankan demo:

```bash
python examples/demo.py --image path/to/input.jpg --out path/to/out.jpg --weights yolov8n.pt
```

**Melatih model custom**

1. Siapkan dataset dan file YAML (contoh `data.yaml`) yang menunjuk ke `train` dan `val`.
2. Jalankan:

```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 50
```

**Ekspor model untuk mobile/integrasi**
Contoh ekspor ke ONNX:

```bash
python export_model.py --weights runs/train/exp/weights/best.pt --format onnx
```

Setelah mendapat `onnx`/`tflite`, Anda bisa memasukkan model tersebut ke pipeline mobile (Android/iOS). Untuk membuat SDK Android, ekspor ke `tflite` atau gunakan `onnx` + ONNX Runtime Mobile.

**Langkah selanjutnya yang bisa saya bantu**

- Bantu siapkan dataset YAML dan skrip konversi ke TFLite.
- Tambahkan wrapper pip package (`setup.py`) untuk mendistribusikan SDK.
- Buat contoh integrasi Android (Ringkas) yang memanggil model TFLite.

---

Jika mau, saya bisa langsung: memasang dependensi, menjalankan demo, atau membuat paket pip minimal.
