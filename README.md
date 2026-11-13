---
title: Tenantify Food Authenticity Detection
emoji: 🍽️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

# 🍽️ Tenantify — AI Food Authenticity Detection Service

Layanan deteksi keaslian foto makanan untuk platform **Tenantify**. Modul ini memverifikasi apakah foto menu yang diunggah oleh calon mitra/tenant adalah **foto makanan asli** (hasil jepretan kamera/ponsel dengan pencahayaan alami) atau **gambar buatan AI** (Midjourney, DALL-E, Stable Diffusion, dll. yang memiliki tekstur plastik/hyper-realistic buatan).

---

## 📌 Ringkasan untuk Fullstack Developer

* **Teknologi Utama**: Python 3.10+, PyTorch, Hugging Face Transformers (`CLIP`), Streamlit.
* **Model AI**: `openai/clip-vit-base-patch32` (Zero-shot classification tanpa perlu training ulang).
* **Ukuran Model di RAM**: ~600 MB (sangat ringan, bisa jalan lancar di CPU maupun GPU).
* **Format Gambar yang Didukung**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

---

## 📂 Struktur Direktori Proyek

```text
Tenantify-Image Detection/
├── app.py                # Entrypoint Utama Hugging Face Spaces (Gradio + Dukungan ZeroGPU)
├── api.py                # REST API Service mandiri (FastAPI + Swagger Docs di /docs)
├── streamlit_app.py      # Alternatif Web Interface lokal (Streamlit)
├── detector.py           # Core Logic & Model Detector (OpenAI CLIP Zero-Shot)
├── detect_food.ipynb     # Jupyter Notebook (lingkungan riset/eksperimen awal)
├── folder_foto/          # Kumpulan dataset sampel foto untuk pengujian
├── requirements.txt      # Daftar dependensi Python (Gradio, Torch, FastAPI, Transformers, dll.)
├── .env.example          # Template konfigurasi environment
├── .env                  # Variabel environment lokal (diabaikan oleh git)
├── .gitignore            # Aturan file yang tidak di-push ke git
└── README.md             # Dokumentasi teknis proyek & metadata konfigurasi Hugging Face Spaces
```

---

## 🚀 Panduan Setup & Menjalankan di Lokal

### 1. Prasyarat
- **Python 3.10** atau lebih baru terpasang di sistem (`python --version`).
- Git.

### 2. Langkah Instalasi

Jalankan perintah berikut di terminal:

```powershell
# 1. Masuk ke folder proyek
cd "Tenantify-Image Detection"

# 2. Buat Virtual Environment (Sangat direkomendasikan)
python -m venv venv

# 3. Aktifkan Virtual Environment
# Di Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Di macOS / Linux:
# source venv/bin/activate

# 4. Install dependensi
pip install -r requirements.txt

# 5. Salin konfigurasi environment
cp .env.example .env
```

### 3. Menjalankan Web UI (Streamlit)

```powershell
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser Anda pada alamat: `http://localhost:8501`.

---

## ⚙️ Variabel Konfigurasi (`.env`)

| Variabel | Tipe | Default | Deskripsi |
| :--- | :---: | :---: | :--- |
| `MODEL_NAME` | `str` | `openai/clip-vit-base-patch32` | Nama model Hugging Face CLIP yang digunakan. |
| `DEVICE` | `str` | `cuda` (atau `cpu`) | Perangkat komputasi. Otomatis fallback ke CPU jika GPU tidak ada. |
| `THRESHOLD` | `float` | `0.70` | Ambang keyakinan minimum (0.50 - 0.95). Di bawah ini status menjadi `TIDAK YAKIN`. |
| `SHOW_DETAIL` | `bool` | `True` | Menampilkan rincian probabilitas tiap kalimat prompt. |
| `HF_TOKEN` | `str` | *(kosong)* | Opsional: Hugging Face User Access Token (tidak wajib untuk model publik). |

---

## 🧠 Cara Kerja Model (Logic & Threshold)

Sistem menggunakan teknik **Zero-shot Classification** dengan model multimodal **CLIP** (Contrastive Language-Image Pretraining):
1. Gambar dikonversi ke vektor representasi visual (Image Embeddings).
2. Teks prompt (5 deskripsi foto asli vs 5 deskripsi foto AI) dikonversi ke vektor teks (Text Embeddings).
3. Dihitung skor kesamaan cosinus (*cosine similarity*), dirata-ratakan per kelas, kemudian dinormalisasi dengan *Softmax* menjadi probabilitas:
   - `real` (Probabilitas Foto Asli): rentang `0.0` - `1.0`
   - `ai` (Probabilitas Buatan AI): rentang `0.0` - `1.0`

### Aturan Keputusan (*Verdict*):
* Jika `real >= THRESHOLD` ➔ **📸 MAKANAN ASLI**
* Jika `ai >= THRESHOLD` ➔ **🤖 BUATAN AI**
* Jika `max(real, ai) < THRESHOLD` ➔ **🤔 TIDAK YAKIN** *(Flagged untuk review manual oleh admin)*

---

## 🔌 Panduan Menjalankan REST API (FastAPI)

Service ini sudah menyediakan REST API berbasis **FastAPI** (`api.py`) dengan dukungan CORS aktif, cocok untuk dihubungkan langsung ke frontend (Next.js/React/Vue/Mobile) maupun backend utama (Express, NestJS, Laravel, Golang, dsb.).

### 1. Menjalankan Server API

```powershell
# Jalankan langsung dengan Python:
python api.py

# Atau gunakan Uvicorn:
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
Server akan berjalan di `http://localhost:8000`.

### 2. Dokumentasi Interaktif (Swagger UI)
Kunjungi browser di:
* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 3. Daftar Endpoint API

#### A. Verifikasi 1 Foto Makanan
* **URL**: `POST /api/v1/verify-food`
* **Content-Type**: `multipart/form-data`
* **Body Form**:
  * `file`: File gambar (`.jpg`, `.jpeg`, `.png`, `.webp`) — *Required*
  * `threshold`: `0.70` (float) — *Optional*

**Contoh Response:**
```json
{
  "success": true,
  "filename": "nasi_goreng_tenant.jpg",
  "verdict": "📸 MAKANAN ASLI",
  "status": "success",
  "is_authentic": true,
  "confidence": {
    "real": 0.9412,
    "ai": 0.0588
  },
  "threshold_used": 0.70
}
```

#### B. Verifikasi Banyak Foto Sekaligus (Batch)
* **URL**: `POST /api/v1/verify-food/batch`
* **Content-Type**: `multipart/form-data`
* **Body Form**:
  * `files`: Daftar file gambar — *Required*
  * `threshold`: `0.70` (float) — *Optional*

**Contoh Response:**
```json
{
  "success": true,
  "summary": {
    "total_images": 2,
    "authentic_count": 1,
    "ai_count": 1,
    "uncertain_count": 0
  },
  "results": [
    {
      "success": true,
      "filename": "menu_real.jpg",
      "verdict": "📸 MAKANAN ASLI",
      "status": "success",
      "is_authentic": true,
      "confidence": { "real": 0.92, "ai": 0.08 },
      "threshold_used": 0.70
    },
    {
      "success": true,
      "filename": "menu_ai.jpg",
      "verdict": "🤖 BUATAN AI",
      "status": "error",
      "is_authentic": false,
      "confidence": { "real": 0.12, "ai": 0.88 },
      "threshold_used": 0.70
    }
  ]
}
```

---

### 4. Contoh Pemanggilan dari Frontend (JavaScript / TypeScript / Next.js)

```typescript
// Contoh implementasi di Frontend / Client
async function verifyFoodImage(imageFile: File, threshold: number = 0.70) {
  const formData = new FormData();
  formData.append("file", imageFile);
  formData.append("threshold", threshold.toString());

  const response = await fetch("http://localhost:8000/api/v1/verify-food", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (data.is_authentic) {
    console.log("Foto valid & asli:", data.confidence.real);
  } else {
    alert(`Peringatan: Gambar terdeteksi ${data.verdict}`);
  }
  return data;
}
```

---

### 5. Contoh Pemanggilan via cURL

```bash
curl -X POST "http://localhost:8000/api/v1/verify-food" \
     -F "file=@folder_foto/AI-1.jpg" \
     -F "threshold=0.70"
```

---

## ☁️ Panduan Deploy ke Hugging Face Spaces (Gradio + ZeroGPU)

Konfigurasi repository ini sudah disesuaikan dengan **Hugging Face Spaces** menggunakan **Gradio SDK**, lisensi **MIT**, dan akselerasi **ZeroGPU** (NVIDIA A100 dinamis saat inferensi).

### 1. Buat Space di Hugging Face
1. Buka [**huggingface.co/new-space**](https://huggingface.co/new-space).
2. **Space name**: misal `tenantify-food-detection`.
3. **License**: Pilih `mit`.
4. **Space SDK**: Pilih **Gradio** (bukan Docker).
5. **Space hardware**: Pilih **ZeroGPU** (atau *CPU basic* jika ZeroGPU belum diaktifkan).
6. **Visibility**: Pilih **Public**.
7. Klik **Create Space**.

---

### 2. Push Kode ke Hugging Face Space

Jalankan perintah berikut di terminal:

```powershell
# 1. Tambahkan remote Hugging Face Space (ganti <username> dan <nama-space>)
git remote add hf https://huggingface.co/spaces/<username>/<nama-space>

# 2. Push kode ke branch main Hugging Face
git push hf main
```
*(Gunakan Hugging Face Access Token sebagai password jika diminta autentikasi).*

Setelah push, Hugging Face akan otomatis menginstall dependensi dari `requirements.txt` dan menjalankan `app.py` dengan ZeroGPU!

---

### 3. Cara Fullstack Developer Mengakses API dari Gradio Space

Gradio di Hugging Face Spaces otomatis menyediakan API endpoint gratis tanpa perlu backend tambahan!

#### A. Menggunakan JavaScript / TypeScript (`@gradio/client`) — Rekomendasi Next.js / React:
```bash
npm install @gradio/client
```
```typescript
import { Client } from "@gradio/client";

// Sambungkan ke Space Hugging Face Anda
const client = await Client.connect("<username>/<nama-space>");

// Kirim gambar (File/Blob) dan ambang batas (threshold)
const result = await client.predict("/predict", {
  image: imageBlob,
  threshold: 0.70,
});

// data[2] berisi respon JSON terstruktur
console.log(result.data[2]);
/*
{
  success: true,
  verdict: "📸 MAKANAN ASLI",
  status: "success",
  is_authentic: true,
  confidence: { real: 0.9412, ai: 0.0588 },
  threshold_used: 0.70
}
*/
```

#### B. Menggunakan Python Client (`gradio_client`):
```bash
pip install gradio_client
```
```python
from gradio_client import Client, handle_file

client = Client("<username>/<nama-space>")
result = client.predict(
    image=handle_file("path/to/food.jpg"),
    threshold=0.70,
    api_name="/predict"
)
print(result[2])  # Respon dict / JSON
```

---

## 👥 Tim & Hak Cipta
Lisensi: **MIT** — Dikembangkan untuk proyek **Tenantify** © 2026.

