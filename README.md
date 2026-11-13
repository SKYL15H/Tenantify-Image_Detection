---
title: Tenantify Food Authenticity API
emoji: 🍽️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
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
├── Dockerfile            # Konfigurasi container untuk deploy ke Hugging Face Spaces / Docker
├── api.py                # REST API Service (FastAPI + Swagger Docs di /docs)
├── app.py                # Web Interface interaktif (Streamlit)
├── detector.py           # Core Logic & Model Detector (OpenAI CLIP Zero-Shot)
├── detect_food.ipynb     # Jupyter Notebook (lingkungan riset/eksperimen awal)
├── folder_foto/          # Kumpulan dataset sampel foto untuk pengujian
├── requirements.txt      # Daftar dependensi Python (FastAPI, Uvicorn, Torch, Streamlit, dll.)
├── .env.example          # Template konfigurasi environment
├── .env                  # Variabel environment lokal (diabaikan oleh git)
├── .gitignore            # Aturan file yang tidak di-push ke git
└── README.md             # Dokumentasi teknis proyek (lengkap dengan metadata HF Spaces)
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

## ☁️ Panduan Deploy ke Hugging Face Spaces

Deploy ke **Hugging Face Spaces** memberikan resource gratis **2 vCPU & 16 GB RAM** (jauh lebih lega dibanding platform lain untuk model AI).

Ada 2 opsi deploy di Hugging Face:

### Opsi 1: Deploy REST API (FastAPI) via Docker Space (Rekomendasi)
Ini membuat API live dengan endpoint HTTPS publik yang langsung bisa ditembak oleh fullstack developer dari frontend manapun.

1. Buka [huggingface.co/new-space](https://huggingface.co/new-space).
2. Isi nama Space, misal: `tenantify-food-detection`.
3. Pilih **Space SDK: Docker** (Blank template).
4. Pilih **Space hardware: CPU basic (2 vCPU · 16 GB RAM · Free)**.
5. Set visibilitas ke **Public** lalu klik **Create Space**.
6. Hubungkan repository GitHub Anda ke Space tersebut:
   * **Cara A (Push langsung lewat Git remote HF)**:
     ```bash
     git remote add space https://huggingface.co/spaces/<username>/<space-name>
     git push space main
     ```
   * **Cara B (Import GitHub)**: Pada menu settings di HF Space, Anda bisa memilih sync repository GitHub.
7. Hugging Face akan otomatis membaca [`Dockerfile`](file:///d:/Project/Hackathon/Tenantify-Image%20Detection/Dockerfile) dan mem-build container pada port 7860.
8. Setelah status **Running**, endpoint API publik Anda siap digunakan:
   * **Swagger Docs**: `https://<username>-<space-name>.hf.space/docs`
   * **Verify Endpoint**: `https://<username>-<space-name>.hf.space/api/v1/verify-food`

---

### Opsi 2: Deploy Web UI Interaktif (Streamlit)
Jika hanya ingin menampilkan visual web Streamlit di Hugging Face:
1. Buka [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pilih **Space SDK: Streamlit**.
3. Di file `README.md`, ubah baris metadata header:
   ```yaml
   sdk: streamlit
   app_file: app.py
   ```
4. Push kode Anda ke repository Space.

---

## ☁️ Alternatif Deploy ke Streamlit Cloud

1. Pastikan seluruh perubahan sudah ter-push ke GitHub:
   ```bash
   git add .
   git commit -m "feat: complete app and documentation"
   git push origin main
   ```
2. Kunjungi [share.streamlit.io](https://share.streamlit.io/).
3. Login menggunakan akun GitHub.
4. Klik **New app**, pilih repository `Tenantify-Image_Detection`, branch `main`, dan Main file path `app.py`.
5. Klik **Deploy!**

---

## 👥 Tim & Hak Cipta
Dikembangkan untuk proyek **Tenantify** — Hak Cipta © 2026.

