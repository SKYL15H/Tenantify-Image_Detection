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
├── app.py                # Web interface (Streamlit) & Core Logic Detector
├── detect_food.ipynb     # Jupyter Notebook (lingkungan riset/eksperimen awal)
├── folder_foto/          # Kumpulan dataset sampel foto untuk pengujian
├── requirements.txt      # Daftar dependensi Python
├── .env.example          # Template konfigurasi environment
├── .env                  # Variabel environment lokal (diabaikan oleh git)
├── .gitignore            # Aturan file yang tidak di-push ke git
└── README.md             # Dokumentasi teknis proyek
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

## 🔌 Kontrak Data & Panduan Integrasi (API / Backend)

Jika tim backend (Node.js/Express, Laravel, FastAPI, Go, dll.) ingin menghubungkan sistem ini ke alur registrasi tenant, ada dua opsi integrasi:

### Opsi A: Import Langsung di Python
Fullstack/Backend developer dapat langsung mengimpor kelas `FoodAuthenticityDetector` dari `app.py`:

```python
from app import FoodAuthenticityDetector, verdict

# Inisialisasi detector (cukup 1x saat server start)
detector = FoodAuthenticityDetector()

# Load gambar dari path file atau buffer bytes
image = detector.load_image("path/ke/foto_tenant.jpg")

if image:
    # Prediksi
    result = detector.predict(image)
    label, status = verdict(result, threshold=0.70)

    print("Hasil:", label)         # e.g. "📸 MAKANAN ASLI"
    print("Prob Asli:", result["real"])  # e.g. 0.932
    print("Prob AI:", result["ai"])      # e.g. 0.068
```

### Opsi B: Membungkus Menjadi REST API (Contoh FastAPI)
Jika backend utama menggunakan bahasa non-Python (Express/NestJS/Laravel), modul ini bisa dibungkus dengan FastAPI dalam 1 file `api.py`:

```python
# api.py
from fastapi import FastAPI, UploadFile, File, Form
from app import FoodAuthenticityDetector, verdict
import io

app = FastAPI(title="Tenantify Image Authenticity API")
detector = FoodAuthenticityDetector()

@app.post("/api/v1/verify-food")
async def verify_food(
    file: UploadFile = File(...),
    threshold: float = Form(0.70)
):
    contents = await file.read()
    image = detector.load_image(io.BytesIO(contents))
    
    if image is None:
        return {"success": False, "message": "Format gambar tidak valid"}

    result = detector.predict(image)
    label, status = verdict(result, threshold=threshold)

    return {
        "success": True,
        "filename": file.filename,
        "verdict": label,
        "status": status,  # "success", "error" (AI detected), "warning" (uncertain)
        "confidence": {
            "real": round(result["real"], 4),
            "ai": round(result["ai"], 4),
        },
        "is_authentic": result["real"] >= threshold
    }
```

#### Contoh JSON Response yang Dihasilkan:
```json
{
  "success": true,
  "filename": "nasi_goreng_tenant1.jpg",
  "verdict": "📸 MAKANAN ASLI",
  "status": "success",
  "confidence": {
    "real": 0.9412,
    "ai": 0.0588
  },
  "is_authentic": true
}
```

---

## ☁️ Panduan Deploy ke Streamlit Cloud

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

