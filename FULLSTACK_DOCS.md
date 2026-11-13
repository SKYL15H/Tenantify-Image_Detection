# 📖 Dokumentasi Integrasi API — Tenantify Food Authenticity Detection

Dokumentasi ini ditujukan khusus untuk **Fullstack / Frontend / Backend Developer** yang ingin mengintegrasikan modul deteksi keaslian foto menu tenant (mendeteksi foto makanan asli vs buatan generator AI) ke dalam aplikasi utama **Tenantify**.

---

## 📌 Ringkasan Layanan & Host

* **Live Space**: [`https://huggingface.co/spaces/s4usan/Tenantify-Image_Detection`](https://huggingface.co/spaces/s4usan/Tenantify-Image_Detection)
* **Space Identifier**: `s4usan/Tenantify-Image_Detection`
* **Hardware**: **Hugging Face ZeroGPU** (Akselerasi dinamis Nvidia A100 GPU — inferensi cepat < 1 detik)
* **Model AI**: `openai/clip-vit-base-patch32` (Multimodal Zero-shot classification)
* **Lisensi**: MIT

---

## 🔌 Cara Integrasi Cepat (Pilih Sesuai Tech Stack)

### Opsi 1: Frontend Next.js / React / Node.js (Sangat Direkomendasikan)

Gunakan SDK resmi Hugging Face Gradio Client:

```bash
npm install @gradio/client
```

#### Contoh Implementasi (TypeScript / JavaScript):

```typescript
import { Client } from "@gradio/client";

export interface VerifyFoodResponse {
  success: boolean;
  verdict: "📸 MAKANAN ASLI" | "🤖 BUATAN AI" | "🤔 TIDAK YAKIN";
  status: "success" | "error" | "warning";
  is_authentic: boolean;
  confidence: {
    real: number; // Rentang 0.0 - 1.0 (e.g. 0.9412)
    ai: number;   // Rentang 0.0 - 1.0 (e.g. 0.0588)
  };
  threshold_used: number;
}

/**
 * Memverifikasi keaslian foto makanan tenant
 * @param file File atau Blob gambar dari form input (<input type="file" />)
 * @param threshold Ambang batas keyakinan (default: 0.70)
 */
export async function verifyFoodImage(
  file: File | Blob,
  threshold: number = 0.70
): Promise<VerifyFoodResponse> {
  try {
    // 1. Hubungkan ke Hugging Face Space
    const client = await Client.connect("s4usan/Tenantify-Image_Detection");

    // 2. Jalankan fungsi prediksi
    const result = await client.predict("/predict", {
      image: file,
      threshold: threshold,
    });

    // Output index 2 adalah data JSON terstruktur
    const data = result.data[2] as VerifyFoodResponse;
    return data;
  } catch (error) {
    console.error("Gagal memverifikasi keaslian foto:", error);
    throw error;
  }
}
```

---

### Opsi 2: Backend Python (FastAPI / Django / Flask)

Jika backend utama Tenantify menggunakan Python:

```bash
pip install gradio_client
```

```python
from gradio_client import Client, handle_file

client = Client("s4usan/Tenantify-Image_Detection")

# Kirim path file atau URL gambar
result = client.predict(
    image=handle_file("path/ke/foto_makanan.jpg"),
    threshold=0.70,
    api_name="/predict"
)

# result[2] adalah dict JSON
verification_data = result[2]
print(verification_data)
# {
#   "success": True,
#   "verdict": "📸 MAKANAN ASLI",
#   "status": "success",
#   "is_authentic": True,
#   "confidence": {"real": 0.9412, "ai": 0.0588},
#   "threshold_used": 0.70
# }
```

---

### Opsi 3: HTTP REST API Direct (cURL / Postman / Axios)

Jika ingin memanggil langsung via HTTP standard:

* **Endpoint**: `https://s4usan-tenantify-image-detection.hf.space/call/predict`
* **Metode**: `POST`
* **Headers**: `Content-Type: application/json`

*(Sangat disarankan menggunakan `@gradio/client` di Opsi 1 karena otomatis menangani upload blob dan polling event secara native tanpa perlu handle protokol streaming manual).*

---

## 📊 Skema Input & Output (Data Contract)

### 1. Parameter Input

| Parameter | Tipe Data | Wajib? | Default | Keterangan |
| :--- | :---: | :---: | :---: | :--- |
| `image` | `File / Blob` | **Ya** | - | File gambar makanan (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`). |
| `threshold` | `number (float)` | Tidak | `0.70` | Batas keyakinan (rentang `0.50` - `0.95`). |

---

### 2. Format Respon JSON (`data[2]`)

```json
{
  "success": true,
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

### 3. Logika & Arti Nilai Status:

| Status (`status`) | Verdict | `is_authentic` | Tindakan di Aplikasi Tenantify |
| :--- | :--- | :---: | :--- |
| **`"success"`** | `📸 MAKANAN ASLI` | `true` | **Lolos Otomatis**: Foto valid, menu siap ditayangkan di katalog. |
| **`"error"`** | `🤖 BUATAN AI` | `false` | **Ditolak**: Beri notifikasi ke tenant untuk mengunggah foto asli (bukan hasil render/generator AI). |
| **`"warning"`** | `🤔 TIDAK YAKIN` | `false` | **Review Manual**: Skor di bawah threshold. Masukkan foto ke dashboard antrean verifikasi admin. |

---

## 💻 Contoh Komponen React Siap Pakai (`FoodUploadVerifier.tsx`)

Bisa langsung di-copy-paste ke proyek Next.js / React kamu:

```tsx
"use client";

import React, { useState } from "react";
import { Client } from "@gradio/client";

export default function FoodUploadVerifier() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleVerify = async () => {
    if (!file) return alert("Pilih foto terlebih dahulu!");
    setLoading(true);
    setResult(null);

    try {
      const client = await Client.connect("s4usan/Tenantify-Image_Detection");
      const res = await client.predict("/predict", {
        image: file,
        threshold: 0.70,
      });

      // Index 2 berisi payload JSON
      setResult(res.data[2]);
    } catch (err) {
      alert("Gagal menghubungi layanan verifikasi foto.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4 border">
      <h2 className="text-xl font-bold text-gray-800">Verifikasi Foto Menu</h2>
      <p className="text-sm text-gray-500">
        Pastikan foto menu adalah foto asli, bukan hasil buatan AI.
      </p>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
      />

      <button
        onClick={handleVerify}
        disabled={loading || !file}
        className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg disabled:bg-gray-300 transition"
      >
        {loading ? "Menganalisis dengan AI..." : "Verifikasi Foto"}
      </button>

      {result && (
        <div
          className={`p-4 rounded-lg border ${
            result.status === "success"
              ? "bg-green-50 border-green-200 text-green-800"
              : result.status === "error"
              ? "bg-red-50 border-red-200 text-red-800"
              : "bg-yellow-50 border-yellow-200 text-yellow-800"
          }`}
        >
          <div className="text-lg font-bold">{result.verdict}</div>
          <div className="text-sm mt-1">
            Skor Keaslian: {(result.confidence.real * 100).toFixed(1)}% | Skor AI: {(result.confidence.ai * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## ⚡ Tips & Catatan Performa (ZeroGPU)

1. **Cold-Start ZeroGPU**: Saat request pertama setelah Space dalam keadaan idle (tidur), sistem membutuhkan sekitar ~2 hingga 4 detik untuk mengalokasikan GPU Nvidia A100. Setelah aktif, request berikutnya berjalan sangat cepat (< 500ms).
2. **Ukuran File**: Disarankan ukuran foto maksimal 5 MB untuk mempercepat proses upload jaringan.
3. **Format**: Format `.jpg`, `.jpeg`, `.png`, dan `.webp` didukung penuh.
