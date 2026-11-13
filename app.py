import os
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from detector import (
    DEFAULT_MODEL_NAME,
    DEFAULT_THRESHOLD,
    FoodAuthenticityDetector,
    verdict,
)

# --- DUKUNGAN HUGGING FACE ZEROGPU ---
try:
    import spaces

    HAS_SPACES = True
except (ImportError, ModuleNotFoundError):
    HAS_SPACES = False


def gpu_decorator(fn):
    """Menghubungkan fungsi ke dynamic Nvidia GPU jika berjalan di Hugging Face ZeroGPU."""
    if HAS_SPACES:
        return spaces.GPU(fn)
    return fn


# Inisialisasi detector (awal di CPU; dinamis berpindah ke GPU saat inferensi di ZeroGPU)
detector = FoodAuthenticityDetector(model_name=DEFAULT_MODEL_NAME, device="cpu")


@gpu_decorator
def analyze_food_authenticity(
    image: Optional[Image.Image],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str, dict[str, float], dict]:
    """
    Fungsi inferensi utama:
    - Mendeteksi apakah gambar adalah foto makanan asli atau buatan AI.
    - Menghasilkan label, distribusi probabilitas, dan respon JSON terstruktur.
    """
    if image is None:
        return (
            "⚠️ Silakan unggah foto makanan terlebih dahulu.",
            {},
            {"error": "No image provided"},
        )

    # Jalankan prediksi CLIP
    result = detector.predict(image)
    label, status = verdict(result, threshold=threshold)

    # Format Markdown Banner
    if status == "success":
        color = "#16a34a"  # Green
        status_text = "MAKANAN ASLI TERVERIFIKASI"
    elif status == "error":
        color = "#dc2626"  # Red
        status_text = "TERDETEKSI GAMBAR BUATAN AI"
    else:
        color = "#ca8a04"  # Yellow
        status_text = "HASIL TIDAK YAKIN (PERLU REVIEW MANUAL)"

    verdict_md = f"""
    <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 12px 18px; border-radius: 8px; margin-bottom: 12px;">
        <h2 style="color: {color}; margin: 0; font-size: 20px;">{label}</h2>
        <p style="margin: 4px 0 0 0; color: #4b5563; font-weight: 500;">Status: {status_text} | Keyakinan Maks: {max(result['real'], result['ai']):.1%}</p>
    </div>
    """

    # Format untuk gr.Label chart
    label_distribution = {
        "Makanan Asli (Real Photo)": result["real"],
        "Buatan AI (AI Generated)": result["ai"],
    }

    # Format JSON untuk konsumsi Fullstack Developer / API Client
    api_response = {
        "success": True,
        "verdict": label,
        "status": status,
        "is_authentic": (result["real"] >= threshold),
        "confidence": {
            "real": round(result["real"], 4),
            "ai": round(result["ai"], 4),
        },
        "threshold_used": threshold,
    }

    return verdict_md, label_distribution, api_response


# Kumpulkan contoh foto sampel jika ada di folder_foto
demo_examples = []
demo_folder = Path("folder_foto")
if demo_folder.is_dir():
    sample_paths = sorted(
        [
            str(p)
            for p in demo_folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    # Ambil beberapa contoh sampel real dan AI
    demo_examples = [[p, DEFAULT_THRESHOLD] for p in sample_paths[:6]]


# --- TAMPILAN GRADIO BLOCKS ---
custom_css = """
footer {visibility: hidden}
#logo-text {margin-bottom: 0px}
"""

with gr.Blocks(
    title="Tenantify — AI Food Authenticity Detector",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=custom_css,
) as demo:
    gr.Markdown(
        """
        # 🍽️ Tenantify — AI Food Authenticity Detector
        Verifikasi keaslian foto menu & makanan tenant untuk platform **Tenantify**.
        Mendeteksi foto jepretan asli (*real food photo*) vs rekayasa generator AI (*Midjourney, DALL-E, Stable Diffusion*) menggunakan **OpenAI CLIP Zero-Shot Classification**.
        """,
        elem_id="logo-text",
    )

    with gr.Row():
        with gr.Column(scale=5):
            input_image = gr.Image(
                type="pil",
                label="Unggah Foto Makanan",
                sources=["upload", "clipboard"],
            )

            with gr.Accordion("⚙️ Pengaturan Tambahan", open=False):
                threshold_slider = gr.Slider(
                    minimum=0.50,
                    maximum=0.95,
                    value=DEFAULT_THRESHOLD,
                    step=0.05,
                    label="Ambang Batas Keyakinan (Threshold)",
                    info="Jika probabilitas di bawah batas ini, sistem akan menandai sebagai 'TIDAK YAKIN' untuk direview manual.",
                )

            btn_submit = gr.Button("🔍 Analisis Foto", variant="primary", size="lg")
            btn_clear = gr.ClearButton(components=[input_image], value="🗑️ Reset")

        with gr.Column(scale=5):
            verdict_output = gr.Markdown(
                label="Hasil Verifikasi",
                value="*Unggah foto dan klik 'Analisis Foto' untuk melihat hasil.*",
            )
            chart_output = gr.Label(
                label="Skor Probabilitas",
                num_top_classes=2,
            )
            json_output = gr.JSON(
                label="Data Response (JSON format untuk Fullstack Developer)",
            )

    # Event handlers
    btn_submit.click(
        fn=analyze_food_authenticity,
        inputs=[input_image, threshold_slider],
        outputs=[verdict_output, chart_output, json_output],
        api_name="predict",
    )

    # Bagian contoh sampel
    if demo_examples:
        gr.Markdown("### 🧪 Contoh Sampel Foto:")
        gr.Examples(
            examples=demo_examples,
            inputs=[input_image, threshold_slider],
            outputs=[verdict_output, chart_output, json_output],
            fn=analyze_food_authenticity,
            cache_examples=False,
        )

    # Panduan API untuk Developer
    with gr.Accordion("🔌 Panduan API untuk Fullstack Developer (Gradio API Client)", open=False):
        gr.Markdown(
            """
            Hugging Face Spaces dengan Gradio otomatis menyediakan API endpoint gratis!
            
            #### 1. Cara panggil dari JavaScript / TypeScript (Next.js / Node.js):
            ```bash
            npm install @gradio/client
            ```
            ```javascript
            import { Client } from "@gradio/client";

            const client = await Client.connect("username-kamu/nama-space");
            const result = await client.predict("/predict", {
                image: imageBlob,
                threshold: 0.70,
            });

            console.log(result.data[2]); // JSON output { success: true, verdict: "...", is_authentic: true }
            ```

            #### 2. Cara panggil dari Python:
            ```bash
            pip install gradio_client
            ```
            ```python
            from gradio_client import Client, handle_file

            client = Client("username-kamu/nama-space")
            result = client.predict(
                image=handle_file("path/to/food.jpg"),
                threshold=0.70,
                api_name="/predict"
            )
            print(result)
            ```
            """
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
