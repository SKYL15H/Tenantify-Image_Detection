import io
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

# Muat variabel environment dari .env jika ada
load_dotenv()

# --- KONFIGURASI DEFAULT ---
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.70))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

REAL_PROMPTS = [
    "a real photo of food taken with a smartphone camera",
    "a candid photograph of a home-cooked meal on a table",
    "an amateur food photo with natural lighting and slight imperfections",
    "a photo of real food with natural crumbs, grease and uneven texture",
    "a restaurant dish photographed by a customer",
]

AI_PROMPTS = [
    "an AI-generated image of food",
    "a digitally rendered, unrealistically perfect dish",
    "a food image with overly smooth plastic-like texture",
    "a hyper-detailed surreal food picture made by midjourney",
    "a computer-generated meal with glossy waxy surfaces and distorted shapes",
]


class FoodAuthenticityDetector:
    """Membungkus model CLIP untuk klasifikasi zero-shot: makanan asli vs buatan AI."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.prompts = REAL_PROMPTS + AI_PROMPTS
        self.n_real = len(REAL_PROMPTS)

    @staticmethod
    def load_image(image_input: Union[str, Path, io.BytesIO, bytes]) -> Optional[Image.Image]:
        """Buka gambar dari path atau stream buffer bytes."""
        try:
            return Image.open(image_input).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        inputs = self.processor(
            text=self.prompts, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        logits = self.model(**inputs).logits_per_image[0]
        real_score = logits[: self.n_real].mean()
        ai_score = logits[self.n_real :].mean()
        probs = torch.softmax(torch.stack([real_score, ai_score]), dim=0)
        per_prompt = torch.softmax(logits, dim=0).tolist()

        return {
            "real": probs[0].item(),
            "ai": probs[1].item(),
            "per_prompt": dict(zip(self.prompts, per_prompt)),
        }


def verdict(result: dict, threshold: float = 0.70) -> tuple[str, str]:
    """Ubah probabilitas menjadi label status dan warna/icon."""
    top = max(result["real"], result["ai"])
    if top < threshold:
        return "🤔 TIDAK YAKIN", "warning"
    if result["real"] > result["ai"]:
        return "📸 MAKANAN ASLI", "success"
    return "🤖 BUATAN AI", "error"


# --- STREAMLIT CACHING ---
@st.cache_resource(show_spinner="Sedang mengunduh dan memuat model AI (CLIP)...")
def get_detector(model_name: str, device: str) -> FoodAuthenticityDetector:
    return FoodAuthenticityDetector(model_name=model_name, device=device)


# --- HALAMAN UTAMA STREAMLIT ---
def main():
    st.set_page_config(
        page_title="Tenantify - Food Authenticity Detector",
        page_icon="🍽️",
        layout="wide",
    )

    st.title("🍽️ Tenantify — AI Food Authenticity Detector")
    st.caption("Verifikasi keaslian foto menu & makanan tenant: mendeteksi foto asli vs buatan generator AI (Midjourney, DALL-E, dsb.) dengan OpenAI CLIP zero-shot classification.")

    # --- SIDEBAR PENGATURAN ---
    st.sidebar.header("⚙️ Pengaturan Model")

    detected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_choice = st.sidebar.radio(
        "Pilih Komputasi (Device):",
        options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
        index=0,
        help="CUDA menggunakan GPU untuk inferensi lebih cepat jika tersedia.",
    )
    st.sidebar.caption(f"Hardware terdeteksi: **{detected_device.upper()}**")

    model_name = st.sidebar.selectbox(
        "Model Hugging Face:",
        options=[
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ],
        index=0,
    )

    threshold = st.sidebar.slider(
        "Ambang Keyakinan (Threshold):",
        min_value=0.50,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Jika probabilitas tertinggi di bawah nilai ini, hasil dinyatakan 'TIDAK YAKIN'.",
    )

    show_detail = st.sidebar.checkbox(
        "Tampilkan detail skor per prompt",
        value=True,
    )

    # Inisialisasi detector
    try:
        detector = get_detector(model_name=model_name, device=device_choice)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return

    # --- TABS INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["📤 Upload Satu Foto", "📁 Batch Processing", "🧪 Uji Gambar Demo"])

    # === TAB 1: SINGLE IMAGE ===
    with tab1:
        st.subheader("Analisis Satu Foto Makanan")
        uploaded_file = st.file_uploader(
            "Pilih foto makanan (.jpg, .png, .webp):",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="single_uploader",
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            image = detector.load_image(uploaded_file)
            if image is None:
                st.error("Gagal membaca file gambar. Pastikan format file valid.")
            else:
                with col1:
                    st.image(image, caption="Foto yang diuji", use_container_width=True)

                with col2:
                    with st.spinner("Menganalisis karakteristik visual..."):
                        res = detector.predict(image)
                        label, status = verdict(res, threshold)

                    st.markdown("### Hasil Deteksi")
                    if status == "success":
                        st.success(f"## {label}")
                    elif status == "error":
                        st.error(f"## {label}")
                    else:
                        st.warning(f"## {label}")

                    st.markdown("#### Probabilitas:")
                    col_real, col_ai = st.columns(2)
                    col_real.metric("📸 Probabilitas Asli", f"{res['real']:.1%}")
                    col_ai.metric("🤖 Probabilitas AI", f"{res['ai']:.1%}")

                    st.progress(res["real"], text=f"Skor Makanan Asli: {res['real']:.1%}")

                    if show_detail:
                        with st.expander("🔍 Detail Skor per Prompt CLIP", expanded=False):
                            df_prompts = pd.DataFrame(
                                [
                                    {"Prompt": p, "Skor": f"{score:.2%}", "Tipe": "Makanan Asli" if p in REAL_PROMPTS else "Buatan AI"}
                                    for p, score in res["per_prompt"].items()
                                ]
                            )
                            st.dataframe(df_prompts, use_container_width=True, hide_index=True)

    # === TAB 2: BATCH PROCESSING ===
    with tab2:
        st.subheader("Analisis Banyak Foto Sekaligus")
        uploaded_files = st.file_uploader(
            "Pilih beberapa file foto makanan:",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="batch_uploader",
        )

        if uploaded_files:
            if st.button("🚀 Mulai Analisis Batch", key="btn_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                rows = []
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Memproses {idx + 1}/{len(uploaded_files)}: {file.name}")
                    img = detector.load_image(file)
                    if img is None:
                        continue

                    res = detector.predict(img)
                    label, _ = verdict(res, threshold)
                    rows.append({
                        "Nama File": file.name,
                        "Hasil": label,
                        "Prob Asli (%)": round(res["real"] * 100, 2),
                        "Prob AI (%)": round(res["ai"] * 100, 2),
                    })
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.text("Analisis batch selesai! 🎉")
                if rows:
                    df_results = pd.DataFrame(rows)
                    st.dataframe(df_results, use_container_width=True)

                    # Export CSV
                    csv = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Hasil Analisis (.csv)",
                        data=csv,
                        file_name="hasil_deteksi_tenantify.csv",
                        mime="text/csv",
                    )

    # === TAB 3: DEMO FOTO ===
    with tab3:
        st.subheader("Coba dengan Sampel Foto Proyek")
        folder_demo = Path("folder_foto")
        if folder_demo.is_dir():
            sample_files = sorted(
                [p for p in folder_demo.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            )
            if sample_files:
                selected_sample = st.selectbox(
                    "Pilih sampel foto dari folder_foto:",
                    options=sample_files,
                    format_func=lambda x: x.name,
                )

                if selected_sample and st.button("🔍 Uji Sampel Ini"):
                    c1, c2 = st.columns([1, 1])
                    img = detector.load_image(selected_sample)
                    if img is not None:
                        with c1:
                            st.image(img, caption=selected_sample.name, use_container_width=True)
                        with c2:
                            with st.spinner("Menganalisis..."):
                                res = detector.predict(img)
                                label, status = verdict(res, threshold)
                            st.markdown("### Hasil Deteksi")
                            if status == "success":
                                st.success(f"## {label}")
                            elif status == "error":
                                st.error(f"## {label}")
                            else:
                                st.warning(f"## {label}")

                            st.metric("Probabilitas Asli", f"{res['real']:.1%}")
                            st.metric("Probabilitas AI", f"{res['ai']:.1%}")
            else:
                st.info("Tidak ada gambar di dalam folder 'folder_foto'.")
        else:
            st.info("Folder sampel 'folder_foto' tidak ditemukan di workspace.")


if __name__ == "__main__":
    main()

