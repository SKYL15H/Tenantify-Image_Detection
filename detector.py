import io
import os
from pathlib import Path
from typing import Optional, Union

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
        """Buka gambar dari path, BytesIO, atau raw bytes."""
        try:
            if isinstance(image_input, bytes):
                image_input = io.BytesIO(image_input)
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
    """
    Ubah probabilitas menjadi label status dan kode klasifikasi:
    - ("📸 MAKANAN ASLI", "success")
    - ("🤖 BUATAN AI", "error")
    - ("🤔 TIDAK YAKIN", "warning")
    """
    top = max(result["real"], result["ai"])
    if top < threshold:
        return "🤔 TIDAK YAKIN", "warning"
    if result["real"] > result["ai"]:
        return "📸 MAKANAN ASLI", "success"
    return "🤖 BUATAN AI", "error"

