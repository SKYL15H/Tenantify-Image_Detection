import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from detector import DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD, FoodAuthenticityDetector, verdict

# Global detector instance
detector: Optional[FoodAuthenticityDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat model saat server startup dan membersihkan resource saat shutdown."""
    global detector
    print("[INFO] Menginisialisasi FoodAuthenticityDetector (CLIP)...")
    detector = FoodAuthenticityDetector(model_name=DEFAULT_MODEL_NAME)
    print(f"[INFO] Model siap digunakan pada device: {detector.device} ✅")
    yield
    print("[INFO] Mematikan service...")


app = FastAPI(
    title="Tenantify — Food Authenticity API",
    description="REST API untuk memverifikasi keaslian foto menu/makanan tenant (Foto Asli vs Buatan AI) menggunakan OpenAI CLIP.",
    version="1.0.0",
    lifespan=lifespan,
)

# Aktifkan CORS agar frontend (React, Next.js, Vue, mobile) bisa akses langsung tanpa blokir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- RESPONSE SCHEMAS ---
class ConfidenceDetail(BaseModel):
    real: float = Field(..., description="Probabilitas makanan asli (0.0 - 1.0)")
    ai: float = Field(..., description="Probabilitas buatan AI (0.0 - 1.0)")


class VerificationResult(BaseModel):
    success: bool
    filename: str
    verdict: str
    status: str
    is_authentic: bool
    confidence: ConfidenceDetail
    threshold_used: float


class BatchSummary(BaseModel):
    total_images: int
    authentic_count: int
    ai_count: int
    uncertain_count: int


class BatchVerificationResponse(BaseModel):
    success: bool
    summary: BatchSummary
    results: list[VerificationResult]


# --- ENDPOINTS ---
@app.get("/", tags=["Health Check"])
async def root():
    """Cek status kesehatan API dan info model."""
    return {
        "service": "Tenantify Food Authenticity Detection API",
        "status": "healthy",
        "model": DEFAULT_MODEL_NAME,
        "device": detector.device if detector else "not initialized",
        "docs_url": "/docs",
    }


@app.post(
    "/api/v1/verify-food",
    response_model=VerificationResult,
    tags=["Food Authenticity"],
    summary="Verifikasi Keaslian Satu Foto Makanan",
)
async def verify_food(
    file: UploadFile = File(..., description="File foto makanan (.jpg, .png, .webp, .jpeg)"),
    threshold: Optional[float] = Form(
        None, description=f"Batas keyakinan (default: {DEFAULT_THRESHOLD})"
    ),
):
    """
    Menganalisis satu foto makanan:
    - Mengembalikan skor probabilitas 'real' dan 'ai'
    - Menentukan verdict: '📸 MAKANAN ASLI', '🤖 BUATAN AI', atau '🤔 TIDAK YAKIN'
    """
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model detector belum selesai dimuat.",
        )

    thresh = threshold if threshold is not None else DEFAULT_THRESHOLD

    try:
        content = await file.read()
        image = detector.load_image(content)
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' bukan gambar yang valid atau format tidak didukung.",
            )

        res = detector.predict(image)
        label, status_code = verdict(res, threshold=thresh)

        return VerificationResult(
            success=True,
            filename=file.filename,
            verdict=label,
            status=status_code,
            is_authentic=(res["real"] >= thresh),
            confidence=ConfidenceDetail(
                real=round(res["real"], 4),
                ai=round(res["ai"], 4),
            ),
            threshold_used=thresh,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}",
        )


@app.post(
    "/api/v1/verify-food/batch",
    response_model=BatchVerificationResponse,
    tags=["Food Authenticity"],
    summary="Verifikasi Banyak Foto Sekaligus (Batch)",
)
async def verify_food_batch(
    files: list[UploadFile] = File(..., description="Daftar file foto makanan"),
    threshold: Optional[float] = Form(
        None, description=f"Batas keyakinan (default: {DEFAULT_THRESHOLD})"
    ),
):
    """
    Menganalisis banyak foto makanan sekaligus dalam satu request:
    - Cocok untuk menu restoran yang berisi banyak gambar sekaligus.
    """
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model detector belum selesai dimuat.",
        )

    thresh = threshold if threshold is not None else DEFAULT_THRESHOLD
    results = []
    authentic_count = 0
    ai_count = 0
    uncertain_count = 0

    for file in files:
        try:
            content = await file.read()
            image = detector.load_image(content)
            if image is None:
                continue

            res = detector.predict(image)
            label, status_code = verdict(res, threshold=thresh)

            if status_code == "success":
                authentic_count += 1
            elif status_code == "error":
                ai_count += 1
            else:
                uncertain_count += 1

            results.append(
                VerificationResult(
                    success=True,
                    filename=file.filename,
                    verdict=label,
                    status=status_code,
                    is_authentic=(res["real"] >= thresh),
                    confidence=ConfidenceDetail(
                        real=round(res["real"], 4),
                        ai=round(res["ai"], 4),
                    ),
                    threshold_used=thresh,
                )
            )
        except Exception:
            continue

    return BatchVerificationResponse(
        success=True,
        summary=BatchSummary(
            total_images=len(results),
            authentic_count=authentic_count,
            ai_count=ai_count,
            uncertain_count=uncertain_count,
        ),
        results=results,
    )


if __name__ == "__main__":
    import uvicorn

    # Default port 7860 untuk kompatibilitas native Hugging Face Spaces
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)

