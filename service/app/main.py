from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from service.app.predictor import PREDICTOR
from service.app.schemas import HealthResponse, PredictResponse

app = FastAPI(title="RoyaDestroyer AI Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return PREDICTOR.health()


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    lote_id: str | None = Form(default=None),
    observaciones: str | None = Form(default=None),
) -> dict:
    del lote_id, observaciones
    if not image.content_type or image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported")
    if not PREDICTOR.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Image payload is empty")
    try:
        return PREDICTOR.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
