from __future__ import annotations

from pydantic import BaseModel


class TopKItem(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictedClass: str
    confidence: float
    topK: list[TopKItem]
    severity: str
    symptoms: list[str]
    recommendations: list[str]
    modelVersion: str
    isInvalidImage: bool


class HealthResponse(BaseModel):
    status: str
    modelLoaded: bool
    modelVersion: str
    classes: int
