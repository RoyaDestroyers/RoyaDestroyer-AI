from __future__ import annotations

from royadestroyer_ai.config import load_settings
from royadestroyer_ai.inference import Predictor

SETTINGS = load_settings()
PREDICTOR = Predictor(
    SETTINGS.model_dir, SETTINGS.image_size, SETTINGS.predict_top_k
)
