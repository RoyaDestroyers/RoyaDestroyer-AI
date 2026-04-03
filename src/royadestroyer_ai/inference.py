from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from royadestroyer_ai.labels import INDEX_TO_LABEL, LABELS
from royadestroyer_ai.postprocess import enrich_prediction
from royadestroyer_ai.preprocessing import add_batch_dimension, load_image_from_bytes


class Predictor:
    def __init__(self, model_dir: Path, image_size: int, top_k: int = 3) -> None:
        self.model_dir = model_dir
        self.image_size = image_size
        self.top_k = top_k
        self.model = None
        self.model_version = "unloaded"
        self._load()

    def _load(self) -> None:
        model_path = self.model_dir / "model.keras"
        metadata_path = self.model_dir / "metadata.json"
        if not model_path.exists():
            return
        self.model = tf.keras.models.load_model(model_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.model_version = metadata.get("model_version", "unknown")
        else:
            self.model_version = self.model_dir.name

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, payload: bytes) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        image = load_image_from_bytes(payload, self.image_size)
        batch = add_batch_dimension(image)
        scores = self.model.predict(batch, verbose=0)[0]
        top_indices = np.argsort(scores)[::-1][: self.top_k]
        predicted_index = int(top_indices[0])
        predicted_label = INDEX_TO_LABEL[predicted_index]
        enrichment = enrich_prediction(predicted_label)
        return {
            "predictedClass": predicted_label,
            "confidence": float(scores[predicted_index]),
            "topK": [
                {"label": INDEX_TO_LABEL[int(index)], "score": float(scores[int(index)])}
                for index in top_indices
            ],
            "severity": enrichment["severity"],
            "symptoms": enrichment["symptoms"],
            "recommendations": enrichment["recommendations"],
            "modelVersion": self.model_version,
            "isInvalidImage": predicted_label == "imagen_invalida",
        }

    def health(self) -> dict:
        return {
            "status": "ok" if self.is_loaded else "degraded",
            "modelLoaded": self.is_loaded,
            "modelVersion": self.model_version,
            "classes": len(LABELS),
        }
