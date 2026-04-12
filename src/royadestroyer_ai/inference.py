from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from royadestroyer_ai.labels import LABELS, build_index_maps
from royadestroyer_ai.model_factory import build_model, resolve_device
from royadestroyer_ai.postprocess import enrich_prediction
from royadestroyer_ai.preprocessing import load_tensor_from_bytes


class Predictor:
    def __init__(self, model_dir: Path, image_size: int, top_k: int = 3) -> None:
        self.model_dir = model_dir
        self.image_size = image_size
        self.top_k = top_k
        self.device = resolve_device()
        self.model = None
        self.model_version = "unloaded"
        self.model_name = "mobilenetv3_large_100"
        self.labels = LABELS
        self.index_to_label = {index: label for index, label in enumerate(self.labels)}
        self._load()

    def _load(self) -> None:
        model_path = self.model_dir / "model.pt"
        metadata_path = self.model_dir / "metadata.json"
        labels_path = self.model_dir / "labels.json"
        if not model_path.exists():
            return
        if labels_path.exists():
            self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
            _, self.index_to_label = build_index_maps(self.labels)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.model_version = metadata.get("model_version", "unknown")
            self.model_name = metadata.get("model_name", self.model_name)
        else:
            self.model_version = self.model_dir.name
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = build_model(num_classes=len(self.labels), model_name=self.model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # Pixel-level thresholds for the pre-model image sanity check.
    # These catch black/white/uniform images before the model even runs.
    _MIN_MEAN = 15.0    # below → nearly black
    _MAX_MEAN = 240.0   # above → blown-out / nearly white
    _MIN_STD  = 12.0    # below → nearly uniform (solid colour, blank screen)

    @staticmethod
    def _is_unusable_image(payload: bytes) -> bool:
        """Return True for images that are too dark, too bright, or too uniform
        to contain useful leaf information.  Uses raw pixel stats so it is
        model-independent."""
        try:
            arr = np.array(
                Image.open(BytesIO(payload)).convert("RGB"), dtype=np.float32
            )
            mean = float(arr.mean())
            std  = float(arr.std())
            return mean < 15.0 or mean > 240.0 or std < 12.0
        except Exception:
            return False  # unreadable format (e.g. HEIC) → let PIL in predict() handle it

    def predict(self, payload: bytes) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        # Fast pre-check: reject obviously bad images without hitting the model.
        if self._is_unusable_image(payload):
            enrichment = enrich_prediction("imagen_invalida")
            return {
                "predictedClass": "imagen_invalida",
                "confidence": 0.0,
                "topK": [{"label": "imagen_invalida", "score": 0.0}],
                "severity": enrichment["severity"],
                "symptoms": enrichment["symptoms"],
                "recommendations": enrichment["recommendations"],
                "modelVersion": self.model_version,
                "isInvalidImage": True,
            }

        batch = load_tensor_from_bytes(payload, self.image_size).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probabilities)[::-1][: self.top_k]
        predicted_index = int(top_indices[0])
        predicted_label = self.index_to_label[predicted_index]
        top_confidence = float(probabilities[predicted_index])

        # Also honour the model's own imagen_invalida prediction.
        is_invalid = predicted_label == "imagen_invalida"

        enrichment = enrich_prediction(predicted_label)
        return {
            "predictedClass": predicted_label,
            "confidence": top_confidence,
            "topK": [
                {
                    "label": self.index_to_label[int(index)],
                    "score": float(probabilities[int(index)]),
                }
                for index in top_indices
            ],
            "severity": enrichment["severity"],
            "symptoms": enrichment["symptoms"],
            "recommendations": enrichment["recommendations"],
            "modelVersion": self.model_version,
            "isInvalidImage": is_invalid,
        }

    def health(self) -> dict:
        return {
            "status": "ok" if self.is_loaded else "degraded",
            "modelLoaded": self.is_loaded,
            "modelVersion": self.model_version,
            "classes": len(self.labels),
        }
