from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

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

    def predict(self, payload: bytes) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        batch = load_tensor_from_bytes(payload, self.image_size).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probabilities)[::-1][: self.top_k]
        predicted_index = int(top_indices[0])
        predicted_label = self.index_to_label[predicted_index]
        enrichment = enrich_prediction(predicted_label)
        return {
            "predictedClass": predicted_label,
            "confidence": float(probabilities[predicted_index]),
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
            "isInvalidImage": predicted_label == "imagen_invalida",
        }

    def health(self) -> dict:
        return {
            "status": "ok" if self.is_loaded else "degraded",
            "modelLoaded": self.is_loaded,
            "modelVersion": self.model_version,
            "classes": len(self.labels),
        }
