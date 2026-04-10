from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path
from unittest import TestCase, mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from royadestroyer_ai.inference import Predictor


class FakeModel(nn.Module):
    def load_state_dict(self, state_dict):  # pragma: no cover - exercised via test
        self.state_dict_value = state_dict
        return self

    def to(self, device):  # pragma: no cover - exercised via test
        self.device = device
        return self

    def eval(self):  # pragma: no cover - exercised via test
        self.is_eval = True
        return self

    def forward(self, inputs):
        return torch.tensor([[0.1, 0.9, 0.2, 0.0, 0.3, 0.4]], device=inputs.device)


class PredictorTest(TestCase):
    def test_health_is_degraded_without_model_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = Predictor(Path(tmpdir), image_size=224)
            self.assertFalse(predictor.is_loaded)
            self.assertEqual(predictor.health()["status"], "degraded")

    def test_predict_with_mocked_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            model_dir.joinpath("metadata.json").write_text(
                json.dumps({"model_version": "roya-model-v1.0.0", "model_name": "mobilenetv3_large_100"}),
                encoding="utf-8",
            )
            model_dir.joinpath("labels.json").write_text(
                json.dumps(
                    [
                        "cercospora",
                        "hoja_sana",
                        "minador",
                        "phoma",
                        "roya_avanzada",
                        "roya_temprana",
                    ]
                ),
                encoding="utf-8",
            )
            model_dir.joinpath("model.pt").write_bytes(b"fake-checkpoint")

            with mock.patch("royadestroyer_ai.inference.build_model", return_value=FakeModel()), mock.patch(
                "torch.load", return_value={"model_state_dict": {}}
            ), mock.patch(
                "royadestroyer_ai.inference.load_tensor_from_bytes",
                return_value=torch.zeros((1, 3, 224, 224)),
            ):
                predictor = Predictor(model_dir, image_size=224, top_k=2)
                result = predictor.predict(b"fake-image")

            self.assertTrue(predictor.is_loaded)
            self.assertEqual(result["predictedClass"], "hoja_sana")
            self.assertEqual(result["modelVersion"], "roya-model-v1.0.0")
            self.assertEqual(len(result["topK"]), 2)
