from __future__ import annotations

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings
from royadestroyer_ai.labels import LABELS
from royadestroyer_ai.metrics import build_report, save_report


def main() -> int:
    settings = load_settings()
    test_dir = settings.data_root / "splits" / "test"
    model_path = settings.model_dir / "model.keras"
    model = tf.keras.models.load_model(model_path)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        shuffle=False,
        image_size=(settings.image_size, settings.image_size),
        batch_size=settings.batch_size,
    )
    predictions = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds], axis=0)
    report = build_report(y_true, y_pred, LABELS)
    save_report(report, settings.artifacts_root / "metrics" / "evaluation.json")
    print("Saved evaluation report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
