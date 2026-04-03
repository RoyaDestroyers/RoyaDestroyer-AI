from __future__ import annotations

import json
from pathlib import Path
import sys

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings
from royadestroyer_ai.labels import LABELS
from royadestroyer_ai.model_factory import build_model


def main() -> int:
    settings = load_settings()
    train_dir = settings.data_root / "splits" / "train"
    val_dir = settings.data_root / "splits" / "val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        seed=settings.seed,
        image_size=(settings.image_size, settings.image_size),
        batch_size=settings.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        seed=settings.seed,
        image_size=(settings.image_size, settings.image_size),
        batch_size=settings.batch_size,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(settings.image_size, len(LABELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(settings.artifacts_root / "checkpoints" / "best.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.epochs_head,
        callbacks=callbacks,
    )

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(settings.model_dir / "model.keras")
    history_path = settings.artifacts_root / "metrics" / "train_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    print(f"Saved model to {settings.model_dir / 'model.keras'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
