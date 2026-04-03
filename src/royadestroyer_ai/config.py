from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    artifacts_root: Path
    model_version: str
    seed: int
    image_size: int
    batch_size: int
    epochs_head: int
    epochs_finetune: int
    predict_top_k: int
    model_dir: Path


def load_settings() -> Settings:
    default_root = Path(__file__).resolve().parents[2]
    project_root = Path(os.getenv("RD_PROJECT_ROOT", default_root)).resolve()
    data_root = Path(os.getenv("RD_DATA_ROOT", project_root / "data")).resolve()
    artifacts_root = Path(
        os.getenv("RD_ARTIFACTS_ROOT", project_root / "artifacts")
    ).resolve()
    model_version = os.getenv("RD_MODEL_VERSION", "roya-model-v1.0.0")
    model_dir = Path(
        os.getenv("RD_MODEL_DIR", artifacts_root / "models" / model_version)
    ).resolve()
    return Settings(
        project_root=project_root,
        data_root=data_root,
        artifacts_root=artifacts_root,
        model_version=model_version,
        seed=int(os.getenv("RD_TRAIN_SEED", "42")),
        image_size=int(os.getenv("RD_IMAGE_SIZE", "224")),
        batch_size=int(os.getenv("RD_BATCH_SIZE", "16")),
        epochs_head=int(os.getenv("RD_EPOCHS_HEAD", "10")),
        epochs_finetune=int(os.getenv("RD_EPOCHS_FINETUNE", "15")),
        predict_top_k=int(os.getenv("RD_PREDICT_TOP_K", "3")),
        model_dir=model_dir,
    )
