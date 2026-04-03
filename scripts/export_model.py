from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings
from royadestroyer_ai.datasets import detect_split_labels


def main() -> int:
    settings = load_settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    active_labels = detect_split_labels(settings.data_root / "splits" / "train")
    if not active_labels:
        raise RuntimeError("No labels detected from train split")

    labels_path = settings.model_dir / "labels.json"
    metadata_path = settings.model_dir / "metadata.json"
    metrics_src = settings.artifacts_root / "metrics" / "evaluation.json"
    metrics_dst = settings.model_dir / "metrics.json"

    labels_path.write_text(json.dumps(active_labels, indent=2), encoding="utf-8")
    metadata = {
        "model_version": settings.model_version,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": settings.image_size,
        "classes": active_labels,
        "framework": "pytorch",
        "model_name": "mobilenetv3_large_100",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metrics_src.exists():
        metrics_dst.write_text(metrics_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Exported metadata to {settings.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
