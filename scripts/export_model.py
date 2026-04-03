from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings
from royadestroyer_ai.labels import LABELS


def main() -> int:
    settings = load_settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    labels_path = settings.model_dir / "labels.json"
    metadata_path = settings.model_dir / "metadata.json"
    metrics_src = settings.artifacts_root / "metrics" / "evaluation.json"
    metrics_dst = settings.model_dir / "metrics.json"

    labels_path.write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
    metadata = {
        "model_version": settings.model_version,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "image_size": settings.image_size,
        "classes": LABELS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metrics_src.exists():
        metrics_dst.write_text(metrics_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Exported metadata to {settings.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
