from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def main() -> int:
    settings = load_settings()
    random.seed(settings.seed)
    unified_root = settings.data_root / "unified"
    splits_root = settings.data_root / "splits"

    if splits_root.exists():
        for path in splits_root.iterdir():
            if path.is_dir():
                shutil.rmtree(path)

    for split_name in RATIOS:
        split_dir = splits_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for class_dir in sorted(unified_root.iterdir()) if unified_root.exists() else []:
        if not class_dir.is_dir():
            continue
        images = [path for path in class_dir.iterdir() if path.is_file()]
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * RATIOS["train"])
        n_val = int(n_total * RATIOS["val"])
        buckets = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }
        summary[class_dir.name] = {}
        for split_name, split_images in buckets.items():
            target_dir = splits_root / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, target_dir / image_path.name)
            summary[class_dir.name][split_name] = len(split_images)

    report_path = settings.data_root / "reports" / "split_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
