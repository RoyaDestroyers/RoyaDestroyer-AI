from __future__ import annotations

import json
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings

MAPPINGS = {
    "jmuben": {
        "Rust": "roya_temprana",
        "Cercospora": "cercospora",
        "Phoma": "phoma",
        "Miner": "minador",
        "Healthy": "hoja_sana",
    },
    "jmuben2": {
        "Rust": "roya_temprana",
        "Cercospora": "cercospora",
        "Phoma": "phoma",
        "Miner": "minador",
        "Healthy": "hoja_sana",
    },
    "clr_eafit": {
        "0": "hoja_sana",
        "1": "roya_temprana",
        "2": "roya_temprana",
        "3": "roya_avanzada",
        "4": "roya_avanzada",
    },
    "rocole": {
        "Rust_Level_1": "roya_temprana",
        "Rust_Level_2": "roya_temprana",
        "Rust_Level_3": "roya_avanzada",
        "Rust_Level_4": "roya_avanzada",
        "Healthy": "hoja_sana",
    },
    "uganda": {
        "CLR": "roya_temprana",
        "Phoma": "phoma",
        "Healthy": "hoja_sana",
    },
    "bracol": {
        "Rust": "roya_temprana",
        "Miner": "minador",
        "Brown Leaf Spot": "phoma",
        "Cercospora Leaf Spot": "cercospora",
        "Healthy": "hoja_sana",
    },
    "own_invalid": {
        "imagen_invalida": "imagen_invalida",
    },
}


def copy_images(src_dir: Path, dst_dir: Path, prefix: str) -> int:
    count = 0
    for image_path in sorted(src_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        target = dst_dir / f"{prefix}_{image_path.name}"
        if target.exists():
            target = dst_dir / f"{prefix}_{count:06d}_{image_path.name}"
        shutil.copy2(image_path, target)
        count += 1
    return count


def main() -> int:
    settings = load_settings()
    raw_root = settings.data_root / "raw"
    unified_root = settings.data_root / "unified"
    summary: dict[str, int] = {}
    if unified_root.exists():
        for path in unified_root.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
    unified_root.mkdir(parents=True, exist_ok=True)

    for dataset_name, class_map in MAPPINGS.items():
        dataset_root = raw_root / dataset_name
        if not dataset_root.exists():
            continue
        for source_class, target_class in class_map.items():
            source_root = dataset_root / source_class
            if not source_root.exists():
                continue
            target_root = unified_root / target_class
            target_root.mkdir(parents=True, exist_ok=True)
            copied = copy_images(source_root, target_root, dataset_name)
            summary[target_class] = summary.get(target_class, 0) + copied

    report_path = settings.data_root / "reports" / "unify_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
