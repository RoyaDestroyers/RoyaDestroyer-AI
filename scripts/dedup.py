from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings


def file_hash(path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    settings = load_settings()
    unified_root = settings.data_root / "unified"
    seen_within_class: dict[tuple[str, str], str] = {}
    seen_global: dict[str, tuple[str, str]] = {}
    removed: list[str] = []
    cross_class_duplicates: list[dict[str, str]] = []

    for class_dir in sorted(unified_root.iterdir()) if unified_root.exists() else []:
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            digest = file_hash(image_path)
            class_key = (class_name, digest)
            if class_key in seen_within_class:
                image_path.unlink()
                removed.append(str(image_path))
                continue

            seen_within_class[class_key] = str(image_path)
            if digest in seen_global:
                first_class, first_path = seen_global[digest]
                if first_class != class_name:
                    cross_class_duplicates.append(
                        {
                            "hash": digest,
                            "first_class": first_class,
                            "first_path": first_path,
                            "second_class": class_name,
                            "second_path": str(image_path),
                        }
                    )
            else:
                seen_global[digest] = (class_name, str(image_path))

    report = {
        "removed_count": len(removed),
        "removed_files": removed[:100],
        "cross_class_duplicate_count": len(cross_class_duplicates),
        "cross_class_duplicates": cross_class_duplicates[:100],
    }
    report_path = settings.data_root / "reports" / "dedup.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
