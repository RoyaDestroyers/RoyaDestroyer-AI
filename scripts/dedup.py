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
    seen: dict[str, str] = {}
    removed: list[str] = []

    for image_path in sorted(unified_root.rglob("*")):
        if not image_path.is_file():
            continue
        digest = file_hash(image_path)
        if digest in seen:
            image_path.unlink()
            removed.append(str(image_path))
        else:
            seen[digest] = str(image_path)

    report = {"removed_count": len(removed), "removed_files": removed[:100]}
    report_path = settings.data_root / "reports" / "dedup.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
