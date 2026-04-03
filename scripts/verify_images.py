from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings


def main() -> int:
    settings = load_settings()
    invalid: list[str] = []
    unified_root = settings.data_root / "unified"
    for image_path in unified_root.rglob("*"):
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            invalid.append(str(image_path))
    report = {"invalid_count": len(invalid), "invalid_files": invalid[:100]}
    report_path = settings.data_root / "reports" / "verify_images.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
