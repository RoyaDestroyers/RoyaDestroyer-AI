from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings


def main() -> int:
    settings = load_settings()
    unified_root = settings.data_root / "unified"
    counts: dict[str, int] = {}
    for class_dir in sorted(unified_root.iterdir()) if unified_root.exists() else []:
        if class_dir.is_dir():
            counts[class_dir.name] = sum(1 for path in class_dir.iterdir() if path.is_file())
    report_path = settings.data_root / "reports" / "distribution.json"
    report_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
