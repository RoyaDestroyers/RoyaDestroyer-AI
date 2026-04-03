from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings


def main() -> int:
    settings = load_settings()
    free_bytes = shutil.disk_usage(settings.project_root).free
    report = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "project_root": str(settings.project_root),
        "data_root": str(settings.data_root),
        "artifacts_root": str(settings.artifacts_root),
        "free_gb": round(free_bytes / (1024**3), 2),
        "python_ok": sys.version_info[:2] == (3, 10),
        "notes": [],
    }
    if not report["python_ok"]:
        report["notes"].append("Recommended Python is 3.10 for TensorFlow compatibility.")
    if report["free_gb"] < 50:
        report["notes"].append("Free disk is under 50 GB; dataset work may be constrained.")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
