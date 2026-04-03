from __future__ import annotations

from pathlib import Path

import pandas as pd

def collect_split_dataframe(split_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for label in detect_split_labels(split_root):
        label_dir = split_root / label
        if not label_dir.exists():
            continue
        for image_path in sorted(label_dir.glob("*")):
            if image_path.is_file():
                rows.append({"path": str(image_path), "label": label})
    return pd.DataFrame(rows)


def detect_split_labels(split_root: Path) -> list[str]:
    return sorted(
        [
            path.name
            for path in split_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
    )
