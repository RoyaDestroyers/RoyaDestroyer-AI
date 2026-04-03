from __future__ import annotations

from pathlib import Path

import pandas as pd

from royadestroyer_ai.labels import LABELS


def collect_split_dataframe(split_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for label in LABELS:
        label_dir = split_root / label
        if not label_dir.exists():
            continue
        for image_path in sorted(label_dir.glob("*")):
            if image_path.is_file():
                rows.append({"path": str(image_path), "label": label})
    return pd.DataFrame(rows)
