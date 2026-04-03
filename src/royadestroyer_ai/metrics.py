from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def build_report(y_true, y_pred, labels: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return report


def save_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
