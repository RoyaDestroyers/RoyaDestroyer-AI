from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings
from royadestroyer_ai.datasets import detect_split_labels
from royadestroyer_ai.metrics import build_report, save_report
from royadestroyer_ai.model_factory import build_model, resolve_device
from royadestroyer_ai.preprocessing import build_eval_transform


def main() -> int:
    settings = load_settings()
    test_dir = settings.data_root / "splits" / "test"
    model_path = settings.model_dir / "model.pt"
    active_labels = detect_split_labels(test_dir)
    if not active_labels:
        raise RuntimeError("No class directories found in test split")
    device = resolve_device()
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get("model_name", "mobilenetv3_large_100")
    model = build_model(num_classes=len(active_labels), model_name=model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dataset = ImageFolder(
        test_dir,
        transform=build_eval_transform(settings.image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    predictions: list[int] = []
    targets: list[int] = []
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            batch_predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            predictions.extend(batch_predictions)
            targets.extend(labels.numpy().tolist())

    y_pred = np.asarray(predictions)
    y_true = np.asarray(targets)
    report = build_report(y_true, y_pred, active_labels)
    save_report(report, settings.artifacts_root / "metrics" / "evaluation.json")
    print("Saved evaluation report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
