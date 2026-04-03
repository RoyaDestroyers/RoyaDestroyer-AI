from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings
from royadestroyer_ai.datasets import detect_split_labels
from royadestroyer_ai.model_factory import build_model, resolve_device
from royadestroyer_ai.preprocessing import build_eval_transform, build_train_transform


def main() -> int:
    settings = load_settings()
    train_dir = settings.data_root / "splits" / "train"
    val_dir = settings.data_root / "splits" / "val"
    active_labels = detect_split_labels(train_dir)
    if not active_labels:
        raise RuntimeError("No class directories found in train split")
    device = resolve_device()

    train_dataset = ImageFolder(
        train_dir,
        transform=build_train_transform(settings.image_size),
    )
    val_dataset = ImageFolder(
        val_dir,
        transform=build_eval_transform(settings.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    model = build_model(num_classes=len(active_labels)).to(device)

    y_indices = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(active_labels)),
        y=y_indices,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    # Give roya classes a slight business-priority boost.
    for label_name, multiplier in {"roya_temprana": 1.5, "roya_avanzada": 1.35}.items():
        if label_name in train_dataset.class_to_idx:
            class_weights[train_dataset.class_to_idx[label_name]] *= multiplier

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    best_path = settings.artifacts_root / "checkpoints" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "device": device.type,
        "class_weights": class_weights.detach().cpu().tolist(),
        "class_names": active_labels,
    }

    for epoch in range(settings.epochs_head):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            predictions = logits.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += inputs.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += inputs.size(0)

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)
        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)
        print(
            f"epoch={epoch + 1} train_loss={epoch_train_loss:.4f} "
            f"train_acc={epoch_train_acc:.4f} val_loss={epoch_val_loss:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": active_labels,
                    "model_name": "mobilenetv3_large_100",
                    "image_size": settings.image_size,
                },
                best_path,
            )

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": active_labels,
            "model_name": "mobilenetv3_large_100",
            "image_size": settings.image_size,
        },
        settings.model_dir / "model.pt",
    )
    (settings.model_dir / "labels.json").write_text(
        json.dumps(active_labels, indent=2), encoding="utf-8"
    )
    history_path = settings.artifacts_root / "metrics" / "train_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {settings.model_dir / 'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
