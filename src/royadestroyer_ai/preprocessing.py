from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_from_bytes(payload: bytes, image_size: int) -> np.ndarray:
    image = Image.open(BytesIO(payload)).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def add_batch_dimension(image_array: np.ndarray) -> np.ndarray:
    return np.expand_dims(image_array, axis=0)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_tensor_from_bytes(payload: bytes, image_size: int) -> torch.Tensor:
    image = Image.open(BytesIO(payload)).convert("RGB")
    transform = build_eval_transform(image_size)
    return transform(image).unsqueeze(0)
