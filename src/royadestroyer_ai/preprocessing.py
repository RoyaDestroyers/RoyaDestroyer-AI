from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def load_image_from_bytes(payload: bytes, image_size: int) -> np.ndarray:
    image = Image.open(BytesIO(payload)).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def add_batch_dimension(image_array: np.ndarray) -> np.ndarray:
    return np.expand_dims(image_array, axis=0)
