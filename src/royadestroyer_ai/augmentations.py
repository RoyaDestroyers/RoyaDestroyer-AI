from __future__ import annotations

import albumentations as A


def build_train_augmentation(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3
            ),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    )


def build_eval_augmentation(image_size: int) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size)])
