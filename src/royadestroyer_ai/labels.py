from __future__ import annotations

LABELS = [
    "roya_temprana",
    "roya_avanzada",
    "cercospora",
    "phoma",
    "minador",
    "hoja_sana",
    "imagen_invalida",
]

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def build_index_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label
