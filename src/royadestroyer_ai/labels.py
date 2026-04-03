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
