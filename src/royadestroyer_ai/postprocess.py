from __future__ import annotations

from royadestroyer_ai.labels import LABELS

RULES = {
    "roya_temprana": {
        "severity": "moderado",
        "symptoms": ["manchas amarillas", "polvillo naranja en enves"],
        "recommendations": ["aislar hojas afectadas", "aplicar manejo fitosanitario"],
    },
    "roya_avanzada": {
        "severity": "grave",
        "symptoms": ["decoloracion extensa", "perdida de tejido foliar"],
        "recommendations": ["intervencion inmediata", "revisar lote completo"],
    },
    "cercospora": {
        "severity": "moderado",
        "symptoms": ["manchas circulares", "tejido necrotico"],
        "recommendations": ["inspeccion dirigida", "aplicar tratamiento segun protocolo"],
    },
    "phoma": {
        "severity": "moderado",
        "symptoms": ["mancha marron", "bordes oscuros"],
        "recommendations": ["confirmar progresion", "aislar area afectada"],
    },
    "minador": {
        "severity": "moderado",
        "symptoms": ["galerias en hoja", "perdida de tejido"],
        "recommendations": ["controlar plaga", "monitorear hojas cercanas"],
    },
    "hoja_sana": {
        "severity": "ninguna",
        "symptoms": [],
        "recommendations": ["continuar monitoreo rutinario"],
    },
    "imagen_invalida": {
        "severity": "ninguna",
        "symptoms": [],
        "recommendations": ["tomar nueva foto con mejor enfoque e iluminacion"],
    },
}


def enrich_prediction(label: str) -> dict:
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    return RULES[label]
