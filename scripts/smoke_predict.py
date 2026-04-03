from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from royadestroyer_ai.config import load_settings
from royadestroyer_ai.inference import Predictor


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/smoke_predict.py <image_path>")
        return 1
    image_path = Path(sys.argv[1])
    settings = load_settings()
    predictor = Predictor(settings.model_dir, settings.image_size, settings.predict_top_k)
    result = predictor.predict(image_path.read_bytes())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
