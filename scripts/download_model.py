from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os

from app.model_manager import download_model


if __name__ == "__main__":
    model_url = os.environ.get("MODEL_URL")
    model_path = os.environ.get("MODEL_PATH", "models/yolo26n.onnx")
    status = download_model(model_url=model_url, model_path=model_path, overwrite=True)
    print(f"Modelo descargado en: {status['model_path']} ({status['size_bytes']} bytes)")
