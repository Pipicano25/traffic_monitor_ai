from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


MIN_MODEL_SIZE_BYTES = 1024 * 1024  # 1 MB: evita aceptar descargas vacías o HTML de error.


def get_model_status(model_path: str, model_url: str) -> dict[str, Any]:
    path = Path(model_path)
    exists = path.exists()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "model_url": model_url,
        "model_path": model_path,
        "exists": exists,
        "size_bytes": size_bytes,
        "ready": exists and size_bytes >= MIN_MODEL_SIZE_BYTES,
    }


def download_model(model_url: str, model_path: str, overwrite: bool = False) -> dict[str, Any]:
    if not model_url:
        raise ValueError("MODEL_URL no está definido")

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size >= MIN_MODEL_SIZE_BYTES and not overwrite:
        return get_model_status(model_path=model_path, model_url=model_url)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with requests.get(model_url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    if not temp_path.exists() or temp_path.stat().st_size < MIN_MODEL_SIZE_BYTES:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            f"La descarga del modelo falló o quedó incompleta. Ruta temporal: {temp_path}"
        )

    temp_path.replace(path)
    return get_model_status(model_path=model_path, model_url=model_url)


def ensure_model_exists(model_url: str, model_path: str) -> dict[str, Any]:
    status = get_model_status(model_path=model_path, model_url=model_url)
    if status["ready"]:
        return status
    return download_model(model_url=model_url, model_path=model_path, overwrite=True)
