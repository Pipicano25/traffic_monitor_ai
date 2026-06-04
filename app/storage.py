from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from google.cloud import storage
except Exception:  # pragma: no cover - permite pruebas locales sin credenciales GCP
    storage = None


LOCAL_LOG_DIR = Path("logs")


def _local_log_path(blob_name: str) -> Path:
    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = blob_name.replace("/", "_") if blob_name else "predicciones_local.txt"
    return LOCAL_LOG_DIR / safe_name


def append_prediction_line(
    bucket_name: str,
    blob_name: str,
    payload: dict[str, Any],
) -> None:
    """Agrega una línea JSON al TXT de monitoreo.

    Siempre escribe una copia local para que el proyecto se pueda probar sin nube.
    Si se configura PREDICTIONS_BUCKET, también actualiza el TXT en Cloud Storage.
    """
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    line = json.dumps(event, ensure_ascii=False) + "\n"

    local_path = _local_log_path(blob_name)
    with local_path.open("a", encoding="utf-8") as file:
        file.write(line)

    if not bucket_name or not blob_name or storage is None:
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    try:
        current_content = blob.download_as_text() if blob.exists() else ""
    except Exception:
        current_content = ""

    blob.upload_from_string(
        current_content + line,
        content_type="text/plain; charset=utf-8",
    )


def read_predictions_text(bucket_name: str, blob_name: str) -> str:
    """Lee el TXT de predicciones desde Cloud Storage o desde el archivo local."""
    if bucket_name and blob_name and storage is not None:
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if blob.exists():
                return blob.download_as_text()
        except Exception:
            pass

    local_path = _local_log_path(blob_name)
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")
    return ""
