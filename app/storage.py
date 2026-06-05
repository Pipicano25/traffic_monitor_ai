from __future__ import annotations

import os
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from google.cloud import storage
except Exception:  # pragma: no cover - permite pruebas locales sin credenciales GCP
    storage = None


LOCAL_LOG_DIR = Path("logs")

# =========================================================================
# LECTURA DINÁMICA DEL ENTORNO INYECTADO POR EL PIPELINE DE GITHUB ACTIONS
# =========================================================================
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "traffic-mlops-storage")
ENV_STAGE = os.getenv("ENV_STAGE", "dev")  # Tomará "dev" o "prod" según la rama de Git 


def _local_log_path(blob_name: str) -> Path:
    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = blob_name.replace("/", "_") if blob_name else f"predicciones_{ENV_STAGE}.txt"
    return LOCAL_LOG_DIR / safe_name


def registrar_prediccion_automatica(payload: dict[str, Any]) -> None:
    """
    Función adaptada para la rúbrica del profesor:
    Calcula automáticamente las rutas según el entorno y llama a append_prediction_line.
    """
    # Cumple el requerimiento: Guarda dentro de la carpeta logs/ y separa por entorno 
    blob_name = f"logs/predicciones_{ENV_STAGE}.txt"
    
    # Inyectamos el entorno dentro del payload para que quede documentado en el JSON
    payload_con_entorno = {
        "entorno": ENV_STAGE.upper(),
        **payload
    }
    
    # Ejecuta la lógica robusta original de persistencia
    append_prediction_line(
        bucket_name=BUCKET_NAME,
        blob_name=blob_name,
        payload=payload_con_entorno
    )


def append_prediction_line(
    bucket_name: str,
    blob_name: str,
    payload: dict[str, Any],
) -> None:
    """Agrega una línea JSON al TXT de monitoreo.

    Siempre escribe una copia local para que el proyecto se pueda probar sin nube.
    Si se configura el almacenamiento en GCP, también actualiza el TXT en Cloud Storage.
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

    # Inicializar variables fuera para evitar el error "local variable 'blob' referenced before assignment"
    client = None
    bucket = None
    blob = None
    current_content = ""

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        current_content = blob.download_as_text() if blob.exists() else ""
    except Exception as e:
        print(f"⚠️ Alerta en descarga de GCP (Normal si no hay credenciales): {str(e)}")
        current_content = ""

    # Sólo intentamos subir si el cliente de almacenamiento se inicializó con éxito
    if client and bucket and blob:
        try:
            blob.upload_from_string(
                current_content + line,
                content_type="text/plain; charset=utf-8",
            )
            print("✅ ¡Log subido exitosamente a Google Cloud Storage!")
        except Exception as e:
            print(f"❌ Error al subir log al Bucket de Google Cloud: {str(e)}")


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