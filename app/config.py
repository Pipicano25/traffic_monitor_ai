from functools import lru_cache
from pydantic import BaseModel
import os


def _parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(','):
        item = item.strip()
        if item:
            result.append(int(item))
    return result


class Settings(BaseModel):
    model_url: str = os.getenv(
        "MODEL_URL",
        "https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx",
    )
    model_path: str = os.getenv("MODEL_PATH", "models/yolo26n.onnx")
    environment: str = os.getenv("ENVIRONMENT", "dev")
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.25"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # Para monitoreo vehicular normalmente conviene contar varias clases COCO:
    # 2=car, 3=motorcycle, 5=bus, 7=truck.
    # Si solo quieres carros estrictamente, usa VEHICLE_CLASS_IDS=2.
    vehicle_class_ids: list[int] = _parse_int_list(os.getenv("VEHICLE_CLASS_IDS", os.getenv("TARGET_CLASS_ID", "2")))

    predictions_bucket: str = os.getenv("PREDICTIONS_BUCKET", "")
    predictions_blob: str = os.getenv("PREDICTIONS_BLOB", "predicciones_dev.txt")


@lru_cache
def get_settings() -> Settings:
    return Settings()
