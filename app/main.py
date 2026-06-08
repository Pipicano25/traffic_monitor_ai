from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.model import VehicleCounter
from app.model_manager import download_model, ensure_model_exists, get_model_status
# Importamos la nueva función asíncrona de guardado automático y la variable de entorno
from app.storage import registrar_prediccion_automatica, ENV_STAGE, read_predictions_text
from app.ui import render_model_admin
from app.visualization import draw_detections_on_image

os.makedirs("outputs", exist_ok=True)
os.makedirs("app/static", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

app = FastAPI(
    title="Sistema inteligente de monitoreo vehicular",
    description="API e interfaz web para conteo y señalización de vehículos usando ONNX.",
    version="3.1.0",
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@lru_cache
def get_counter() -> VehicleCounter:
    settings = get_settings()

    ensure_model_exists(
        model_url=settings.model_url,
        model_path=settings.model_path,
    )

    return VehicleCounter(
        model_path=settings.model_path,
        vehicle_class_ids=settings.vehicle_class_ids,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
    )


def normalize_detections(detections: list[dict]) -> list[dict]:
    normalized = []

    for det in detections:
        if "box_xyxy" in det:
            x1, y1, x2, y2 = det["box_xyxy"]
        elif all(k in det for k in ["x1", "y1", "x2", "y2"]):
            x1 = det["x1"]
            y1 = det["y1"]
            x2 = det["x2"]
            y2 = det["y2"]
        else:
            continue

        normalized.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(det.get("confidence", 0.0)),
                "class_id": int(det.get("class_id", -1)),
                "label": str(det.get("label", "vehicle")),
            }
        )

    return normalized


def build_prediction_response(result: dict) -> dict:
    settings = get_settings()
    detections = normalize_detections(result.get("detections", []))

    return {
        "request_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": ENV_STAGE.upper(),
        "model_path": settings.model_path,
        "vehicle_class_ids": settings.vehicle_class_ids,
        "vehicle_class_labels": result.get("vehicle_class_labels", []),
        "confidence_threshold": settings.conf_threshold,
        "count": result["count"],
        "class_counts": result.get("class_counts", {}),
        "total_detections_debug": result.get("total_detections_debug", 0),
        "detections": detections,
    }


def register_prediction(response: dict) -> None:
    """Extrae las métricas esenciales y las registra de forma automática según la rama."""
    payload = {
        "request_id": response["request_id"],
        "timestamp_utc": response["timestamp_utc"],
        "environment": ENV_STAGE,
        "count": response["count"],
        "class_counts": response["class_counts"],
        "confidence_threshold": response["confidence_threshold"],
    }

    # REQUERIMIENTO COMPLETADO: Guarda de manera inteligente en logs/predicciones_dev.txt o _prod.txt 
    registrar_prediccion_automatica(payload)


def save_result_files(response: dict, annotated_image_bytes: bytes) -> dict:
    request_id = response["request_id"]

    image_filename = f"{request_id}.jpg"
    json_filename = f"{request_id}.json"
    txt_filename = f"{request_id}.txt"

    image_path = os.path.join("outputs", image_filename)
    json_path = os.path.join("outputs", json_filename)
    txt_path = os.path.join("outputs", txt_filename)

    with open(image_path, "wb") as f:
        f.write(annotated_image_bytes)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(response, ensure_ascii=False))
        f.write("\n")

    return {
        "image_url": f"/outputs/{image_filename}",
        "json_url": f"/outputs/{json_filename}",
        "txt_url": f"/outputs/{txt_filename}",
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    settings = get_settings()
    status = get_model_status(settings.model_path, settings.model_url)

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "environment": f"{ENV_STAGE} ({ENV_STAGE.upper()})",
            "vehicle_class_ids": settings.vehicle_class_ids,
            "conf_threshold": settings.conf_threshold,
            "model_ready": bool(status.get("ready")),
            "model_path": settings.model_path,
            "model_size_mb": round(int(status.get("size_bytes", 0)) / (1024 * 1024), 2),
        },
    )


@app.post("/ui/predict", response_class=HTMLResponse)
async def predict_from_ui(
    request: Request, file: UploadFile = File(...)
) -> HTMLResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "El archivo debe ser una imagen.",
            },
            status_code=400,
        )

    try:
        image_bytes = await file.read()

        result = get_counter().predict(image_bytes)
        response = build_prediction_response(result)

        # Aquí se llama al registro dinámico en el bucket 
        register_prediction(response)

        annotated_image_bytes = draw_detections_on_image(
            image_bytes=image_bytes,
            detections=response["detections"],
        )

        files = save_result_files(
            response=response,
            annotated_image_bytes=annotated_image_bytes,
        )

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "response": response,
                "count": response["count"],
                "class_counts": response.get("class_counts", {}),
                "detections": response.get("detections", []),
                "image_url": files["image_url"],
                "json_url": files["json_url"],
                "txt_url": files["txt_url"],
            },
        )

    except Exception as exc:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": str(exc),
            },
            status_code=500,
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    try:
        image_bytes = await file.read()

        result = get_counter().predict(image_bytes)
        response = build_prediction_response(result)

        # Aquí se llama al registro dinámico en el bucket 
        register_prediction(response)

        return response

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/download-history")
def download_history() -> Response:
    """Descarga de forma dinámica el archivo txt de logs según el contenedor que responda."""
    bucket_name = os.getenv("GCP_BUCKET_NAME", "traffic-mlops-storage")
    blob_name = f"logs/predicciones_{ENV_STAGE}.txt"

    content = read_predictions_text(
        bucket_name=bucket_name,
        blob_name=blob_name,
    )

    if not content:
        content = f"Aún no existen predicciones registradas para el ambiente: {ENV_STAGE.upper()}.\n"

    filename = f"predicciones_{ENV_STAGE}.txt"

    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.get("/model-status")
def model_status() -> dict:
    settings = get_settings()
    return get_model_status(settings.model_path, settings.model_url)


@app.get("/admin/model", response_class=HTMLResponse)
def model_admin() -> HTMLResponse:
    settings = get_settings()
    status = get_model_status(settings.model_path, settings.model_url)
    return HTMLResponse(render_model_admin(status=status, message=None))


@app.post("/admin/download-model", response_class=HTMLResponse)
def download_model_from_ui() -> HTMLResponse:
    settings = get_settings()

    try:
        status = download_model(
            model_url=settings.model_url,
            model_path=settings.model_path,
            overwrite=True,
        )

        get_counter.cache_clear()

        return HTMLResponse(
            render_model_admin(
                status=status,
                message="Modelo descargado/verificado correctamente.",
            )
        )

    except Exception as exc:
        status = get_model_status(settings.model_path, settings.model_url)

        return HTMLResponse(
            render_model_admin(
                status=status,
                message=f"Error al descargar el modelo: {exc}",
            ),
            status_code=500,
        )


@app.get("/health")
def health() -> dict:
    settings = get_settings()

    return {
        "status": "ok",
        "environment": ENV_STAGE.upper(),
        "stage_mlops": ENV_STAGE.upper(),
        "model_path": settings.model_path,
        "interface": "enabled",
        "annotation": "enabled",
    }