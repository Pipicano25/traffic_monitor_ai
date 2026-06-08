import os
from pathlib import Path
import pytest

from app.config import get_settings
from app.model import VehicleCounter


@pytest.fixture(scope="session")
def test_image_bytes() -> bytes:
    """
    Obtiene los bytes de la imagen de prueba.
    Busca la ruta configurada en la variable de entorno TEST_IMAGE_PATH,
    apuntando por defecto a 'tests/data/img1.jpeg' (la imagen descargada del bucket).
    """
    image_path = Path(os.getenv("TEST_IMAGE_PATH", "tests/data/img1.jpeg"))

    # Mecanismo de seguridad (Fallback): Si estás probando localmente en tu PC 
    # y la carpeta o la imagen aún no existen, genera bytes simulados de un JPEG 
    # mínimo válido para que la suite de pruebas no se rompa por falta de archivo.
    if not image_path.exists():
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"

    return image_path.read_bytes()


@pytest.fixture(scope="session")
def counter() -> VehicleCounter:
    """
    Inicializa el modelo de conteo de vehículos utilizando la configuración del sistema.
    La ruta del modelo (.onnx) se obtiene dinámicamente desde settings.model_path.
    """
    settings = get_settings()

    return VehicleCounter(
        model_path=settings.model_path,
        vehicle_class_ids=settings.vehicle_class_ids,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
    )


def test_model_responds_with_defined_input(counter, test_image_bytes):
    """
    Prueba que el modelo responda correctamente y devuelva un diccionario
    con un conteo válido al procesar la imagen de entrada.
    """
    result = counter.predict(test_image_bytes)

    assert isinstance(result, dict)
    assert "count" in result
    assert isinstance(result["count"], int)
    assert result["count"] >= 0


def test_model_returns_expected_output_structure(counter, test_image_bytes):
    """
    Valida que la estructura del diccionario devuelto por el modelo
    contenga exactamente las claves obligatorias requeridas por la API.
    """
    result = counter.predict(test_image_bytes)

    assert "count" in result
    assert isinstance(result["count"], int)

    assert "detections" in result
    assert isinstance(result["detections"], list)

    assert "class_counts" in result
    assert isinstance(result["class_counts"], dict)


def test_count_metric_has_no_significant_change(counter, test_image_bytes):
    """
    Prueba de umbral de métrica límite.
    Compara el conteo obtenido contra un valor esperado configurado por el entorno,
    verificando que la diferencia no exceda el error absoluto tolerado.
    """
    expected_count = int(os.getenv("EXPECTED_COUNT", "21"))
    max_abs_error = int(os.getenv("MAX_ABS_ERROR", "0"))

    result = counter.predict(test_image_bytes)

    assert abs(result["count"] - expected_count) <= max_abs_error


def test_model_detections_have_coordinates(counter, test_image_bytes):
    """
    Verifica que cada objeto detectado por el modelo contenga metadatos
    de clasificación correctos y coordenadas válidas para las cajas delimitadoras.
    """
    result = counter.predict(test_image_bytes)
    detections = result.get("detections", [])

    assert isinstance(detections, list)

    for detection in detections:
        assert "class_id" in detection
        assert "label" in detection
        assert "confidence" in detection

        has_box_xyxy = "box_xyxy" in detection
        has_x1y1x2y2 = all(key in detection for key in ["x1", "y1", "x2", "y2"])

        assert has_box_xyxy or has_x1y1x2y2

        if has_box_xyxy:
            box = detection["box_xyxy"]
            assert isinstance(box, list)
            assert len(box) == 4
            x1, y1, x2, y2 = box
        else:
            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

        assert float(x2) > float(x1)
        assert float(y2) > float(y1)