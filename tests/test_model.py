import os
from pathlib import Path

import pytest

from app.config import get_settings
from app.model import VehicleCounter


@pytest.fixture(scope="session")
def test_image_bytes() -> bytes:
    image_path = Path(os.getenv("TEST_IMAGE_PATH", "tests/data/traffic_test.jpg"))

    if not image_path.exists():
        raise AssertionError(f"No existe imagen de prueba: {image_path}")

    return image_path.read_bytes()


@pytest.fixture(scope="session")
def counter() -> VehicleCounter:
    settings = get_settings()

    return VehicleCounter(
        model_path=settings.model_path,
        vehicle_class_ids=settings.vehicle_class_ids,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
    )


def test_model_responds_with_defined_input(counter, test_image_bytes):
    result = counter.predict(test_image_bytes)

    assert isinstance(result, dict)
    assert "count" in result
    assert isinstance(result["count"], int)
    assert result["count"] >= 0


def test_model_returns_expected_output_structure(counter, test_image_bytes):
    result = counter.predict(test_image_bytes)

    assert "count" in result
    assert isinstance(result["count"], int)

    assert "detections" in result
    assert isinstance(result["detections"], list)

    assert "class_counts" in result
    assert isinstance(result["class_counts"], dict)


def test_count_metric_has_no_significant_change(counter, test_image_bytes):
    """
    Esta prueba reemplaza el expected_metrics.json.

    Ajusta EXPECTED_COUNT según la cantidad aproximada de carros/vehículos
    que esperas detectar en tu imagen de prueba.
    """

    expected_count = int(os.getenv("EXPECTED_COUNT", "20"))
    max_abs_error = int(os.getenv("MAX_ABS_ERROR", "3"))

    result = counter.predict(test_image_bytes)

    assert abs(result["count"] - expected_count) <= max_abs_error


def test_model_detections_have_coordinates(counter, test_image_bytes):
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
