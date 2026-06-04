from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_with_image():
    image_path = Path("tests/data/traffic_test.jpg")

    if not image_path.exists():
        raise AssertionError(f"No existe imagen de prueba: {image_path}")

    with image_path.open("rb") as image_file:
        response = client.post(
            "/predict",
            files={"file": ("traffic_test.jpg", image_file, "image/jpeg")},
        )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert "environment" in data
    assert "count" in data
    assert isinstance(data["count"], int)
    assert data["count"] >= 0
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_predict_endpoint_rejects_non_image_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"esto no es una imagen", "text/plain")},
    )

    assert response.status_code == 400
