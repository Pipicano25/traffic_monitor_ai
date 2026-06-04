import cv2
import numpy as np


def draw_detections_on_image(image_bytes: bytes, detections: list[dict]) -> bytes:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No fue posible leer la imagen.")

    height, width = image.shape[:2]

    thickness = max(2, int(min(width, height) / 350))
    font_scale = max(0.45, min(width, height) / 1100)

    for det in detections:
        x1 = int(max(0, min(width, det["x1"])))
        y1 = int(max(0, min(height, det["y1"])))
        x2 = int(max(0, min(width, det["x2"])))
        y2 = int(max(0, min(height, det["y2"])))

        label = det.get("label", "vehicle")
        confidence = float(det.get("confidence", 0.0))
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness,
        )

        text_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        text_w, text_h = text_size
        text_y1 = max(0, y1 - text_h - baseline - 6)
        text_y2 = max(text_h + baseline + 6, y1)

        cv2.rectangle(
            image,
            (x1, text_y1),
            (min(width, x1 + text_w + 8), text_y2),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            image,
            text,
            (x1 + 4, text_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )

    if not success:
        raise ValueError("No fue posible generar la imagen anotada.")

    return encoded.tobytes()
