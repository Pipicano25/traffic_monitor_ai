from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
}


@dataclass
class Detection:
    class_id: int
    confidence: float
    box_xyxy: list[float]

    @property
    def label(self) -> str:
        return COCO_CLASS_NAMES.get(self.class_id, f"class_{self.class_id}")


class VehicleCounter:
    def __init__(
        self,
        model_path: str,
        vehicle_class_ids: list[int] | None = None,
        target_class_id: int | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No se encontró el modelo ONNX en {model_path}. "
                "Descárgalo desde /admin/model o ejecuta scripts/download_model.py."
            )

        if vehicle_class_ids is None:
            vehicle_class_ids = [target_class_id if target_class_id is not None else 2]

        self.model_path = model_path
        self.vehicle_class_ids = sorted(set(int(x) for x in vehicle_class_ids))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = self._infer_input_size(self.input_shape)

        # Se actualiza en preprocess. Permite convertir cajas del espacio 640x640
        # al tamaño real de la imagen original.
        self._last_ratio = 1.0
        self._last_pad = (0.0, 0.0)

    @staticmethod
    def _infer_input_size(input_shape: list[Any]) -> int:
        numeric_dims = [d for d in input_shape if isinstance(d, int)]
        if len(numeric_dims) >= 2:
            return int(max(numeric_dims[-2:]))
        return 640

    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("No fue posible leer la imagen enviada")

        original_h, original_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Letterbox: mantiene proporción y evita deformar la vía/vehículos.
        resized, ratio, pad = self._letterbox(
            rgb, new_shape=(self.input_size, self.input_size)
        )
        self._last_ratio = ratio
        self._last_pad = pad

        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
        return tensor, (original_w, original_h)

    @staticmethod
    def _letterbox(
        image: np.ndarray, new_shape: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        h, w = image.shape[:2]
        new_h, new_w = new_shape
        ratio = min(new_w / w, new_h / h)
        resized_w, resized_h = int(round(w * ratio)), int(round(h * ratio))

        resized = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        pad_x = (new_w - resized_w) / 2
        pad_y = (new_h - resized_h) / 2
        x0, y0 = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
        canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
        return canvas, ratio, (pad_x, pad_y)

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        input_tensor, original_size = self.preprocess(image_bytes)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs[0], original_size)
        vehicle_detections = [
            d for d in detections if d.class_id in self.vehicle_class_ids
        ]

        class_counts: dict[str, int] = {}
        for det in vehicle_detections:
            class_counts[det.label] = class_counts.get(det.label, 0) + 1

        return {
            "count": len(vehicle_detections),
            "vehicle_class_ids": self.vehicle_class_ids,
            "vehicle_class_labels": [
                COCO_CLASS_NAMES.get(i, f"class_{i}") for i in self.vehicle_class_ids
            ],
            "class_counts": class_counts,
            "total_detections_debug": len(detections),
            "detections": [
                {
                    "class_id": d.class_id,
                    "label": d.label,
                    "confidence": round(float(d.confidence), 4),
                    "box_xyxy": [
                        float(d.box_xyxy[0]),
                        float(d.box_xyxy[1]),
                        float(d.box_xyxy[2]),
                        float(d.box_xyxy[3]),
                    ],
                }
                for d in vehicle_detections
            ],
        }

    def postprocess(
        self, raw_output: np.ndarray, original_size: tuple[int, int]
    ) -> list[Detection]:
        """Soporta dos salidas comunes:

        1) YOLO26 end-to-end / NMS-free: [1, 300, 6] o [300, 6]
           con columnas x1, y1, x2, y2, score, class_id.
        2) YOLO clásico ONNX: [1, 84, 8400] o [1, 8400, 84]
           con columnas x_center, y_center, width, height, scores_clases.
        """
        output = np.squeeze(raw_output)

        if output.ndim != 2:
            raise ValueError(f"Forma de salida ONNX no soportada: {raw_output.shape}")

        # Caso YOLO26: 6 columnas. Si viene [6, 300], se transpone.
        if 6 in output.shape:
            if output.shape[1] != 6 and output.shape[0] == 6:
                output = output.T
            if output.shape[1] == 6:
                return self._postprocess_e2e(output, original_size)

        # Caso YOLO clásico: [84, 8400] -> [8400, 84].
        if output.shape[0] < output.shape[1]:
            output = output.T
        return self._postprocess_classic(output, original_size)

    def _postprocess_e2e(
        self, output: np.ndarray, original_size: tuple[int, int]
    ) -> list[Detection]:
        boxes_xyxy = output[:, :4].astype(np.float32)
        col4 = output[:, 4].astype(np.float32)
        col5 = output[:, 5].astype(np.float32)

        # Normalmente es [x1,y1,x2,y2,score,class_id].
        # Se deja robusto por si el export cambia a [x1,y1,x2,y2,class_id,score].
        col4_looks_score = float(np.nanmax(col4)) <= 1.5
        col5_looks_score = float(np.nanmax(col5)) <= 1.5

        if col4_looks_score and not col5_looks_score:
            confidences = col4
            class_ids = np.rint(col5).astype(int)
        elif col5_looks_score and not col4_looks_score:
            confidences = col5
            class_ids = np.rint(col4).astype(int)
        else:
            # Fallback más frecuente.
            confidences = col4
            class_ids = np.rint(col5).astype(int)

        mask = confidences >= self.conf_threshold
        boxes_xyxy = boxes_xyxy[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        boxes_xyxy = self._scale_boxes_from_letterbox(boxes_xyxy, original_size)
        return [
            Detection(class_id=int(cid), confidence=float(conf), box_xyxy=box.tolist())
            for cid, conf, box in zip(class_ids, confidences, boxes_xyxy)
        ]

    def _postprocess_classic(
        self, output: np.ndarray, original_size: tuple[int, int]
    ) -> list[Detection]:
        boxes_xywh = output[:, :4].astype(np.float32)
        class_scores = output[:, 4:].astype(np.float32)

        if class_scores.size == 0:
            raise ValueError(f"La salida no contiene scores de clases: {output.shape}")

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy = self._scale_boxes_from_letterbox(boxes_xyxy, original_size)

        keep_indexes = self.nms(boxes_xyxy, confidences, self.iou_threshold)
        return [
            Detection(
                class_id=int(class_ids[idx]),
                confidence=float(confidences[idx]),
                box_xyxy=boxes_xyxy[idx].tolist(),
            )
            for idx in keep_indexes
        ]

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        boxes = np.zeros_like(boxes_xywh, dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        return boxes

    def _scale_boxes_from_letterbox(
        self, boxes_xyxy: np.ndarray, original_size: tuple[int, int]
    ) -> np.ndarray:
        if len(boxes_xyxy) == 0:
            return boxes_xyxy.astype(np.float32)

        original_w, original_h = original_size
        pad_x, pad_y = self._last_pad
        ratio = max(self._last_ratio, 1e-9)

        boxes = boxes_xyxy.astype(np.float32).copy()
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, :4] /= ratio

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)
        return boxes

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / np.maximum(union, 1e-6)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep
