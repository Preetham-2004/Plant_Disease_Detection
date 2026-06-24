import base64
import io
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from ultralytics import YOLO

from ..config import CLASSIFIER_MODEL_PATH, DEFAULT_YOLO_CONFIDENCE, YOLO_MODEL_PATH


CLASS_LABELS = ["Healthy", "Moderate", "Severe"]
PART_PRIORITIES = {"leaf": 4, "fruit": 3, "flower": 2, "stem": 1}
ANNOTATION_COLORS = {
    "leaf": "#3ba55d",
    "fruit": "#f28f3b",
    "flower": "#c05c7a",
    "stem": "#5f7c8a",
}
MIN_PART_CONFIDENCE = 0.55
MIN_PART_MARGIN = 0.10
MIN_PART_AREA_RATIO = 0.01


@dataclass
class SelectedDetection:
    index: int
    label: str
    confidence: float
    bbox: list[int]
    segmentation_mode: str
    accepted: bool
    rejection_reason: str | None = None


class PlantDiseasePipeline:
    def __init__(
        self,
        yolo_model_path=YOLO_MODEL_PATH,
        classifier_model_path=CLASSIFIER_MODEL_PATH,
        yolo_confidence=DEFAULT_YOLO_CONFIDENCE,
    ) -> None:
        self.yolo_model_path = str(yolo_model_path)
        self.classifier_model_path = str(classifier_model_path)
        self.yolo_confidence = yolo_confidence

        self.detector = YOLO(self.yolo_model_path)
        self.classifier = tf.keras.models.load_model(self.classifier_model_path)

        _, height, width, channels = self.classifier.input_shape
        self.classifier_input_shape = [height, width, channels]
        self.yolo_task = self.detector.task
        self.part_names = {int(k): v for k, v in self.detector.names.items()}

    def predict(self, image_bytes: bytes, filename: str) -> dict:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)

        detection_result = self.detector.predict(
            source=image_np,
            conf=self.yolo_confidence,
            verbose=False,
        )[0]

        selected_detection = self._select_detection(detection_result, pil_image.size)
        segmented_image = self._extract_region(pil_image, detection_result, selected_detection)
        annotated_image = self._annotate_image(pil_image, detection_result, selected_detection)

        if selected_detection.accepted:
            probabilities = self._classify(segmented_image)
            probability_payload = [
                {"label": label, "confidence": float(score)}
                for label, score in zip(CLASS_LABELS, probabilities)
            ]
            best_idx = int(np.argmax(probabilities))
            disease_label = CLASS_LABELS[best_idx]
            disease_confidence = float(probabilities[best_idx])
        else:
            probability_payload = [{"label": label, "confidence": 0.0} for label in CLASS_LABELS]
            disease_label = "Uncertain"
            disease_confidence = 0.0

        detections = self._serialize_detections(detection_result, selected_detection)

        return {
            "filename": filename,
            "disease_label": disease_label,
            "disease_confidence": disease_confidence,
            "probabilities": probability_payload,
            "detections": detections,
            "selected_part": selected_detection.label,
            "segmentation_mode": selected_detection.segmentation_mode,
            "input_resolution": self.classifier_input_shape[:2],
            "advisory": self._build_advisory(selected_detection),
            "original_image": self._to_data_url(pil_image),
            "segmented_image": self._to_data_url(segmented_image),
            "annotated_image": self._to_data_url(annotated_image),
        }

    def _select_detection(self, result, image_size: tuple[int, int]) -> SelectedDetection:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            width, height = image_size
            return SelectedDetection(
                index=-1,
                label="no-part-detected",
                confidence=0.0,
                bbox=[0, 0, width, height],
                segmentation_mode="no-detection",
                accepted=False,
                rejection_reason="No plant part was detected confidently enough to route to the disease model.",
            )

        image_width, image_height = image_size
        image_area = max(image_width * image_height, 1)
        ranking = []
        for idx, box in enumerate(boxes):
            cls_idx = int(box.cls.item())
            label = self.part_names.get(cls_idx, f"class-{cls_idx}")
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            area = max(x2 - x1, 1) * max(y2 - y1, 1)
            area_ratio = area / image_area
            ranking.append(
                {
                    "priority": PART_PRIORITIES.get(label, 0),
                    "confidence": confidence,
                    "area": area,
                    "area_ratio": area_ratio,
                    "index": idx,
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        ranking.sort(
            key=lambda item: (
                item["confidence"],
                item["area_ratio"],
                item["priority"],
                item["area"],
            ),
            reverse=True,
        )
        best = ranking[0]
        segmentation_mode = "mask" if getattr(result, "masks", None) is not None else "bbox-crop"
        accepted = True
        rejection_reason = None

        if best["confidence"] < MIN_PART_CONFIDENCE:
            accepted = False
            rejection_reason = (
                f"Best detected part '{best['label']}' had low confidence "
                f"({best['confidence']:.2f} < {MIN_PART_CONFIDENCE:.2f})."
            )
        elif best["area_ratio"] < MIN_PART_AREA_RATIO:
            accepted = False
            rejection_reason = (
                f"Best detected part '{best['label']}' covered too little of the image "
                f"({best['area_ratio']:.3f} < {MIN_PART_AREA_RATIO:.3f})."
            )
        elif len(ranking) > 1:
            runner_up = ranking[1]
            confidence_gap = best["confidence"] - runner_up["confidence"]
            if best["label"] != runner_up["label"] and confidence_gap < MIN_PART_MARGIN:
                accepted = False
                rejection_reason = (
                    f"Part detection was ambiguous between '{best['label']}' and "
                    f"'{runner_up['label']}' ({best['confidence']:.2f} vs "
                    f"{runner_up['confidence']:.2f})."
                )

        return SelectedDetection(
            index=best["index"],
            label=best["label"],
            confidence=best["confidence"],
            bbox=best["bbox"],
            segmentation_mode=segmentation_mode,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def _extract_region(self, image: Image.Image, result, selected: SelectedDetection) -> Image.Image:
        if selected.index < 0:
            return image.copy()

        x1, y1, x2, y2 = self._clip_bbox(selected.bbox, image.size)
        cropped = image.crop((x1, y1, x2, y2))

        masks = getattr(result, "masks", None)
        if selected.segmentation_mode != "mask" or masks is None:
            return cropped

        mask_canvas = Image.new("L", image.size, 0)
        if len(masks.xy) <= selected.index:
            return cropped

        polygon = masks.xy[selected.index]
        polygon_points = [(float(x), float(y)) for x, y in polygon]
        ImageDraw.Draw(mask_canvas).polygon(polygon_points, fill=255)

        masked_image = Image.new("RGB", image.size, (0, 0, 0))
        masked_image.paste(image, mask=mask_canvas)
        return masked_image.crop((x1, y1, x2, y2))

    def _annotate_image(self, image: Image.Image, result, selected: SelectedDetection) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return annotated

        for idx, box in enumerate(boxes):
            cls_idx = int(box.cls.item())
            label = self.part_names.get(cls_idx, f"class-{cls_idx}")
            confidence = float(box.conf.item())
            color = ANNOTATION_COLORS.get(label, "#ffffff")
            x1, y1, x2, y2 = self._clip_bbox(
                [int(round(v)) for v in box.xyxy[0].tolist()],
                image.size,
            )
            width = 6 if idx == selected.index else 3
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            draw.text((x1 + 8, max(10, y1 - 22)), f"{label} {confidence:.2f}", fill=color)

        return annotated

    def _classify(self, image: Image.Image) -> np.ndarray:
        height, width, _ = self.classifier_input_shape
        resized = image.resize((width, height))
        array = np.asarray(resized, dtype=np.float32)
        array = tf.keras.applications.inception_v3.preprocess_input(array)
        batch = np.expand_dims(array, axis=0)
        probabilities = self.classifier.predict(batch, verbose=0)[0]
        return probabilities.astype(float)

    def _serialize_detections(self, result, selected: SelectedDetection) -> list[dict]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for idx, box in enumerate(boxes):
            cls_idx = int(box.cls.item())
            label = self.part_names.get(cls_idx, f"class-{cls_idx}")
            confidence = float(box.conf.item())
            bbox = [int(round(v)) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                    "selected": idx == selected.index,
                }
            )
        return detections

    def _clip_bbox(self, bbox: list[int], image_size: tuple[int, int]) -> list[int]:
        width, height = image_size
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return [x1, y1, x2, y2]

    def _build_advisory(self, selected: SelectedDetection) -> str:
        if not selected.accepted:
            return selected.rejection_reason or (
                "The detected plant part was not reliable enough for disease classification."
            )
        if selected.segmentation_mode == "mask":
            return "YOLO provided a segmentation mask before disease classification."
        if selected.segmentation_mode == "bbox-crop":
            return "The current YOLO checkpoint is a detection model, so the backend used the highest-confidence plant-part crop instead of a pixel mask."
        return "The detected plant part was accepted and routed to the disease classifier."

    def _to_data_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
