"""
Ensemble inference service aligned with the CLI behavior in `scripts/inference/inference_ensemble.py`.
"""

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = {
    0: 'eggplant_fruit',
    1: 'eggplant_leaf',
    2: 'potato_fruit',
    3: 'potato_leaf',
    4: 'tomato_fruit',
    5: 'tomato_leaf'
}


@dataclass
class EnsembleDetection:
    """Single detection result from ensemble."""
    box: list[float]  # [x1, y1, x2, y2]
    conf: float
    cls_id: int
    cls_name: str
    consensus: bool  # True if both models agree
    agreement: int  # Number of models that detected this


class EnsembleDetector:
    """Combine predictions from original and finetuned models."""
    
    def __init__(self, model1_path, model2_path, conf=0.5):
        """Initialize ensemble with the same weights and defaults as the CLI script."""
        self.model1 = YOLO(model1_path)
        self.model1_weight = 0.6  # Original model has higher mAP50 (0.8162)

        self.model2 = YOLO(model2_path)
        self.model2_weight = 0.4  # Finetuned model has lower mAP50 (0.8000)

        self.conf = conf
        self.class_names = CLASS_NAMES
    
    def predict_single(self, image_source, model, conf):
        """Run prediction with a single model on image bytes or path."""
        result = model.predict(source=image_source, conf=conf, verbose=False)
        return result[0]
    
    def ensemble_predict(self, image_source):
        """Return merged detections using the same rules as the CLI ensemble script."""
        # Run both models
        result1 = self.predict_single(image_source, self.model1, self.conf)
        result2 = self.predict_single(image_source, self.model2, self.conf)
        
        detections = []
        
        # Collect all detections from both models
        for result, model_id, weight in [
            (result1, 0, self.model1_weight),
            (result2, 1, self.model2_weight)
        ]:
            if len(result.boxes) > 0:
                for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    detections.append({
                        'box': box.cpu().numpy(),
                        'conf': float(conf),
                        'cls_id': int(cls_id),
                        'model_id': model_id,
                        'weight': weight
                    })
        
        # Match the CLI implementation exactly.
        merged_detections = self._merge_detections(detections)
        
        # Convert to EnsembleDetection objects
        results = []
        for det in merged_detections:
            results.append(EnsembleDetection(
                box=det['box'].tolist(),
                conf=det['conf'],
                cls_id=det['cls_id'],
                cls_name=self.class_names.get(det['cls_id'], 'unknown'),
                consensus=det['consensus'],
                agreement=det['agreement']
            ))
        
        return results
    
    def _merge_detections(self, detections):
        """Merge detections exactly like `scripts/inference/inference_ensemble.py`."""
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Find matching detections from other model
            matches = [det1]
            
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                # The CLI script matches on class + IoU only.
                if det1['cls_id'] == det2['cls_id']:
                    iou = self._calc_iou(det1['box'], det2['box'])
                    if iou > 0.3:
                        matches.append(det2)
                        used.add(j)
            
            # Combine matches
            if len(matches) > 1:
                confidences = [m['conf'] * m['weight'] for m in matches]
                final_conf = sum(confidences) / sum(m['weight'] for m in matches)
                final_conf = min(final_conf * 1.2, 1.0)

                merged.append({
                    'box': matches[0]['box'],
                    'conf': final_conf,
                    'cls_id': matches[0]['cls_id'],
                    'consensus': True,
                    'agreement': len(matches)
                })
            else:
                merged.append({
                    'box': det1['box'],
                    'conf': det1['conf'] * det1['weight'],
                    'cls_id': det1['cls_id'],
                    'consensus': False,
                    'agreement': 1
                })
            
            used.add(i)
        
        return merged
    
    def _calc_iou(self, box1, box2):
        """Calculate IoU between two boxes (xyxy format)."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def render_annotated_image(self, image_path: str | Path, detections: list[EnsembleDetection]) -> bytes:
        """Draw the same overlay style used by the CLI script."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image for annotation: {image_path}")

        img_h, img_w = image.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.box)
            color = (0, 255, 0) if detection.consensus else (0, 165, 255)
            thickness = 3 if detection.consensus else 2

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            label = f"{detection.cls_name} {detection.conf:.1%}"


            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness_text = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]

            padding = 6
            bg_y1 = max(0, y1 - text_size[1] - padding)
            bg_x2 = min(img_w, x1 + text_size[0] + padding)
            bg_x1 = max(0, x1)

            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, y1), color, -1)
            cv2.putText(image, label, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thickness_text)

        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode annotated ensemble image.")
        return encoded.tobytes()

    def save_annotated_image(
        self,
        annotated_bytes: bytes,
        output_dir: str | Path,
        filename: str,
    ) -> Path:
        """Save the annotated image under the requested ensemble output directory."""
        output_path = Path(output_dir) / "predict" / Path(filename).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(annotated_bytes)
        return output_path

    def to_data_url(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
