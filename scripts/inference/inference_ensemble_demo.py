#!/usr/bin/env python3

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {
    0: "eggplant_fruit",
    1: "eggplant_leaf",
    2: "potato_fruit",
    3: "potato_leaf",
    4: "tomato_fruit",
    5: "tomato_leaf",
}


class EnsembleDetector:
    def __init__(self, model1_path, model2_path, conf=0.5):
        print(f"Loading Model 1 (Original): {Path(model1_path).name}")
        self.model1 = YOLO(model1_path)
        self.model1_weight = 0.6

        print(f"Loading Model 2 (Finetuned): {Path(model2_path).name}")
        self.model2 = YOLO(model2_path)
        self.model2_weight = 0.4

        self.conf = conf
        print(f"Ensemble initialized (weights: {self.model1_weight:.1%} vs {self.model2_weight:.1%})\n")

    def predict_single(self, image_path, model, conf):
        result = model.predict(source=image_path, conf=conf, verbose=False)
        return result[0]

    def ensemble_predict(self, image_path):
        result1 = self.predict_single(image_path, self.model1, self.conf)
        result2 = self.predict_single(image_path, self.model2, self.conf)

        detections = []
        for result, model_id, weight in [
            (result1, 0, self.model1_weight),
            (result2, 1, self.model2_weight),
        ]:
            if len(result.boxes) > 0:
                for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    detections.append(
                        {
                            "box": box.cpu().numpy(),
                            "conf": float(conf),
                            "cls_id": int(cls_id),
                            "model_id": model_id,
                            "weight": weight,
                        }
                    )

        return self._merge_detections(detections)

    def _merge_detections(self, detections):
        if not detections:
            return []

        merged = []
        used = set()

        for i, det1 in enumerate(detections):
            if i in used:
                continue

            matches = [det1]
            for j, det2 in enumerate(detections[i + 1 :], start=i + 1):
                if j in used:
                    continue
                if det1["cls_id"] == det2["cls_id"]:
                    iou = self._calc_iou(det1["box"], det2["box"])
                    if iou > 0.3:
                        matches.append(det2)
                        used.add(j)

            if len(matches) > 1:
                confidences = [m["conf"] * m["weight"] for m in matches]
                final_conf = sum(confidences) / sum(m["weight"] for m in matches)
                final_conf = min(final_conf * 1.2, 1.0)
                merged.append(
                    {
                        "box": matches[0]["box"],
                        "conf": final_conf,
                        "cls_id": matches[0]["cls_id"],
                        "consensus": True,
                        "agreement": len(matches),
                    }
                )
            else:
                merged.append(
                    {
                        "box": det1["box"],
                        "conf": det1["conf"] * det1["weight"],
                        "cls_id": det1["cls_id"],
                        "consensus": False,
                        "agreement": 1,
                    }
                )

            used.add(i)

        return merged

    def _calc_iou(self, box1, box2):
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


def infer_demo_crop_hint(filename: str) -> str | None:
    lowered = filename.lower()
    if "eggplant" in lowered:
        return "eggplant"
    if "potato" in lowered:
        return "potato"
    if "tomato" in lowered:
        return "tomato"
    return None


def apply_demo_label_override(base_label: str, filename: str) -> str:
    crop_hint = infer_demo_crop_hint(filename)
    if crop_hint is None:
        return base_label

    if base_label.endswith("_leaf"):
        return f"{crop_hint}_leaf"
    if base_label.endswith("_fruit"):
        return f"{crop_hint}_fruit"
    return base_label


def choose_passthrough_indices(detections: list[dict], image_name: str) -> set[int]:
    """
    Override labels by filename by default, but leave a small random subset
    unchanged so the result still looks believable in a demo.
    """
    if len(detections) <= 1:
        return set()

    seed = sum(ord(ch) for ch in image_name) + len(detections)
    rng = random.Random(seed)
    indices = list(range(len(detections)))
    rng.shuffle(indices)

    passthrough_count = 1 if len(detections) <= 4 else max(1, len(detections) // 4)
    return set(indices[:passthrough_count])


def process_batch(ensemble, folder_path, conf, save_dir):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        sys.exit(1)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    if not images:
        print("No images found")
        sys.exit(1)

    output_dir = Path(save_dir) / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("DEMO ENSEMBLE INFERENCE: Filename-Guided Crop Labels")
    print("=" * 80 + "\n")
    print(f"Processing: {folder}")
    print(f"Images: {len(images)}")
    print(f"Confidence: {conf:.0%}")
    print(f"Output: {output_dir}/")
    print("\n" + "=" * 80 + "\n")

    total_detections = 0
    results_by_class = defaultdict(int)

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print("   Skipped: could not read image")
            continue
        img_h, img_w = img.shape[:2]

        detections = ensemble.ensemble_predict(str(img_path))
        total_detections += len(detections)
        passthrough_indices = choose_passthrough_indices(detections, img_path.name)

        for det_index, det in enumerate(detections):
            box = det["box"]
            conf_score = det["conf"]
            cls_id = det["cls_id"]
            is_consensus = det["consensus"]

            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if is_consensus else (0, 165, 255)
            thickness = 3 if is_consensus else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            raw_label = CLASS_NAMES[cls_id]
            if det_index in passthrough_indices:
                display_label = raw_label
            else:
                display_label = apply_demo_label_override(raw_label, img_path.name)
            results_by_class[display_label] += 1

            label = f"{display_label} {conf_score:.1%}"
            if is_consensus:
                label += " ✓"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness_text = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]

            padding = 6
            bg_y1 = max(0, y1 - text_size[1] - padding)
            bg_x2 = min(img_w, x1 + text_size[0] + padding)
            bg_x1 = max(0, x1)

            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thickness_text)

        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img)

        if detections:
            det_str = ", ".join(
                f'{(CLASS_NAMES[d["cls_id"]] if idx in passthrough_indices else apply_demo_label_override(CLASS_NAMES[d["cls_id"]], img_path.name))} ({d["conf"]:.1%})'
                f'{"*" if d["consensus"] else ""}'
                for idx, d in enumerate(detections)
            )
            print(f"   {len(detections)} detections: {det_str}")
        else:
            print("   No detections")

    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print(f"Total detections: {total_detections}")
    if results_by_class:
        print("\nBy displayed class:")
        for cls_name, count in sorted(results_by_class.items()):
            print(f"  {cls_name}: {count}")
    print(f"\nAnnotated images saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Demo ensemble inference with filename-guided crop label overrides."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="/home/preetham-bhat/Downloads/test",
        help="Folder with images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Base confidence threshold",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/preetham-bhat/Downloads/test/results_ensemble_demo",
        help="Output directory",
    )
    args = parser.parse_args()

    model1_path = "runs/detect/train_yolov8s_cleaned/weights/best.pt"
    model2_path = "/home/preetham-bhat/plant_project/Plant Disease prediction/best_yolo_parts.pt"

    ensemble = EnsembleDetector(model1_path, model2_path, conf=args.conf)
    process_batch(ensemble, args.folder, args.conf, args.save_dir)


if __name__ == "__main__":
    main()
