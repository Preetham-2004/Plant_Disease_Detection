#!/usr/bin/env python3

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = {
    0: "eggplant_fruit",
    1: "eggplant_leaf",
    2: "potato_fruit",
    3: "potato_leaf",
    4: "tomato_fruit",
    5: "tomato_leaf",
}

DEFAULT_MODEL = Path(
    "/home/preetham-bhat/plant_project/Plant Disease prediction/runs/detect/train_yolov8s_unified_finetune/weights/best.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-model YOLO inference on a folder of images.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="/home/preetham-bhat/Downloads/test",
        help="Folder containing input images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results_v3",
        help="Directory where annotated outputs will be saved",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="YOLO model checkpoint to use",
    )
    return parser


def process_folder(model: YOLO, folder_path: str, conf: float, save_dir: str) -> None:
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in image_extensions]
    if not images:
        print(f"No images found in: {folder}")
        sys.exit(1)

    output_dir = Path(save_dir) / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("YOLO V3 Single-Model Inference")
    print("=" * 80)
    print(f"Input folder: {folder}")
    print(f"Images: {len(images)}")
    print(f"Confidence: {conf}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    total_detections = 0
    by_class = defaultdict(int)

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print("   Skipped: could not read image")
            continue

        result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
        detections = []

        if len(result.boxes) > 0:
            for box, score, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                score = float(score)
                cls_id = int(cls_id)
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                detections.append((x1, y1, x2, y2, score, cls_name))

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
                label = f"{cls_name} {score:.1%}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                y_top = max(0, y1 - h - 8)
                cv2.rectangle(image, (x1, y_top), (x1 + w + 8, y1), (0, 200, 255), -1)
                cv2.putText(
                    image,
                    label,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                total_detections += 1
                by_class[cls_name] += 1

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), image)

        if detections:
            summary = ", ".join(f"{cls_name} ({score:.1%})" for _, _, _, _, score, cls_name in detections)
            print(f"   {len(detections)} detections: {summary}")
        else:
            print("   No detections")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total detections: {total_detections}")
    if by_class:
        for cls_name, count in sorted(by_class.items()):
            print(f"  {cls_name}: {count}")
    print(f"Annotated images saved to: {output_dir}")
    print("=" * 80)


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    process_folder(model, args.folder, args.conf, args.save_dir)


if __name__ == "__main__":
    main()
