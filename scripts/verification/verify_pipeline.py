from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.services.inference import CLASS_LABELS, PlantDiseasePipeline


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Verify YOLOv8 -> crop -> InceptionV3 pipeline on a single image.",
    )
    parser.add_argument(
        "image",
        help="Path to the image that should be verified.",
    )
    parser.add_argument(
        "--output-dir",
        default="verification_outputs",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--yolo-model",
        default="/home/preetham-bhat/plant_project/Plant Disease prediction/best_yolo_parts.pt",
        help="Path to YOLOv8 model weights.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 Use your newly trained YOLOv8 model
    pipeline = PlantDiseasePipeline(
        yolo_model_path=args.yolo_model
    )

    # Load image
    pil_image = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_image)

    # YOLO detection
    detection_result = pipeline.detector.predict(
        source=image_np,
        conf=pipeline.yolo_confidence,
        verbose=False,
    )[0]

    # Pipeline steps
    selected = pipeline._select_detection(detection_result, pil_image.size)
    selected_region = pipeline._extract_region(pil_image, detection_result, selected)
    annotated = pipeline._annotate_image(pil_image, detection_result, selected)
    probabilities = pipeline._classify(selected_region)

    # Save outputs
    annotated_path = output_dir / f"{image_path.stem}_annotated.png"
    crop_path = output_dir / f"{image_path.stem}_selected_region.png"
    annotated.save(annotated_path)
    selected_region.save(crop_path)

    # Console output
    print("=" * 72)
    print("Pipeline Verification")
    print("=" * 72)
    print(f"Image: {image_path}")
    print(f"YOLO model: {pipeline.yolo_model_path}")
    print(f"YOLO task: {pipeline.yolo_task}")
    print(f"Classifier model: {pipeline.classifier_model_path}")
    print(f"Classifier input shape: {pipeline.classifier_input_shape}")
    print()

    boxes = detection_result.boxes
    detection_count = 0 if boxes is None else len(boxes)
    print(f"YOLO detections: {detection_count}")

    if detection_count == 0:
        print("No plant part was detected.")
    else:
        print("Detected regions:")
        for idx, box in enumerate(boxes):
            cls_idx = int(box.cls.item())
            label = pipeline.part_names.get(cls_idx, f"class-{cls_idx}")
            confidence = float(box.conf.item())
            bbox = [int(round(v)) for v in box.xyxy[0].tolist()]
            marker = " <== selected" if idx == selected.index else ""
            print(
                f"  [{idx}] label={label:<10} conf={confidence:.4f} bbox={bbox}{marker}"
            )

    print()
    print(f"Selected part: {selected.label}")
    print(f"Selection mode: {selected.segmentation_mode}")
    print(f"Selected bbox: {selected.bbox}")
    print()

    print("Classifier probabilities:")
    for label, score in zip(CLASS_LABELS, probabilities):
        print(f"  {label:<10} {float(score):.4f}")

    best_idx = int(np.argmax(probabilities))
    print()
    print(
        f"Predicted class: {CLASS_LABELS[best_idx]} "
        f"({float(probabilities[best_idx]):.4f})"
    )

    print(f"\nAnnotated image saved to: {annotated_path}")
    print(f"Selected region saved to: {crop_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()