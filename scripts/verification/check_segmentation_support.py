from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Inspect local YOLO checkpoints and verify whether they return segmentation masks.",
    )
    parser.add_argument(
        "--models-root",
        default=".",
        help="Directory to scan recursively for .pt files.",
    )
    parser.add_argument(
        "--sample-image",
        default="YoloDataset/parts_merged/images/valid/Tomato_flower__IMG_06094455_jpg.rf.2300dee08d8debef1c347d851c1ad088.jpg",
        help="Image used for a real inference mask check.",
    )
    parser.add_argument(
        "--report-json",
        default="analysis_outputs/segmentation_support_report.json",
        help="Where to save the checkpoint capability report.",
    )
    return parser


def resolve_project_path(project_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (project_dir / path).resolve()


def main() -> None:
    args = build_parser().parse_args()
    project_dir = Path(__file__).resolve().parent
    models_root = resolve_project_path(project_dir, args.models_root)
    sample_image_path = resolve_project_path(project_dir, args.sample_image)
    report_json_path = resolve_project_path(project_dir, args.report_json)

    sample_array = np.asarray(Image.open(sample_image_path).convert("RGB"))
    checkpoints = []

    for model_path in sorted(models_root.rglob("*.pt")):
        try:
            model = YOLO(str(model_path))
            result = model.predict(source=sample_array, conf=0.25, verbose=False)[0]
            masks = getattr(result, "masks", None)
            mask_count = 0 if masks is None else len(getattr(masks, "xy", []) or [])
            checkpoints.append(
                {
                    "path": str(model_path),
                    "task": model.task,
                    "names": {str(k): str(v) for k, v in model.names.items()},
                    "produced_masks": mask_count > 0,
                    "mask_count": mask_count,
                    "num_boxes": 0 if result.boxes is None else len(result.boxes),
                }
            )
        except Exception as exc:
            checkpoints.append(
                {
                    "path": str(model_path),
                    "error": str(exc),
                }
            )

    segment_models = [item for item in checkpoints if item.get("task") == "segment"]
    models_with_masks = [item for item in checkpoints if item.get("produced_masks")]

    report = {
        "models_root": str(models_root),
        "sample_image": str(sample_image_path),
        "segment_model_count": len(segment_models),
        "models_with_masks_on_sample": len(models_with_masks),
        "checkpoints": checkpoints,
    }

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Segmentation Support Check")
    print("=" * 72)
    print(f"Models scanned: {len(checkpoints)}")
    print(f"Sample image: {sample_image_path}")
    print(f"Segment checkpoints found: {len(segment_models)}")
    print(f"Models producing masks on sample: {len(models_with_masks)}")
    print(f"Report saved to: {report_json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
