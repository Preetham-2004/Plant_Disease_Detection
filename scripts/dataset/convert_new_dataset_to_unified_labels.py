#!/usr/bin/env python3
"""
Convert new OBB datasets into the same label schema used by unified_plant_dataset.

This script:
1. Backs up original OBB labels into train/labels_obb_backup
2. Converts 8-point OBB polygons to axis-aligned YOLO boxes
3. Remaps all source classes to the unified 6-class IDs
4. Updates each dataset's data.yaml to the unified class definition
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


ROOT = Path("/home/preetham-bhat/plant_project/Plant Disease prediction/new")

UNIFIED_NAMES = {
    0: "eggplant_fruit",
    1: "eggplant_leaf",
    2: "potato_fruit",
    3: "potato_leaf",
    4: "tomato_fruit",
    5: "tomato_leaf",
}

DATASET_TARGET_CLASS = {
    "eggplantfruit": 0,
    "EGGPLANT.yolov8-obb": 1,
    "Leaf-Tomato-Current.yolov8-obb": 5,
    "Tomato Fruit Detector.yolov8-obb": 4,
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_center = clamp((min_x + max_x) / 2.0)
    y_center = clamp((min_y + max_y) / 2.0)
    width = clamp(max_x - min_x)
    height = clamp(max_y - min_y)
    return x_center, y_center, width, height


def convert_label_file(source_file: Path, target_file: Path, unified_class_id: int) -> int:
    converted_lines: list[str] = []

    with source_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) == 9:
                coords = [float(value) for value in parts[1:]]
                bbox = polygon_to_bbox(coords)
            elif len(parts) == 5:
                bbox = tuple(float(value) for value in parts[1:])
            else:
                raise ValueError(
                    f"Unsupported label format in {source_file} line {line_number}: {line}"
                )

            bbox_text = " ".join(format_float(value) for value in bbox)
            converted_lines.append(f"{unified_class_id} {bbox_text}")

    output = "\n".join(converted_lines)
    if output:
        output += "\n"
    target_file.write_text(output, encoding="utf-8")
    return len(converted_lines)


def update_data_yaml(dataset_dir: Path) -> None:
    data_yaml_path = dataset_dir / "data.yaml"
    new_config = {
        "path": str(dataset_dir),
        "train": "train/images",
        "nc": 6,
        "names": UNIFIED_NAMES,
    }
    data_yaml_path.write_text(
        yaml.safe_dump(new_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def ensure_backup(labels_dir: Path) -> Path:
    backup_dir = labels_dir.parent / "labels_obb_backup"
    if not backup_dir.exists():
        shutil.copytree(labels_dir, backup_dir)
    return backup_dir


def convert_dataset(dataset_name: str, unified_class_id: int) -> dict[str, int]:
    dataset_dir = ROOT / dataset_name
    labels_dir = dataset_dir / "train" / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    backup_dir = ensure_backup(labels_dir)
    converted_files = 0
    converted_objects = 0

    for source_file in sorted(backup_dir.glob("*.txt")):
        target_file = labels_dir / source_file.name
        converted_objects += convert_label_file(source_file, target_file, unified_class_id)
        converted_files += 1

    update_data_yaml(dataset_dir)

    return {
        "files": converted_files,
        "objects": converted_objects,
        "class_id": unified_class_id,
    }


def main() -> None:
    print("Converting new datasets to unified YOLO bbox labels...")
    print()

    for dataset_name, unified_class_id in DATASET_TARGET_CLASS.items():
        stats = convert_dataset(dataset_name, unified_class_id)
        print(
            f"{dataset_name}: {stats['files']} files, {stats['objects']} objects -> "
            f"class {stats['class_id']} ({UNIFIED_NAMES[stats['class_id']]})"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
