#!/usr/bin/env python3
"""
Audit unified detector labels for likely crop/part mismatches.

This script does not try to "fix" labels automatically. Instead, it uses
filename hints from the merged dataset to flag suspicious image/label pairs
that are worth manual review before retraining.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


CLASS_NAMES = {
    0: "eggplant_fruit",
    1: "eggplant_leaf",
    2: "potato_fruit",
    3: "potato_leaf",
    4: "tomato_fruit",
    5: "tomato_leaf",
}


def load_class_names(dataset_path: Path) -> dict[int, str]:
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        return CLASS_NAMES

    with data_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    names = config.get("names", {})
    loaded = {}
    for key, value in names.items():
        loaded[int(key)] = str(value)
    return loaded or CLASS_NAMES


def infer_expected_classes_from_name(image_name: str, class_names: dict[int, str]) -> set[int]:
    text = image_name.lower()
    expected_crops = set()
    expected_parts = set()

    crop_hints = {
        "eggplant": ["eggplant", "brinjal"],
        "potato": ["potato", "phytopthora", "healthy potato", "rs_early", "erly-b", "gls", "lb_"],
        "tomato": [
            "tomato",
            "ylcv",
            "yellowleafcurl",
            "leaf mold",
            "target spot",
            "mosaicvirus",
            "bacterial_spot",
            "late_blight",
            "early_blight",
        ],
    }
    part_hints = {
        "fruit": ["fruit", "tomato", "brak", "good", "big", "medium", "small"],
        "leaf": ["leaf", "healthy", "disease", "blight", "mildew", "mold", "spot", "virus"],
    }

    for crop_name, hints in crop_hints.items():
        if any(hint in text for hint in hints):
            expected_crops.add(crop_name)

    for part_name, hints in part_hints.items():
        if any(hint in text for hint in hints):
            expected_parts.add(part_name)

    if not expected_crops and not expected_parts:
        return set()

    expected_classes = set()
    for class_id, class_name in class_names.items():
        crop_name, part_name = class_name.split("_", 1)
        crop_matches = not expected_crops or crop_name in expected_crops
        part_matches = not expected_parts or part_name in expected_parts
        if crop_matches and part_matches:
            expected_classes.add(class_id)
    return expected_classes


def parse_label_classes(label_path: Path) -> list[int]:
    classes = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                classes.append(int(parts[0]))
            except ValueError:
                continue
    return classes


def audit_split(dataset_path: Path, split: str, class_names: dict[int, str]) -> dict:
    images_dir = dataset_path / split / "images"
    labels_dir = dataset_path / split / "labels"

    summary = {
        "images_checked": 0,
        "missing_labels": 0,
        "invalid_class_ids": 0,
        "class_instance_counts": Counter(),
        "hint_counts": Counter(),
        "suspicious_files": [],
    }

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    for image_path in image_paths:
        summary["images_checked"] += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        expected_classes = infer_expected_classes_from_name(image_path.name, class_names)
        if expected_classes:
            for class_id in sorted(expected_classes):
                summary["hint_counts"][class_names[class_id]] += 1

        if not label_path.exists():
            summary["missing_labels"] += 1
            continue

        label_classes = parse_label_classes(label_path)
        unique_classes = sorted(set(label_classes))

        for class_id in label_classes:
            if class_id in class_names:
                summary["class_instance_counts"][class_names[class_id]] += 1
            else:
                summary["invalid_class_ids"] += 1

        if expected_classes and unique_classes:
            if not set(unique_classes).intersection(expected_classes):
                summary["suspicious_files"].append({
                    "split": split,
                    "image": image_path.name,
                    "label": label_path.name,
                    "expected_from_name": [class_names[c] for c in sorted(expected_classes)],
                    "label_classes": [class_names.get(c, f"unknown_{c}") for c in unique_classes],
                })

    summary["class_instance_counts"] = dict(summary["class_instance_counts"])
    summary["hint_counts"] = dict(summary["hint_counts"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit unified detector labels using filename hints.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/preetham-bhat/plant_project/Plant Disease prediction/unified_plant_dataset"),
        help="Path to the unified detector dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/preetham-bhat/plant_project/Plant Disease prediction/analysis_outputs/unified_label_audit.json"),
        help="Where to write the JSON audit report.",
    )
    parser.add_argument(
        "--max-suspicious",
        type=int,
        default=200,
        help="Maximum suspicious examples to keep in the saved report.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    class_names = load_class_names(dataset_path)

    report = {
        "dataset": str(dataset_path),
        "class_names": class_names,
        "splits": {},
    }

    total_suspicious = 0
    for split in ("train", "val"):
        split_report = audit_split(dataset_path, split, class_names)
        total_suspicious += len(split_report["suspicious_files"])
        split_report["suspicious_files"] = split_report["suspicious_files"][:args.max_suspicious]
        report["splits"][split] = split_report

    report["total_suspicious_files"] = total_suspicious

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 80)
    print("UNIFIED DETECTOR LABEL AUDIT")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Saved report: {args.output}")
    print()

    for split, split_report in report["splits"].items():
        print(f"{split.upper()}:")
        print(f"  Images checked: {split_report['images_checked']}")
        print(f"  Missing labels: {split_report['missing_labels']}")
        print(f"  Invalid class IDs: {split_report['invalid_class_ids']}")
        print(f"  Suspicious files: {len(split_report['suspicious_files'])}")
        if split_report["class_instance_counts"]:
            print("  Label instances by class:")
            for class_name, count in sorted(split_report["class_instance_counts"].items()):
                print(f"    {class_name:16s} {count}")
        print()

    print(f"Total suspicious files found: {total_suspicious}")


if __name__ == "__main__":
    main()
