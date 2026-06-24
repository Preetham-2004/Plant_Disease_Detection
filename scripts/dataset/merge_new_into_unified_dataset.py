#!/usr/bin/env python3
"""
Merge the converted `new` datasets into `unified_plant_dataset`.

The merge process:
1. Reads the existing unified train/val samples plus all converted samples from `new`
2. Validates YOLO detection labels against the unified 6-class schema
3. Removes duplicate images by SHA256 hash, preferring existing unified samples
4. Rebuilds train/val with a fresh deterministic 80/20 split
5. Writes collision-proof filenames with source prefixes
"""

from __future__ import annotations

import os
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path("/home/preetham-bhat/plant_project/Plant Disease prediction")
UNIFIED_DIR = ROOT / "unified_plant_dataset"
NEW_DIR = ROOT / "new"
TEMP_OUTPUT_DIR = ROOT / "unified_plant_dataset_rebuild_tmp"

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {
    0: "eggplant_fruit",
    1: "eggplant_leaf",
    2: "potato_fruit",
    3: "potato_leaf",
    4: "tomato_fruit",
    5: "tomato_leaf",
}
SEED = 42
TARGET_IMAGE_COUNTS = {
    1: 1487,
    3: 1800,
    5: 1900,
}


@dataclass
class Sample:
    source_tag: str
    image_path: Path
    label_path: Path
    image_hash: str
    label_text: str
    classes: set[int]


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label_file(label_path: Path) -> tuple[str, set[int]]:
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = label_path.read_text(encoding="latin-1")

    classes: set[int] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} has {len(parts)} fields, expected 5")

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} contains non-numeric values") from exc

        if class_id not in CLASS_NAMES:
            raise ValueError(f"{label_path}:{line_number} has invalid class id {class_id}")

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise ValueError(f"{label_path}:{line_number} center coords out of range")

        if not (0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
            raise ValueError(f"{label_path}:{line_number} width/height out of range")

        classes.add(class_id)

    return text, classes


def paired_label_path(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def iter_dataset_samples(images_dir: Path, labels_dir: Path, source_tag: str) -> Iterable[Sample]:
    if not images_dir.exists():
        return

    image_paths = sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES]
    )

    label_stems = {path.stem for path in labels_dir.glob("*.txt")} if labels_dir.exists() else set()

    for image_path in image_paths:
        label_path = paired_label_path(image_path, labels_dir)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image {image_path}")

        label_text, classes = validate_label_file(label_path)
        yield Sample(
            source_tag=source_tag,
            image_path=image_path,
            label_path=label_path,
            image_hash=compute_sha256(image_path),
            label_text=label_text,
            classes=classes,
        )

    image_stems = {path.stem for path in image_paths}
    extra_labels = sorted(label_stems - image_stems)
    if extra_labels:
        raise FileNotFoundError(
            f"{labels_dir} contains labels without matching images, e.g. {extra_labels[:5]}"
        )


def collect_all_samples() -> list[Sample]:
    sources = [
        (
            UNIFIED_DIR / "train" / "images",
            UNIFIED_DIR / "train" / "labels",
            "unified_train",
        ),
        (
            UNIFIED_DIR / "val" / "images",
            UNIFIED_DIR / "val" / "labels",
            "unified_val",
        ),
        (
            NEW_DIR / "eggplantfruit" / "train" / "images",
            NEW_DIR / "eggplantfruit" / "train" / "labels",
            "new_eggplantfruit",
        ),
        (
            NEW_DIR / "EGGPLANT.yolov8-obb" / "train" / "images",
            NEW_DIR / "EGGPLANT.yolov8-obb" / "train" / "labels",
            "new_eggplant_leaf",
        ),
        (
            NEW_DIR / "Leaf-Tomato-Current.yolov8-obb" / "train" / "images",
            NEW_DIR / "Leaf-Tomato-Current.yolov8-obb" / "train" / "labels",
            "new_tomato_leaf",
        ),
        (
            NEW_DIR / "Tomato Fruit Detector.yolov8-obb" / "train" / "images",
            NEW_DIR / "Tomato Fruit Detector.yolov8-obb" / "train" / "labels",
            "new_tomato_fruit",
        ),
    ]

    samples: list[Sample] = []
    for images_dir, labels_dir, source_tag in sources:
        samples.extend(iter_dataset_samples(images_dir, labels_dir, source_tag))
    return samples


def deduplicate_samples(samples: list[Sample]) -> tuple[list[Sample], list[tuple[Sample, Sample]]]:
    unique_by_hash: dict[str, Sample] = {}
    conflicts: list[tuple[Sample, Sample]] = []

    for sample in samples:
        existing = unique_by_hash.get(sample.image_hash)
        if existing is None:
            unique_by_hash[sample.image_hash] = sample
            continue

        if existing.label_text.strip() != sample.label_text.strip():
            conflicts.append((existing, sample))

    return list(unique_by_hash.values()), conflicts


def rebalance_samples(samples: list[Sample]) -> list[Sample]:
    rng = random.Random(SEED)
    kept: list[Sample] = []
    target_buckets: dict[int, list[Sample]] = {class_id: [] for class_id in TARGET_IMAGE_COUNTS}

    for sample in samples:
        sample_classes = set(sample.classes)
        matched_target_classes = [class_id for class_id in TARGET_IMAGE_COUNTS if class_id in sample_classes]
        if not matched_target_classes:
            kept.append(sample)
            continue
        if len(matched_target_classes) > 1:
            raise RuntimeError(
                f"Sample {sample.image_path} belongs to multiple target classes, cannot rebalance safely"
            )
        target_buckets[matched_target_classes[0]].append(sample)

    for class_id, bucket in target_buckets.items():
        rng.shuffle(bucket)
        target_count = min(TARGET_IMAGE_COUNTS[class_id], len(bucket))
        kept.extend(bucket[:target_count])

    return kept


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    val_target = round(len(shuffled) * 0.2)
    train_samples = shuffled[:]
    val_samples: list[Sample] = []

    class_to_indices: dict[int, list[int]] = {class_id: [] for class_id in CLASS_NAMES}
    for index, sample in enumerate(train_samples):
        for class_id in sample.classes:
            class_to_indices[class_id].append(index)

    move_indices: set[int] = set()
    for class_id, indices in class_to_indices.items():
        non_empty_indices = indices[:]
        rng.shuffle(non_empty_indices)
        needed = max(1, round(len(non_empty_indices) * 0.2)) if non_empty_indices else 0
        for idx in non_empty_indices:
            if len(move_indices) >= val_target and len([i for i in move_indices if class_id in train_samples[i].classes]) >= needed:
                break
            if len([i for i in move_indices if class_id in train_samples[i].classes]) >= needed:
                break
            move_indices.add(idx)

    remaining_indices = [idx for idx in range(len(train_samples)) if idx not in move_indices]
    rng.shuffle(remaining_indices)
    for idx in remaining_indices:
        if len(move_indices) >= val_target:
            break
        move_indices.add(idx)

    move_indices = set(sorted(move_indices))
    final_train: list[Sample] = []
    final_val: list[Sample] = []
    for idx, sample in enumerate(train_samples):
        if idx in move_indices:
            final_val.append(sample)
        else:
            final_train.append(sample)

    for class_id in CLASS_NAMES:
        if not any(class_id in sample.classes for sample in final_train):
            raise RuntimeError(f"Class {class_id} missing from train split")
        if not any(class_id in sample.classes for sample in final_val):
            raise RuntimeError(f"Class {class_id} missing from val split")

    return final_train, final_val


def write_split(samples: list[Sample], split_name: str) -> None:
    images_dir = TEMP_OUTPUT_DIR / split_name / "images"
    labels_dir = TEMP_OUTPUT_DIR / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        output_stem = f"{index:06d}_{sample.source_tag}_{sample.image_path.stem}"
        output_image = images_dir / f"{output_stem}{sample.image_path.suffix.lower()}"
        output_label = labels_dir / f"{output_stem}.txt"
        os.link(sample.image_path, output_image)
        output_label.write_text(sample.label_text.strip() + ("\n" if sample.label_text.strip() else ""), encoding="utf-8")


def write_data_yaml() -> None:
    config = {
        "path": str(UNIFIED_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": 6,
        "names": CLASS_NAMES,
    }
    (TEMP_OUTPUT_DIR / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def class_image_counts(samples: list[Sample]) -> dict[int, int]:
    counts = {class_id: 0 for class_id in CLASS_NAMES}
    for sample in samples:
        for class_id in sample.classes:
            counts[class_id] += 1
    return counts


def replace_unified_dataset() -> None:
    for entry in ["train", "val", "data.yaml"]:
        path = UNIFIED_DIR / entry
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    shutil.move(str(TEMP_OUTPUT_DIR / "train"), str(UNIFIED_DIR / "train"))
    shutil.move(str(TEMP_OUTPUT_DIR / "val"), str(UNIFIED_DIR / "val"))
    shutil.move(str(TEMP_OUTPUT_DIR / "data.yaml"), str(UNIFIED_DIR / "data.yaml"))
    shutil.rmtree(TEMP_OUTPUT_DIR)


def main() -> None:
    if TEMP_OUTPUT_DIR.exists():
        shutil.rmtree(TEMP_OUTPUT_DIR)

    print("Collecting samples...")
    all_samples = collect_all_samples()
    print(f"Collected {len(all_samples)} image-label pairs")

    print("Removing duplicate images...")
    unique_samples, conflicts = deduplicate_samples(all_samples)
    print(f"Kept {len(unique_samples)} unique images")
    print(f"Removed {len(all_samples) - len(unique_samples)} duplicates by content hash")

    if conflicts:
        print(f"Warning: {len(conflicts)} duplicate-image label conflicts detected; kept first occurrence")
        for existing, duplicate in conflicts[:10]:
            print(f"  - {existing.image_path} conflicts with {duplicate.image_path}")

    print("Rebalancing target leaf classes...")
    balanced_samples = rebalance_samples(unique_samples)
    print(f"Kept {len(balanced_samples)} samples after balancing")

    print("Creating train/val split...")
    train_samples, val_samples = split_samples(balanced_samples)
    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")

    print("Writing rebuilt dataset...")
    write_split(train_samples, "train")
    write_split(val_samples, "val")
    write_data_yaml()

    print("Replacing unified dataset contents...")
    replace_unified_dataset()

    print("Done.")
    print("Class image counts (combined):")
    for class_id, count in class_image_counts(balanced_samples).items():
        print(f"  {class_id} {CLASS_NAMES[class_id]}: {count}")


if __name__ == "__main__":
    main()
