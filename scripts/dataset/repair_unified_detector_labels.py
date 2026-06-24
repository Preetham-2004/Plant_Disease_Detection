#!/usr/bin/env python3
"""
Conservative label repair tool for the unified detector dataset.

This only applies high-confidence filename-based remaps and writes backups
before editing any label file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = {
    0: "eggplant_fruit",
    1: "eggplant_leaf",
    2: "potato_fruit",
    3: "potato_leaf",
    4: "tomato_fruit",
    5: "tomato_leaf",
}


def replacement_for_name(image_stem: str, class_id: int) -> int | None:
    text = image_stem.lower()

    # "Healthy Potato ..." images are clearly leaf images, but many were merged
    # as potato_fruit.
    if "healthy potato" in text and class_id == 2:
        return 3

    # These filename patterns come from potato leaf disease sets. They should
    # never be tomato_leaf, and any fruit label on them is also suspicious.
    potato_leaf_hints = ("rs_erly", "erly-b", "gls", "lb_", "ghlb")
    if any(hint in text for hint in potato_leaf_hints) and class_id in {2, 5}:
        return 3

    return None


def repair_label_file(label_path: Path, backup_root: Path) -> tuple[int, list[dict]]:
    original = label_path.read_text(encoding="utf-8")
    lines = original.splitlines()

    updated_lines = []
    changes = []
    file_changed = False

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if not parts:
            updated_lines.append(line)
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            updated_lines.append(line)
            continue

        new_class_id = replacement_for_name(label_path.stem, class_id)
        if new_class_id is None or new_class_id == class_id:
            updated_lines.append(line)
            continue

        parts[0] = str(new_class_id)
        updated_lines.append(" ".join(parts))
        file_changed = True
        changes.append({
            "line": line_number,
            "from": class_id,
            "to": new_class_id,
            "from_name": CLASS_NAMES.get(class_id, f"unknown_{class_id}"),
            "to_name": CLASS_NAMES.get(new_class_id, f"unknown_{new_class_id}"),
        })

    if not file_changed:
        return 0, []

    backup_path = backup_root / label_path.relative_to(label_path.parents[2])
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, backup_path)
    label_path.write_text("\n".join(updated_lines) + ("\n" if original.endswith("\n") or updated_lines else ""), encoding="utf-8")

    return len(changes), changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair obvious unified detector label mismatches.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/preetham-bhat/plant_project/Plant Disease prediction/unified_plant_dataset"),
        help="Path to the unified detector dataset.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("/home/preetham-bhat/plant_project/Plant Disease prediction/unified_plant_dataset_backups/conservative_repairs"),
        help="Where to store backups before rewriting labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without modifying files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/home/preetham-bhat/plant_project/Plant Disease prediction/analysis_outputs/unified_label_repair_report.json"),
        help="Where to write the JSON repair summary.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    report = {
        "dataset": str(dataset),
        "dry_run": args.dry_run,
        "files_changed": 0,
        "line_changes": 0,
        "changes_by_rule": Counter(),
        "changes_by_split": Counter(),
        "examples": [],
    }

    for split in ("train", "val"):
        labels_dir = dataset / split / "labels"
        for label_path in sorted(labels_dir.glob("*.txt")):
            original = label_path.read_text(encoding="utf-8")
            if not original.strip():
                continue

            pending_changes = []
            for line_number, line in enumerate(original.splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                new_class_id = replacement_for_name(label_path.stem, class_id)
                if new_class_id is None or new_class_id == class_id:
                    continue
                rule_name = f"{CLASS_NAMES[class_id]}->{CLASS_NAMES[new_class_id]}"
                pending_changes.append((line_number, class_id, new_class_id, rule_name))

            if not pending_changes:
                continue

            report["files_changed"] += 1
            report["line_changes"] += len(pending_changes)
            report["changes_by_split"][split] += len(pending_changes)

            for _, _, _, rule_name in pending_changes:
                report["changes_by_rule"][rule_name] += 1

            if len(report["examples"]) < 25:
                report["examples"].append({
                    "split": split,
                    "file": label_path.name,
                    "changes": [
                        {
                            "line": line_number,
                            "from": CLASS_NAMES[class_id],
                            "to": CLASS_NAMES[new_class_id],
                        }
                        for line_number, class_id, new_class_id, _ in pending_changes
                    ],
                })

            if not args.dry_run:
                repair_label_file(label_path, args.backup_dir.resolve())

    report["changes_by_rule"] = dict(report["changes_by_rule"])
    report["changes_by_split"] = dict(report["changes_by_split"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 80)
    print("UNIFIED LABEL REPAIR")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Dry run: {args.dry_run}")
    print(f"Files changed: {report['files_changed']}")
    print(f"Line changes: {report['line_changes']}")
    print(f"Report: {args.report}")
    if not args.dry_run:
        print(f"Backups: {args.backup_dir.resolve()}")
    print("Changes by rule:")
    for rule_name, count in sorted(report["changes_by_rule"].items()):
        print(f"  {rule_name:30s} {count}")


if __name__ == "__main__":
    main()
