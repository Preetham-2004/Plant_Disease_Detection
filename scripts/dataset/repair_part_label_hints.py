from __future__ import annotations

import json
import shutil
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RULES = ["flower:2:3"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Audit and optionally repair YOLO part labels using filename hints. "
            "Default safe rule rewrites flower-hint images from class 2 to class 3."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default="YoloDataset/parts_merged",
        help="YOLO dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to inspect.",
    )
    parser.add_argument(
        "--rule",
        action="append",
        default=None,
        help="Rewrite rule formatted as hint:from_id:to_id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the requested rewrites. Without this flag the script is dry-run only.",
    )
    parser.add_argument(
        "--backup-dir",
        default="YoloDataset/label_backups/hint_repairs",
        help="Directory where original labels are backed up before rewrite.",
    )
    parser.add_argument(
        "--report-json",
        default="analysis_outputs/part_label_hint_repair_report.json",
        help="Where to save the audit/repair report.",
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


def infer_hint(path: Path) -> str | None:
    lower_name = path.name.lower()
    for token in ("flower", "fruit", "leaf", "stem"):
        if token in lower_name:
            return token
    return None


def parse_rule(rule_text: str) -> tuple[str, int, int]:
    try:
        hint, from_id, to_id = rule_text.split(":")
        return hint.strip().lower(), int(from_id), int(to_id)
    except ValueError as exc:
        raise ValueError(
            f"Invalid rule '{rule_text}'. Expected format hint:from_id:to_id"
        ) from exc


def label_path_for_image(image_path: Path, split: str) -> Path:
    return Path(str(image_path).replace(f"/images/{split}/", f"/labels/{split}/")).with_suffix(".txt")


def rewrite_label_file(label_path: Path, from_id: int, to_id: int) -> tuple[list[str], int]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    replacements = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            rewritten.append(line)
            continue

        parts = stripped.split()
        cls_id = int(float(parts[0]))
        if cls_id == from_id:
            parts[0] = str(to_id)
            replacements += 1
        rewritten.append(" ".join(parts))

    return rewritten, replacements


def main() -> None:
    args = build_parser().parse_args()
    project_dir = Path(__file__).resolve().parent

    dataset_dir = resolve_project_path(project_dir, args.dataset_dir)
    backup_dir = resolve_project_path(project_dir, args.backup_dir)
    report_json_path = resolve_project_path(project_dir, args.report_json)
    rules = [parse_rule(rule_text) for rule_text in (args.rule or DEFAULT_RULES)]

    summary: dict[str, dict[str, object]] = {}
    samples: list[dict[str, object]] = []
    touched_files: set[Path] = set()
    total_replacements = 0

    for split in args.splits:
        images_dir = dataset_dir / "images" / split
        split_summary = {
            "images_with_hints": 0,
            "label_files_matched": 0,
            "replacements_by_rule": Counter(),
            "labels_seen_by_hint": defaultdict(Counter),
        }

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            hint = infer_hint(image_path)
            if hint is None:
                continue

            split_summary["images_with_hints"] += 1
            label_path = label_path_for_image(image_path, split)
            if not label_path.exists():
                continue

            original_lines = label_path.read_text(encoding="utf-8").splitlines()
            class_ids = []
            for line in original_lines:
                line = line.strip()
                if line:
                    class_ids.append(int(float(line.split()[0])))
            if not class_ids:
                continue

            split_summary["label_files_matched"] += 1
            split_summary["labels_seen_by_hint"][hint].update(class_ids)

            for rule_hint, from_id, to_id in rules:
                if hint != rule_hint:
                    continue

                rewritten_lines, replacements = rewrite_label_file(label_path, from_id, to_id)
                if replacements == 0:
                    continue

                total_replacements += replacements
                rule_key = f"{rule_hint}:{from_id}->{to_id}"
                split_summary["replacements_by_rule"][rule_key] += replacements

                if len(samples) < 20:
                    samples.append(
                        {
                            "split": split,
                            "image": str(image_path),
                            "label_file": str(label_path),
                            "rule": rule_key,
                            "replacements": replacements,
                        }
                    )

                if args.apply:
                    backup_path = backup_dir / label_path.relative_to(dataset_dir)
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    if label_path not in touched_files:
                        shutil.copy2(label_path, backup_path)
                        touched_files.add(label_path)
                    label_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

        summary[split] = {
            "images_with_hints": split_summary["images_with_hints"],
            "label_files_matched": split_summary["label_files_matched"],
            "replacements_by_rule": {
                key: int(value) for key, value in split_summary["replacements_by_rule"].items()
            },
            "labels_seen_by_hint": {
                hint: {str(cls_id): int(count) for cls_id, count in counts.items()}
                for hint, counts in split_summary["labels_seen_by_hint"].items()
            },
        }

    report = {
        "dataset_dir": str(dataset_dir),
        "apply_mode": args.apply,
        "rules": [f"{hint}:{from_id}->{to_id}" for hint, from_id, to_id in rules],
        "total_replacements": total_replacements,
        "backed_up_files": len(touched_files),
        "backup_dir": str(backup_dir),
        "splits": summary,
        "sample_matches": samples,
    }

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Part Label Hint Repair")
    print("=" * 72)
    print(f"Dataset: {dataset_dir}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Rules: {report['rules']}")
    print(f"Total replacements: {total_replacements}")
    if args.apply:
        print(f"Backed up files: {len(touched_files)}")
        print(f"Backup dir: {backup_dir}")
    print(f"Report saved to: {report_json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
