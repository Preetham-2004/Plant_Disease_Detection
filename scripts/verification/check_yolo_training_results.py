from argparse import ArgumentParser
from csv import DictReader
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "detect" / "train_yolov8s_unified_finetune"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Summarize a YOLO training run and judge whether the results look healthy.",
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--metric", default="metrics/mAP50-95(B)")
    return parser


def load_results(results_csv: Path) -> list[dict[str, float]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No epochs found in {results_csv}")
    return rows


def score_run(best_map5095: float, best_map50: float, best_recall: float) -> tuple[str, list[str]]:
    notes: list[str] = []

    if best_map5095 >= 0.65 and best_map50 >= 0.85 and best_recall >= 0.80:
        status = "good"
        notes.append("Validation metrics are strong for a 6-class plant detector.")
    elif best_map5095 >= 0.55 and best_map50 >= 0.78 and best_recall >= 0.72:
        status = "promising"
        notes.append("The run looks usable, but there is still room to improve class separation.")
    else:
        status = "needs_work"
        notes.append("Validation quality is still weak enough that more tuning or cleanup is likely needed.")

    return status, notes


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_csv = run_dir / "results.csv"
    rows = load_results(results_csv)

    metric_name = args.metric
    if metric_name not in rows[0]:
        raise KeyError(f"Metric '{metric_name}' not found in {results_csv.name}")

    best_row = max(rows, key=lambda row: row[metric_name])
    last_row = rows[-1]
    first_row = rows[0]

    best_epoch = int(best_row["epoch"])
    completed_epochs = int(last_row["epoch"])
    best_map50 = best_row["metrics/mAP50(B)"]
    best_map5095 = best_row["metrics/mAP50-95(B)"]
    best_precision = best_row["metrics/precision(B)"]
    best_recall = best_row["metrics/recall(B)"]

    status, notes = score_run(best_map5095, best_map50, best_recall)

    print("=" * 72)
    print("YOLO Training Summary")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Epochs completed: {completed_epochs}")
    print(f"Best epoch by {metric_name}: {best_epoch}")
    print()
    print("Best validation metrics:")
    print(f"  Precision:  {best_precision:.4f}")
    print(f"  Recall:     {best_recall:.4f}")
    print(f"  mAP50:      {best_map50:.4f}")
    print(f"  mAP50-95:   {best_map5095:.4f}")
    print()
    print("Trend:")
    print(f"  Epoch 1 mAP50-95:   {first_row['metrics/mAP50-95(B)']:.4f}")
    print(f"  Last epoch mAP50-95:{last_row['metrics/mAP50-95(B)']:.4f}")
    print(f"  Epoch 1 cls loss:   {first_row['train/cls_loss']:.4f}")
    print(f"  Last epoch cls loss:{last_row['train/cls_loss']:.4f}")
    print()
    print(f"Assessment: {status}")
    for note in notes:
        print(f"  - {note}")

    if completed_epochs < best_epoch:
        print("  - Training is still in progress before the best observed epoch.")
    elif completed_epochs == best_epoch:
        print("  - The current epoch is the best so far.")
    else:
        epochs_since_best = completed_epochs - best_epoch
        print(f"  - Best epoch was {epochs_since_best} epoch(s) ago.")

    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    print()
    print("Weights:")
    print(f"  best.pt exists: {best_weights.exists()}")
    print(f"  last.pt exists: {last_weights.exists()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
