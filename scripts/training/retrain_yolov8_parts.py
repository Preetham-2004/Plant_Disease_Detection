from argparse import ArgumentParser
from pathlib import Path
import shutil

import torch
import yaml
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
UNIFIED_DATASET_DIR = PROJECT_DIR / "unified_plant_dataset"
DATA_YAML = UNIFIED_DATASET_DIR / "data.yaml"
DEFAULT_MODEL = PROJECT_DIR / "runs" / "detect" / "train_yolov8s_cleaned" / "weights" / "best.pt"
RUNS_DIR = PROJECT_DIR / "runs" / "detect"
DEPLOY_WEIGHTS = PROJECT_DIR / "best_yolo_parts.pt"

EXPECTED_CLASSES = {
    0: "eggplant_fruit",
    1: "eggplant_leaf",
    2: "potato_fruit",
    3: "potato_leaf",
    4: "tomato_fruit",
    5: "tomato_leaf",
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Fine-tune an existing YOLO model on the merged unified plant dataset.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="train_yolov8s_unified_finetune")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true")
    return parser


def verify_dataset(data_yaml_path: Path) -> tuple[Path, int, int]:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml_path}")

    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"]).expanduser().resolve()

    required_dirs = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "val" / "images",
        dataset_root / "val" / "labels",
    ]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Dataset incomplete. Missing:\n" + "\n".join(missing))

    names = data.get("names", {})
    for class_id, expected_name in EXPECTED_CLASSES.items():
        actual_name = names.get(class_id, names.get(str(class_id)))
        if actual_name != expected_name:
            raise ValueError(
                f"Class mapping mismatch for class {class_id}: expected {expected_name}, got {actual_name}"
            )

    train_count = len(list((dataset_root / "train" / "images").glob("*")))
    val_count = len(list((dataset_root / "val" / "images").glob("*")))
    return dataset_root, train_count, val_count


def choose_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        return "cpu"
    return requested_device


def main() -> None:
    args = build_parser().parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Base model not found: {model_path}")

    data_yaml_path = Path(args.data).expanduser().resolve()
    dataset_root, train_images, val_images = verify_dataset(data_yaml_path)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    print("=" * 72)
    print("YOLOv8 Unified Fine-Tuning")
    print("=" * 72)
    print(f"Base model: {model_path}")
    print(f"Dataset yaml: {data_yaml_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Workers: {args.workers}")
    print(f"Patience: {args.patience}")
    print(f"Device: {device}")
    print("=" * 72)

    model = YOLO(str(model_path))

    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        project=str(RUNS_DIR),
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        amp=True,
        plots=True,
        save=True,
        val=True,
        seed=42,
        deterministic=True,
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.7,
        close_mosaic=10,
        mosaic=0.35,
        mixup=0.03,
        copy_paste=0.0,
        degrees=8.0,
        translate=0.08,
        scale=0.35,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.25,
        auto_augment=None,
    )

    print("\nFinal Metrics:")
    print(results.results_dict)

    best_weights = RUNS_DIR / args.name / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"Best weights: {best_weights}")

    if best_weights.exists():
        shutil.copy2(best_weights, DEPLOY_WEIGHTS)
        print(f"Copied best weights to: {DEPLOY_WEIGHTS}")


if __name__ == "__main__":
    main()
