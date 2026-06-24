from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "Dataset"
DEFAULT_MODEL = PROJECT_DIR / "models" / "inceptionv3_severity_best.keras"
OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"

IMG_SIZE = (299, 299)
CLASS_NAMES = ["Healthy", "Moderate", "Severe"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEVERE_KEYS = [
    "late_blight",
    "yellow_leaf_curl_virus",
    "mosaic_virus",
    "virus",
    "nematode",
    "phytopthora",
    "bacterial_spot",
    "wet_rot",
    "brown_rot",
    "soft_rot",
    "dry_rot",
    "pink_rot",
    "blackleg",
    "shoot_and_fruit_borer",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Evaluate the InceptionV3 severity model on a balanced sample from Dataset.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def norm(text: str) -> str:
    return text.strip().lower()


def infer_label(path: Path) -> str:
    parts = [norm(part) for part in path.parts]
    joined = "/".join(parts)

    if "healthy_leaf" in joined or "healty_brinjal" in joined:
        return "Healthy"
    if "/healthy" in joined or "/fruit/healthy" in joined or "healthy_potato" in joined:
        return "Healthy"

    for key in SEVERE_KEYS:
        if key in joined:
            return "Severe"

    if "/fruit/" in joined and any(token in joined for token in ["rot", "hole", "black_hole"]):
        return "Severe"

    return "Moderate"


def load_image(path: Path) -> np.ndarray:
    data = tf.io.read_file(str(path))
    image = tf.image.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return tf.expand_dims(image, axis=0).numpy()


def sample_images(dataset_dir: Path, per_class: int, seed: int) -> list[tuple[Path, str]]:
    buckets: dict[str, list[Path]] = {label: [] for label in CLASS_NAMES}

    for path in dataset_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = infer_label(path)
        buckets[label].append(path)

    rng = random.Random(seed)
    sampled: list[tuple[Path, str]] = []
    for label in CLASS_NAMES:
        candidates = buckets[label]
        if len(candidates) < per_class:
            raise ValueError(f"Not enough images for {label}: requested {per_class}, found {len(candidates)}")
        rng.shuffle(candidates)
        sampled.extend((path, label) for path in candidates[:per_class])

    rng.shuffle(sampled)
    return sampled


def main() -> None:
    args = build_parser().parse_args()

    model_path = Path(args.model).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "inceptionv3_dataset_sample_predictions.csv"
    cm_path = OUTPUT_DIR / "inceptionv3_dataset_sample_confusion_matrix.png"

    sampled = sample_images(dataset_dir, args.per_class, args.seed)
    model = tf.keras.models.load_model(model_path)

    print("=" * 72)
    print("InceptionV3 Dataset Sample Evaluation")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Per-class sample size: {args.per_class}")
    print(f"Total sampled images: {len(sampled)}")
    print("=" * 72)

    rows: list[dict[str, object]] = []
    confusion = np.zeros((3, 3), dtype=int)
    correct = 0

    for index, (image_path, true_label) in enumerate(sampled, start=1):
        batch = load_image(image_path)
        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        true_idx = CLASS_NAMES.index(true_label)
        confusion[true_idx, pred_idx] += 1
        if pred_label == true_label:
            correct += 1

        rows.append(
            {
                "image_path": str(image_path),
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": round(float(probs[pred_idx]), 6),
                "healthy_prob": round(float(probs[0]), 6),
                "moderate_prob": round(float(probs[1]), 6),
                "severe_prob": round(float(probs[2]), 6),
                "is_correct": pred_label == true_label,
            }
        )

        if index % 25 == 0 or index == len(sampled):
            print(f"Processed {index}/{len(sampled)}")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / len(sampled)
    print()
    print(f"Overall accuracy: {accuracy:.4f} ({correct}/{len(sampled)})")

    print("\nPer-class accuracy:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        total = confusion[class_index].sum()
        cls_acc = confusion[class_index, class_index] / total if total else 0.0
        print(f"  {class_name}: {cls_acc:.4f} ({confusion[class_index, class_index]}/{total})")

    print("\nPrediction distribution:")
    pred_counts = Counter(row["predicted_label"] for row in rows)
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {pred_counts[class_name]}")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("InceptionV3 Severity Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"\nSaved predictions CSV: {csv_path}")
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
