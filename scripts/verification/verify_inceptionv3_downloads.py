from argparse import ArgumentParser
from pathlib import Path
import csv

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "inceptionv3_severity_best.keras"
DEFAULT_INPUT_DIR = Path.home() / "Downloads/test"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "analysis_outputs" / "downloads_inceptionv3_predictions.csv"

IMG_SIZE = (299, 299)
CLASS_NAMES = ["Healthy", "Moderate", "Severe"]
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Verify the InceptionV3 severity model on images in Downloads.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser


def norm(text: str) -> str:
    return text.strip().lower()


def infer_label(path: Path) -> str | None:
    parts = [norm(part) for part in path.parts]
    joined = "/".join(parts)

    if "healthy_leaf" in joined or "healty_brinjal" in joined:
        return "Healthy"
    if "/healthy" in joined or "/fruit/healthy" in joined or "healthy_potato" in joined:
        return "Healthy"

    for key in SEVERE_KEYS:
        if key in joined:
            return "Severe"

    # If the path explicitly contains one of the severity names, trust it.
    for label in CLASS_NAMES:
        if norm(label) in joined:
            return label

    return None


def list_images(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    )


def load_image(path: Path) -> np.ndarray:
    data = tf.io.read_file(str(path))
    image = tf.image.decode_image(
        data,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return tf.expand_dims(image, axis=0).numpy()


def main() -> None:
    args = build_parser().parse_args()

    model_path = Path(args.model).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("InceptionV3 Downloads Verification")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(f"Input directory: {input_dir}")
    print(f"Images found: {len(images)}")
    print(f"Output CSV: {output_csv}")
    print("=" * 72)

    model = tf.keras.models.load_model(model_path)

    rows: list[dict[str, object]] = []
    inferred_total = 0
    inferred_correct = 0
    skipped_images: list[tuple[Path, str]] = []

    for index, image_path in enumerate(images, start=1):
        try:
            batch = load_image(image_path)
        except Exception as exc:
            skipped_images.append((image_path, str(exc)))
            print(f"Skipped unreadable image: {image_path.name}")
            continue

        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        inferred_label = infer_label(image_path)
        is_correct = inferred_label == pred_label if inferred_label is not None else None

        if inferred_label is not None:
            inferred_total += 1
            if is_correct:
                inferred_correct += 1

        rows.append(
            {
                "image_path": str(image_path),
                "predicted_label": pred_label,
                "confidence": round(confidence, 6),
                "healthy_prob": round(float(probs[0]), 6),
                "moderate_prob": round(float(probs[1]), 6),
                "severe_prob": round(float(probs[2]), 6),
                "inferred_label": inferred_label or "",
                "is_correct": "" if is_correct is None else str(bool(is_correct)),
            }
        )

        if index % 25 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)}")

    if not rows:
        raise RuntimeError("No valid images could be processed.")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    predicted_counts = {label: 0 for label in CLASS_NAMES}
    for row in rows:
        predicted_counts[str(row["predicted_label"])] += 1

    print()
    print("Prediction counts:")
    for label in CLASS_NAMES:
        print(f"  {label}: {predicted_counts[label]}")

    if inferred_total > 0:
        accuracy = inferred_correct / inferred_total
        print()
        print(f"Approx accuracy on inferable filenames/folders: {accuracy:.4f} ({inferred_correct}/{inferred_total})")
    else:
        print()
        print("No ground-truth labels could be inferred from Downloads paths, so accuracy was not computed.")

    if skipped_images:
        print()
        print(f"Skipped unreadable images: {len(skipped_images)}")
        for image_path, reason in skipped_images[:10]:
            print(f"  {image_path.name}: {reason}")

    print(f"\nSaved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
