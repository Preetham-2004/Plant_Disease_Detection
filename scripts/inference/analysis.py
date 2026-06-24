import random
from pathlib import Path
import os
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR = Path("Dataset")
IMG_SIZE = (299, 299)
BATCH_SIZE = 8

LABEL2ID = {"Healthy": 0, "Moderate": 1, "Severe": 2}

AUTOTUNE = tf.data.AUTOTUNE

sns.set_style("whitegrid")

# =========================
# OUTPUT DIRECTORY
# =========================
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print("\n🔄 Loading trained model...")
model = tf.keras.models.load_model("models/inceptionv3_severity_final.keras")
print("✅ Model loaded successfully!")

# =========================
# HELPERS
# =========================
def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def crop_from_path(p: Path):
    parts = list(p.parts)
    for i, token in enumerate(parts):
        if token.lower() == "dataset" and i + 1 < len(parts):
            return parts[i + 1]
    return "Unknown"


def norm(s):
    return s.strip().lower()


def infer_label(p: Path):
    parts = [norm(x) for x in p.parts]
    joined = "/".join(parts)

    if "healthy_leaf" in joined or "healty_brinjal" in joined:
        return "Healthy"
    if "/healthy" in joined or "/fruit/healthy" in joined or "healthy_potato" in joined:
        return "Healthy"

    severe_keys = [
        "late_blight","yellow_leaf_curl_virus","mosaic_virus","virus",
        "nematode","phytopthora","bacterial_spot","wet_rot","brown_rot",
        "soft_rot","dry_rot","pink_rot","blackleg","shoot_and_fruit_borer",
    ]
    for k in severe_keys:
        if k in joined:
            return "Severe"

    if "/fruit/" in joined:
        if any(s in joined for s in ["rot", "hole", "black_hole"]):
            return "Severe"

    return "Moderate"


# =========================
# DATASET LOAD
# =========================
print("\n📂 Scanning dataset...")

files = list_images(DATASET_DIR)
print("Images found:", len(files))

rows = []
for p in files:
    rows.append((str(p), crop_from_path(p), infer_label(p)))

df = pd.DataFrame(rows, columns=["filepath", "crop", "label"])
df["y"] = df["label"].map(LABEL2ID)

print("📊 Dataset ready!")

# =========================
# TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

_, temp_df = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
_, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["y"], random_state=SEED)


def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.cond(
        tf.image.is_jpeg(img),
        lambda: tf.image.decode_jpeg(img, channels=3),
        lambda: tf.image.decode_png(img, channels=3),
    )
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32)


def preprocess(img):
    return tf.keras.applications.inception_v3.preprocess_input(img)


def make_ds(frame):
    ds = tf.data.Dataset.from_tensor_slices((frame["filepath"].values, frame["y"].values))

    def load(path, y):
        img = decode_img(path)
        img = preprocess(img)
        return img, tf.one_hot(y, 3)

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)

    return ds


test_ds = make_ds(test_df)

# =========================
# ANALYSIS
# =========================
print("\n" + "="*60)
print("📄 PROJECT ANALYSIS & RESULTS")
print("="*60)

# MODEL PERFORMANCE
loss, acc = model.evaluate(test_ds, verbose=0)

print("\n🧠 MODEL PERFORMANCE")
print(f"Accuracy: {acc:.4f}")
print(f"Loss: {loss:.4f}")

# PREDICTIONS
print("\n🔍 Generating predictions...")

y_true, y_pred = [], []

for bx, by in test_ds:
    pred = model.predict(bx, verbose=0)
    y_true.extend(np.argmax(by.numpy(), axis=1))
    y_pred.extend(np.argmax(pred, axis=1))

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=3).numpy()

print("\nConfusion Matrix:\n", cm)

# =========================
# SAVE GRAPHS
# =========================

# Confusion Matrix
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Healthy","Moderate","Severe"],
            yticklabels=["Healthy","Moderate","Severe"])
plt.title("Confusion Matrix")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# Class Distribution
dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.figure(figsize=(8,5))
sns.countplot(x=df["label"], order=["Healthy","Moderate","Severe"])
plt.title("Class Distribution")
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
plt.close()

# Prediction Distribution
pred_path = os.path.join(OUTPUT_DIR, "prediction_distribution.png")
plt.figure(figsize=(8,5))
sns.countplot(x=y_pred)
plt.xticks([0,1,2], ["Healthy","Moderate","Severe"])
plt.title("Prediction Distribution")
plt.savefig(pred_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n📁 Saved outputs:")
print(cm_path)
print(dist_path)
print(pred_path)

# =========================
# AUTO OPEN (LINUX)
# =========================
print("\n📂 Opening graphs...")

subprocess.run(["xdg-open", cm_path])
subprocess.run(["xdg-open", dist_path])
subprocess.run(["xdg-open", pred_path])

print("\n✅ Analysis complete!")
print("="*60)