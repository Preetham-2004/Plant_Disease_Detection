import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_PATH = Path("models/inceptionv3_severity_final.keras")
DATASET_DIR = Path("Dataset")
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
LABEL2ID = {"Healthy": 0, "Moderate": 1, "Severe": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
AUTOTUNE = tf.data.AUTOTUNE

# =========================
# HELPER FUNCTIONS (from your original script)
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

    # Healthy
    if "healthy_leaf" in joined or "healty_brinjal" in joined:
        return "Healthy"
    if "/healthy" in joined or joined.endswith("/healthy"):
        return "Healthy"
    if "/fruit/healthy" in joined:
        return "Healthy"
    if "healthy_potato" in joined:
        return "Healthy"

    # Severe
    severe_keys = [
        "late_blight", "yellow_leaf_curl_virus", "mosaic_virus",
        "mosaic_virus_disease", "/plant/viral", "/plant/wilt",
        "/plant/graymold", "virus", "nematode", "phytopthora",
        "bacterial_spot", "wet_rot", "brown_rot", "soft_rot",
        "dry_rot", "pink_rot", "blackleg", "shoot_and_fruit_borer",
    ]

    for k in severe_keys:
        if k in joined:
            return "Severe"

    fruit_severe_signals = ["rot", "hole", "black_hole", "gray_and_black_holes"]
    if "/fruit/" in joined:
        for s in fruit_severe_signals:
            if s in joined:
                return "Severe"

    return "Moderate"

# =========================
# LOAD DATASET
# =========================
print("Scanning dataset...")
files = list_images(DATASET_DIR)
print(f"Images found: {len(files)}")

rows = []
for p in files:
    label = infer_label(p)
    crop = crop_from_path(p)
    rows.append((str(p), crop, label))

df = pd.DataFrame(rows, columns=["filepath", "crop", "label"])
df["y"] = df["label"].map(LABEL2ID)

# Create test split (same as original)
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["y"]
)
_, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["y"]
)
print(f"Test set size: {len(test_df)}")

# =========================
# IMAGE LOADING FUNCTIONS
# =========================
def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.cond(
        tf.image.is_jpeg(img),
        lambda: tf.image.decode_jpeg(img, channels=3),
        lambda: tf.image.decode_png(img, channels=3),
    )
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img

def preprocess(img):
    return tf.keras.applications.inception_v3.preprocess_input(img)

def make_ds(frame: pd.DataFrame):
    paths = frame["filepath"].values
    ys = frame["y"].values.astype(np.int32)
    
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    
    def load_data(path, y):
        img = decode_img(path)
        img = preprocess(img)
        return img, y
    
    ds = ds.map(load_data, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

test_ds = make_ds(test_df)

# =========================
# LOAD MODEL AND PREDICT
# =========================
print(f"\nLoading model from {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for bx, by in test_ds:
    pred = model.predict(bx, verbose=0)
    y_true.extend(by.numpy())
    y_pred.extend(np.argmax(pred, axis=1))

# =========================
# CONFUSION MATRIX
# =========================
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=3).numpy()
print("\nConfusion Matrix:")
print(cm)

# =========================
# GENERATE HEATMAP
# =========================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Moderate', 'Severe'],
            yticklabels=['Healthy', 'Moderate', 'Severe'],
            linewidths=1, linecolor='black')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - InceptionV3 Severity Classification', fontsize=14)
plt.tight_layout()

# Save the heatmap
heatmap_path = Path('confusion_matrix_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {heatmap_path}")

# Show the heatmap
plt.show()

print("Done!")