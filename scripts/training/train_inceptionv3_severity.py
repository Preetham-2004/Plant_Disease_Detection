import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import mixed_precision

# =========================
# GPU MEMORY FIX
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Mixed precision
mixed_precision.set_global_policy("mixed_float16")

# =========================
# CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR = Path("Dataset")

IMG_SIZE = (224, 224)   
BATCH_SIZE = 8          

EPOCHS_HEAD = 6
EPOCHS_FINE = 10

LR_HEAD = 1e-3
LR_FINE = 1e-5

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

LABEL2ID = {"Healthy": 0, "Moderate": 1, "Severe": 2}

AUTOTUNE = tf.data.AUTOTUNE

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
# DATASET SCAN
# =========================
print("Scanning dataset...")

files = list_images(DATASET_DIR)
print("Images found:", len(files))

rows = []
for p in files:
    rows.append((str(p), crop_from_path(p), infer_label(p)))

df = pd.DataFrame(rows, columns=["filepath", "crop", "label"])
df["y"] = df["label"].map(LABEL2ID)

print("\nLabel counts")
print(df["label"].value_counts())

print("\nCrop + label distribution")
print(df.groupby(["crop", "label"]).size().sort_values(ascending=False).head(30))


# =========================
# SPLIT
# =========================
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["y"], random_state=SEED)

print("\nSplit sizes", {"train": len(train_df), "val": len(val_df), "test": len(test_df)})


# =========================
# CLASS WEIGHTS
# =========================
cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2]), y=train_df["y"])
class_weight = {i: w for i, w in enumerate(cw)}

print("\nClass weights:", class_weight)


# =========================
# IMAGE PIPELINE
# =========================
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


data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.10),
])


def make_ds(frame, training):
    ds = tf.data.Dataset.from_tensor_slices((frame["filepath"].values, frame["y"].values))

    if training:
        ds = ds.shuffle(4000, seed=SEED)

    def load(path, y):
        img = decode_img(path)
        if training:
            img = data_aug(img)
        img = preprocess(img)
        return img, tf.one_hot(y, 3)

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)

    # ❌ REMOVED CACHE (IMPORTANT)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)

    return ds


train_ds = make_ds(train_df, True)
val_ds = make_ds(val_df, False)
test_ds = make_ds(test_df, False)


# =========================
# MODEL
# =========================
base = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(3, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# =========================
# TRAIN
# =========================
print("\nTraining head...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, class_weight=class_weight)

print("\nFine tuning...")
base.trainable = True
for layer in base.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, class_weight=class_weight)

print("\nTesting...")
print(model.evaluate(test_ds))