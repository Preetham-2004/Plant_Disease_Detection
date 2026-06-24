#!/usr/bin/env python3
"""
Quick-start guide for training YOLOv8 models on the unified plant disease dataset.
"""

# ============================================================================
# QUICK START: YOLOv8 Training on Unified Plant Disease Dataset
# ============================================================================

# 1. INSTALL ULTRALYTICS (if not already installed)
# pip install ultralytics opencv-python

# 2. BASIC TRAINING EXAMPLE
# ============================================================================

from ultralytics import YOLO
import torch

# Verify GPU is available
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

# Path to dataset config
DATA_YAML = '/home/preetham-bhat/plant_project/Plant Disease prediction/unified_plant_dataset/data.yaml'

# Load a pretrained YOLOv8 model (nano, small, medium, large, xlarge)
# Nano (yolov8n) - fastest, least accurate
# Small (yolov8s) - balanced
# Medium (yolov8m) - better accuracy, slower
model = YOLO('yolov8s.pt')  # Start with 'yolov8s.pt' for good balance

# Train the model
results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    device=0,  # GPU ID (0 for first GPU, or 'cpu' for CPU)
    patience=20,  # Early stopping after 20 epochs with no improvement
    save=True,
    cache=True,  # Cache images for faster training
    batch=32,  # Batch size (adjust based on GPU memory)
    augment=True,  # Data augmentation
    hsv_h=0.015,  # HSV hue augmentation
    hsv_s=0.7,  # HSV saturation augmentation
    hsv_v=0.4,  # HSV value augmentation
    fliplr=0.5,  # Flip left-right probability
    flipud=0.1,  # Flip up-down probability
    mosaic=1.0,  # Mosaic augmentation
    optimizer='SGD',  # SGD or Adam
    lr0=0.01,  # Initial learning rate
    lrf=0.01,  # Final learning rate (decay)
    momentum=0.937,  # SGD momentum
    weight_decay=0.0005,  # L2 weight decay
    warmup_epochs=3,  # Linear warmup
    warmup_momentum=0.8,
    cos_lr=True,  # Use cosine annealing for learning rate decay
    name='plant_detection_v1'  # Experiment name
)

# ============================================================================
# 3. VALIDATE THE MODEL
# ============================================================================

metrics = model.val()
print(f"Validation Results:")
print(f"  mAP50: {metrics.box.map50:.3f}")
print(f"  mAP50-95: {metrics.box.map:.3f}")

# ============================================================================
# 4. TEST INFERENCE
# ============================================================================

# Predict on a single image
results = model.predict(source='path/to/image.jpg', conf=0.5)

# Predict on a video
results = model.predict(source='path/to/video.mp4', conf=0.5)

# Predict on webcam (0 = default webcam)
results = model.predict(source=0, conf=0.5)

# ============================================================================
# 5. EXPORT MODEL FOR DEPLOYMENT
# ============================================================================

# Export to ONNX
model.export(format='onnx')

# Export to TensorFlow SavedModel
model.export(format='saved_model')

# Export to CoreML (macOS)
model.export(format='coreml')

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

# Let YOLO auto-find best hyperparameters
model = YOLO('yolov8s.pt')
results = model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    tune=True,  # Enable hyperparameter tuning
)

# ============================================================================
# CLASS INFORMATION
# ============================================================================

CLASSES = {
    0: 'eggplant_fruit',
    1: 'eggplant_leaf',
    2: 'potato_fruit',
    3: 'potato_disease',
    4: 'tomato_fruit',
    5: 'tomato_leaf'
}

print("Class Mapping:")
for class_id, class_name in CLASSES.items():
    print(f"  {class_id}: {class_name}")

# ============================================================================
# ADVANCED: CUSTOM TRAINING WITH DIFFERENT MODEL SIZES
# ============================================================================

import os
from datetime import datetime

# Train multiple models with different sizes
model_sizes = ['n', 's', 'm']  # nano, small, medium
results_dict = {}

for size in model_sizes:
    print(f"\n{'='*60}")
    print(f"Training YOLOv8{size.upper()}...")
    print(f"{'='*60}")
    
    model = YOLO(f'yolov8{size}.pt')
    
    exp_name = f'plant_detection_{size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        device=0,
        patience=20,
        name=exp_name
    )
    
    results_dict[size] = results
    
    # Validate
    metrics = model.val()
    print(f"YOLOv8{size.upper()} - mAP50: {metrics.box.map50:.3f}")

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Issue: "CUDA Out of Memory"
# Solution: Reduce batch size or image size
# model.train(..., batch=16, imgsz=480)

# Issue: "Very low validation accuracy"
# Solution: Check that data.yaml path is correct
# Verify directory structure exists

# Issue: "Model not improving"
# Solution: Increase patience, adjust learning rate, or train longer
# model.train(..., patience=30, epochs=200)

# ============================================================================
# USEFUL COMMANDS
# ============================================================================

# CLI Training
# yolo detect train data=/path/to/data.yaml model=yolov8s.pt epochs=100 imgsz=640

# CLI Validation
# yolo detect val model=runs/detect/train/weights/best.pt data=/path/to/data.yaml

# CLI Prediction
# yolo detect predict model=runs/detect/train/weights/best.pt source=/path/to/image.jpg

print("\n✓ Quick start guide loaded. Run this script to train YOLOv8 models!")
