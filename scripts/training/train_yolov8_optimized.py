#!/usr/bin/env python3
"""
YOLOv8 Training Script Optimized for RTX 4050 (4GB VRAM, 16GB RAM)

Configuration:
- GPU: RTX 4050 (4GB VRAM)
- RAM: 16GB
- Dataset: 8,158 images, 12,328 boxes
- Classes: 6 plant detector classes

Optimizations applied:
✓ Batch size: 8 (safer for 4GB VRAM with YOLOv8s)
✓ Image size: 640 (better detail for confusing leaf/fruit classes)
✓ Mixed precision: Enabled
✓ Model: YOLOv8s (better accuracy than nano)
✓ Workers: 4
"""

import torch
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    script_dir = Path(__file__).resolve().parent

    print("\n" + "="*70)
    print("YOLOv8 Training on Plant Disease Dataset")
    print("="*70)
    
    # Check GPU availability
    print(f"\nGPU Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  Device: cuda:0")
        device = 0
    else:
        print("  ⚠️  No GPU detected - training will be VERY slow on CPU!")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
        device = "cpu"
    
    # Dataset path
    dataset_path = script_dir / "unified_plant_dataset" / "data.yaml"
    
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print(f"Make sure the dataset exists next to this script: {script_dir}")
        sys.exit(1)
    
    print(f"\n✅ Dataset found: {dataset_path}")

    project_dir = script_dir / "runs" / "detect"
    run_name = "train_yolov8s_cleaned"
    
    # Load pretrained model (small - better accuracy)
    print("\nLoading YOLOv8 Small model...")
    model = YOLO('yolov8s.pt')
    
    # Display model info
    print("✅ Model loaded: YOLOv8 Small")
    print("   Parameters: ~11M")
    print("   Better fit for your confusion-heavy detector task")
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION (Optimized for RTX 4050)")
    print("="*70)
    
    config = {
        # Model and data
        'model': 'yolov8s.pt',
        'data': str(dataset_path),
        
        # Training parameters
        'epochs': 100,
        'patience': 20,  # Early stopping
        'imgsz': 640,  # Keep more detail for fine-grained class differences
        'batch': 8,  # Safer for YOLOv8s on 4GB VRAM
        'device': device,
        
        # Optimization
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Hardware
        'workers': 4,  # Data loading workers
        'amp': True,  # Automatic Mixed Precision (reduce VRAM ~30%)
        
        # Augmentation
        'augment': True,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,  # Disabled for small VRAM
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0,
        'translate': 0.1,
        'scale': 0.5,
        'perspective': 0.0,
        
        # Saving
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'project': str(project_dir),
        'name': run_name,
        'exist_ok': True,
        
        # Logging
        'verbose': True,
        'plots': True,
    }
    
    # Print config
    print("\nTraining Configuration:")
    for key, value in config.items():
        if key not in ['data', 'project', 'name', 'model']:
            print(f"  {key}: {value}")
    
    print(f"\n  data: {config['data']}")
    print(f"  model: {config['model']}")
    print(f"  project: {config['project']}")
    print(f"  name: {config['name']}")
    
    # Calculate expected memory usage
    print("\n" + "="*70)
    print("MEMORY ANALYSIS")
    print("="*70)
    print(f"\nEstimated VRAM usage:")
    print(f"  Model weights: ~450 MB")
    print(f"  Activations (batch_size=8): ~1.8 GB")
    print(f"  Optimizer states: ~700 MB")
    print(f"  Mixed precision buffer: ~400 MB")
    print(f"  ─────────────────────────────")
    print(f"  Total: ~3.3 GB / 4.0 GB ✅ TIGHT BUT REASONABLE")
    
    print(f"\nRAM usage (host):")
    print(f"  Data loading (workers=4): ~2-3 GB")
    print(f"  Model cache: ~500 MB")
    print(f"  System: ~2 GB")
    print(f"  ─────────────────────────────")
    print(f"  Total: ~5-6 GB / 16 GB ✅ PLENTY")
    
    print("\n" + "="*70)
    print("STARTING TRAINING...")
    print("="*70)
    print("\nℹ️  First epoch will be slower (data loading optimization)")
    print("ℹ️  Subsequent epochs will be slower than YOLOv8n")
    print("ℹ️  Total training time may be ~5-7 hours for 100 epochs\n")
    
    # Start training
    try:
        results = model.train(**config)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY! ✅")
        print("="*70)
        
        # Validation
        print("\nRunning validation...")
        best_model_path = project_dir / run_name / "weights" / "best.pt"
        trained_model = YOLO(str(best_model_path)) if best_model_path.exists() else model
        metrics = trained_model.val(data=str(dataset_path), device=device)
        
        print("\nFinal Metrics:")
        print(f"  mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # Predictions
        print("\n" + "="*70)
        print("MODEL SAVED")
        print("="*70)
        print(f"Best model: {project_dir / run_name / 'weights' / 'best.pt'}")
        print(f"Last model: {project_dir / run_name / 'weights' / 'last.pt'}")
        
        print("\nTo use the model for predictions:")
        print(f"  model = YOLO('{project_dir / run_name / 'weights' / 'best.pt'}')")
        print("  results = model.predict(source='image.jpg')")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
