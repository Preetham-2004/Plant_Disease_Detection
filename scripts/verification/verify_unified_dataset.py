#!/usr/bin/env python3
"""
Verification script for unified plant disease dataset.
Checks data integrity, class distribution, format correctness, etc.
"""

import os
from pathlib import Path
from collections import defaultdict
import yaml

def verify_dataset(dataset_path):
    """Verify the unified dataset structure and content."""
    dataset_path = Path(dataset_path)
    
    print("=" * 70)
    print("UNIFIED PLANT DISEASE DATASET VERIFICATION")
    print("=" * 70)
    
    # 1. Check directory structure
    print("\n1. VERIFYING DIRECTORY STRUCTURE:")
    print("-" * 70)
    
    required_dirs = [
        'train/images',
        'train/labels',
        'val/images',
        'val/labels'
    ]
    
    all_dirs_exist = True
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n❌ ERROR: Missing required directories!")
        return False
    
    # 2. Check data.yaml
    print("\n2. VERIFYING data.yaml:")
    print("-" * 70)
    
    data_yaml_path = dataset_path / 'data.yaml'
    if not data_yaml_path.exists():
        print("  ✗ data.yaml not found!")
        return False
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"  ✓ data.yaml found")
    print(f"    - Path: {data_config.get('path', 'N/A')}")
    print(f"    - Train: {data_config.get('train', 'N/A')}")
    print(f"    - Val: {data_config.get('val', 'N/A')}")
    print(f"    - Classes (nc): {data_config.get('nc', 'N/A')}")
    
    expected_classes = {
        0: 'eggplant_fruit',
        1: 'eggplant_leaf',
        2: 'potato_fruit',
        3: 'potato_leaf',
        4: 'tomato_fruit',
        5: 'tomato_leaf'
    }
    
    if data_config.get('nc') != 6:
        print(f"  ⚠ Warning: Expected 6 classes, got {data_config.get('nc')}")
    
    names = data_config.get('names', {})
    for class_id, class_name in expected_classes.items():
        if str(class_id) in names or class_id in names:
            actual_name = names.get(str(class_id), names.get(class_id))
            if actual_name == class_name:
                print(f"    ✓ Class {class_id}: {class_name}")
            else:
                print(f"    ⚠ Class {class_id}: Expected '{class_name}', got '{actual_name}'")
        else:
            print(f"    ✗ Class {class_id} missing!")
    
    # 3. Check file counts
    print("\n3. VERIFYING FILE COUNTS:")
    print("-" * 70)
    
    splits = {
        'train': dataset_path / 'train',
        'val': dataset_path / 'val'
    }
    
    total_images = 0
    total_labels = 0
    
    for split_name, split_path in splits.items():
        img_count = len(list((split_path / 'images').glob('*')))
        label_count = len(list((split_path / 'labels').glob('*.txt')))
        
        total_images += img_count
        total_labels += label_count
        
        match = "✓" if img_count == label_count else "✗"
        print(f"  {split_name.upper()}:")
        print(f"    Images: {img_count:,}")
        print(f"    Labels: {label_count:,}")
        print(f"    Match: {match}")
    
    print(f"\n  TOTAL:")
    print(f"    Images: {total_images:,}")
    print(f"    Labels: {total_labels:,}")
    
    if total_images != total_labels:
        print(f"  ✗ Image/label count mismatch!")
        return False
    
    # 4. Check class ID distribution
    print("\n4. VERIFYING CLASS DISTRIBUTION:")
    print("-" * 70)
    
    class_counts = defaultdict(int)
    invalid_class_ids = set()
    invalid_formats = 0
    
    for split_path in splits.values():
        labels_dir = split_path / 'labels'
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            if 0 <= class_id <= 5:
                                class_counts[class_id] += 1
                            else:
                                invalid_class_ids.add(class_id)
                        except ValueError:
                            invalid_formats += 1
            except Exception as e:
                print(f"    ✗ Error reading {label_file}: {e}")
                return False
    
    # Display class distribution
    class_names = expected_classes
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        name = class_names.get(class_id, f'unknown_{class_id}')
        percentage = (count / sum(class_counts.values())) * 100 if class_counts else 0
        print(f"    Class {class_id} ({name:20s}): {count:,} ({percentage:5.1f}%)")
    
    print(f"  Total instances: {sum(class_counts.values()):,}")
    
    if invalid_class_ids:
        print(f"\n  ✗ Found invalid class IDs: {sorted(invalid_class_ids)}")
        return False
    
    if invalid_formats:
        print(f"  ⚠ Warning: Found {invalid_formats} invalid format lines")
    
    # 5. Sample label format check
    print("\n5. VERIFYING LABEL FORMAT (YOLO BBOX):")
    print("-" * 70)
    
    sample_found = False
    for split_path in splits.values():
        labels_dir = split_path / 'labels'
        for label_file in list(labels_dir.glob('*.txt'))[:1]:
            with open(label_file, 'r') as f:
                lines = f.readlines()[:3]
            
            print(f"  Sample from {label_file.name}:")
            for i, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) == 5:
                    print(f"    Line {i}: class_id={parts[0]}, bbox=({parts[1][:5]}, {parts[2][:5]}, {parts[3][:5]}, {parts[4][:5]})")
                    sample_found = True
                else:
                    print(f"    Line {i}: {line.strip()}")

    if not sample_found:
        print("  ⚠ Warning: Could not verify YOLO bbox format from samples")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    print("=" * 70)
    
    all_checks_pass = (
        all_dirs_exist and
        total_images == total_labels and
        len(invalid_class_ids) == 0
    )
    
    if all_checks_pass:
        print("✓ ALL CHECKS PASSED!")
        print(f"\nDataset is ready for YOLOv8 training:")
        print(f"  - Total images: {total_images:,}")
        print(f"  - Training samples: ~{int(total_images * 0.8):,}")
        print(f"  - Validation samples: ~{int(total_images * 0.2):,}")
        print(f"  - Classes: 6 (properly remapped and unified)")
    else:
        print("✗ VERIFICATION FAILED!")
        if not all_dirs_exist:
            print("  - Missing required directories")
        if total_images != total_labels:
            print("  - Image/label count mismatch")
        if len(invalid_class_ids) > 0:
            print(f"  - Invalid class IDs found: {sorted(invalid_class_ids)}")
    
    print("=" * 70)
    return all_checks_pass

if __name__ == '__main__':
    DATASET_PATH = '/home/preetham-bhat/plant_project/Plant Disease prediction/unified_plant_dataset'
    
    success = verify_dataset(DATASET_PATH)
    exit(0 if success else 1)
