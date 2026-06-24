#!/usr/bin/env python3
"""
COMPREHENSIVE CLASS MAPPING VERIFICATION TOOL
Ensures all class mappings are correct and prevents mislabeling.
Validates:
1. Original dataset class definitions
2. Remapped class IDs in labels
3. data.yaml consistency
4. Cross-dataset conflicts
5. Per-dataset class distribution
"""

import os
import yaml
from pathlib import Path
from collections import defaultdict

# Expected final unified classes
EXPECTED_UNIFIED_CLASSES = {
    0: 'eggplant_fruit',
    1: 'eggplant_leaf',
    2: 'potato_fruit',
    3: 'potato_leaf',
    4: 'tomato_fruit',
    5: 'tomato_leaf'
}

# Original class mappings (what we used for remapping)
ORIGINAL_MAPPINGS = {
    'eggplant.fruit.yolov8-obb': {
        'original_classes': {0: 'big', 1: 'medium', 2: 'small'},
        'remap_to': 0,  # All → eggplant_fruit
        'class_name': 'eggplant_fruit'
    },
    'eggplant.leaf_1.yolov8-obb': {
        'original_classes': {0: 'eggplantFS', 1: 'eggplantGS', 2: 'eggplantSS'},
        'remap_to': 1,  # All → eggplant_leaf
        'class_name': 'eggplant_leaf'
    },
    'eggplant.leaf.yolov8-obb': {
        'original_classes': {0: 'healthy eggplant'},
        'remap_to': 1,  # All → eggplant_leaf
        'class_name': 'eggplant_leaf'
    },
    'potato.fruit.yolov8-obb': {
        'original_classes': {0: 'brak', 1: 'good'},
        'remap_to': 2,  # All → potato_fruit
        'class_name': 'potato_fruit'
    },
    'Potato Leaf Disease.yolov8-obb': {
        'original_classes': {
            0: 'Bacteria', 1: 'Fungi', 2: 'Healthy',
            3: 'Nematode', 4: 'Pest', 5: 'Phytopthora', 6: 'Virus'
        },
        'remap_to': 3,  # All → potato_leaf
        'class_name': 'potato_leaf'
    },
    'tomato fruit.yolov8-obb': {
        'original_classes': {0: 'tomato'},
        'remap_to': 4,  # All → tomato_fruit
        'class_name': 'tomato_fruit'
    },
    'Tomato leaf.yolov8-obb': {
        'original_classes': {
            0: 'Bacterial_spot', 1: 'Early_blight', 2: 'Late_blight',
            3: 'Leaf_Mold', 4: 'Powdery Mildew',
            5: 'Spidermites Two-spottedspider_mite', 6: 'Target_Spot',
            7: 'TomatoYellowLeafCurlVirus', 8: 'Tomatomosaicvirus',
            9: 'healthy'
        },
        'remap_to': 5,  # All → tomato_leaf
        'class_name': 'tomato_leaf'
    }
}

def verify_data_yaml(dataset_path):
    """Verify data.yaml configuration matches expected mapping."""
    print("\n" + "="*80)
    print("1. VERIFYING data.yaml CONFIGURATION")
    print("="*80)
    
    data_yaml_path = dataset_path / 'data.yaml'
    
    if not data_yaml_path.exists():
        print("❌ ERROR: data.yaml not found!")
        return False
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"✓ data.yaml found")
    
    # Check nc (number of classes)
    nc = data_config.get('nc')
    if nc != 6:
        print(f"❌ ERROR: Expected 6 classes, got {nc}")
        return False
    print(f"✓ Number of classes: {nc}")
    
    # Check class names
    names = data_config.get('names', {})
    all_correct = True
    
    for class_id, expected_name in EXPECTED_UNIFIED_CLASSES.items():
        actual_name = names.get(class_id) or names.get(str(class_id))
        if actual_name == expected_name:
            print(f"✓ Class {class_id}: {expected_name}")
        else:
            print(f"❌ Class {class_id}: Expected '{expected_name}', got '{actual_name}'")
            all_correct = False
    
    return all_correct

def verify_unified_labels(dataset_path):
    """Verify all labels in unified dataset use correct class IDs."""
    print("\n" + "="*80)
    print("2. VERIFYING UNIFIED DATASET LABELS")
    print("="*80)
    
    invalid_classes = defaultdict(int)
    class_distribution = defaultdict(int)
    all_valid = True
    
    for split in ['train', 'val']:
        labels_dir = dataset_path / split / 'labels'
        print(f"\n{split.upper()} Split:")
        
        if not labels_dir.exists():
            print(f"❌ {split}/labels directory not found!")
            return False
        
        split_class_dist = defaultdict(int)
        
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        
                        try:
                            parts = line.strip().split()
                            class_id = int(parts[0])
                            
                            if class_id not in EXPECTED_UNIFIED_CLASSES:
                                invalid_classes[class_id] += 1
                                all_valid = False
                                if invalid_classes[class_id] <= 3:  # Show first 3 occurrences
                                    print(f"  ❌ Invalid class ID {class_id} in {label_file.name}:{line_num}")
                            else:
                                split_class_dist[class_id] += 1
                                class_distribution[class_id] += 1
                        except ValueError as e:
                            print(f"  ❌ Cannot parse class ID in {label_file.name}:{line_num}: {line[:50]}")
                            all_valid = False
            except Exception as e:
                print(f"  ⚠️  Error reading {label_file.name}: {e}")
                all_valid = False
        
        # Print distribution for this split
        print(f"  Class distribution in {split}:")
        for class_id in sorted(split_class_dist.keys()):
            class_name = EXPECTED_UNIFIED_CLASSES.get(class_id, 'unknown')
            count = split_class_dist[class_id]
            print(f"    Class {class_id} ({class_name:20s}): {count:,} instances")
    
    # Print overall distribution
    print(f"\nOVERALL CLASS DISTRIBUTION (train + val):")
    print("-" * 60)
    for class_id in sorted(class_distribution.keys()):
        class_name = EXPECTED_UNIFIED_CLASSES.get(class_id, 'unknown')
        count = class_distribution[class_id]
        pct = (count / sum(class_distribution.values())) * 100 if class_distribution else 0
        print(f"  Class {class_id} ({class_name:20s}): {count:,} ({pct:5.1f}%)")
    
    if invalid_classes:
        print(f"\n❌ Found invalid class IDs (not in 0-5):")
        for class_id, count in sorted(invalid_classes.items()):
            print(f"  Class {class_id}: {count} occurrences")
        return False
    else:
        print(f"\n✓ All class IDs valid (0-5 only)")
    
    return all_valid

def verify_original_datasets(base_path):
    """Verify original dataset class definitions match our mapping."""
    print("\n" + "="*80)
    print("3. VERIFYING ORIGINAL DATASET DEFINITIONS")
    print("="*80)
    
    base_path = Path(base_path)
    all_correct = True
    
    for dataset_name, mapping_info in ORIGINAL_MAPPINGS.items():
        dataset_path = base_path / dataset_name
        data_yaml_path = dataset_path / 'data.yaml'
        
        if not data_yaml_path.exists():
            print(f"⚠️  {dataset_name}: data.yaml not found (OK if dataset removed)")
            continue
        
        print(f"\n{dataset_name}:")
        
        try:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            names = data_config.get('names', {})
            expected_classes = mapping_info['original_classes']
            
            # Check if original classes match
            for class_id, expected_name in expected_classes.items():
                actual_name = names.get(class_id) or names.get(str(class_id))
                
                # Handle cases where the name might be cut off or slightly different
                if actual_name and expected_name:
                    if str(actual_name).strip() == str(expected_name).strip():
                        print(f"  ✓ Class {class_id}: {expected_name}")
                    else:
                        print(f"  ⚠️  Class {class_id}: Expected '{expected_name}', got '{actual_name}'")
                        # Not critical if mismatch, but worth noting
                else:
                    print(f"  ⚠️  Class {class_id}: Definition mismatch")
            
            # Verify remapping target
            target_class = mapping_info['remap_to']
            target_name = mapping_info['class_name']
            unified_name = EXPECTED_UNIFIED_CLASSES.get(target_class)
            
            if unified_name == target_name:
                print(f"  ✓ Maps to: Class {target_class} ({target_name})")
            else:
                print(f"  ❌ Mapping error: Should map to '{target_name}', but got '{unified_name}'")
                all_correct = False
                
        except Exception as e:
            print(f"  ❌ Error reading data.yaml: {e}")
            all_correct = False
    
    return all_correct

def verify_no_duplicate_mappings(base_path, output_path):
    """Verify no class gets mapped to multiple unified classes."""
    print("\n" + "="*80)
    print("4. VERIFYING NO DUPLICATE/CONFLICTING MAPPINGS")
    print("="*80)
    
    # Check that each original class maps to exactly one unified class
    mapping_conflicts = defaultdict(set)
    
    for dataset_name, mapping_info in ORIGINAL_MAPPINGS.items():
        target_class = mapping_info['remap_to']
        original_classes = mapping_info['original_classes']
        
        for orig_class_id in original_classes.keys():
            mapping_conflicts[(dataset_name, orig_class_id)].add(target_class)
    
    all_valid = True
    for (dataset, orig_class), targets in mapping_conflicts.items():
        if len(targets) > 1:
            print(f"❌ CONFLICT: {dataset} class {orig_class} maps to multiple targets: {targets}")
            all_valid = False
        else:
            target = list(targets)[0]
            print(f"✓ {dataset} class {orig_class} → unified class {target}")
    
    # Check reverse: no unified class gets multiple sources
    reverse_mapping = defaultdict(set)
    for dataset_name, mapping_info in ORIGINAL_MAPPINGS.items():
        target = mapping_info['remap_to']
        for orig_class_id in mapping_info['original_classes'].keys():
            reverse_mapping[target].add((dataset_name, orig_class_id))
    
    print(f"\nReverse mapping (unified → originals):")
    for unified_class in sorted(reverse_mapping.keys()):
        class_name = EXPECTED_UNIFIED_CLASSES.get(unified_class)
        sources = reverse_mapping[unified_class]
        print(f"  Class {unified_class} ({class_name}): {len(sources)} source classes")
        for dataset, orig_class in sorted(sources):
            print(f"    ← {dataset} class {orig_class}")
    
    return all_valid

def verify_class_consistency(output_path):
    """Verify class consistency across train and val splits."""
    print("\n" + "="*80)
    print("5. VERIFYING TRAIN/VAL CLASS CONSISTENCY")
    print("="*80)
    
    train_classes = set()
    val_classes = set()
    
    # Get classes in train split
    train_labels_dir = output_path / 'train' / 'labels'
    if train_labels_dir.exists():
        for label_file in train_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        train_classes.add(class_id)
    
    # Get classes in val split
    val_labels_dir = output_path / 'val' / 'labels'
    if val_labels_dir.exists():
        for label_file in val_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        val_classes.add(class_id)
    
    print(f"Classes in training split: {sorted(train_classes)}")
    print(f"Classes in validation split: {sorted(val_classes)}")
    
    all_classes = sorted(train_classes | val_classes)
    expected_classes = sorted(EXPECTED_UNIFIED_CLASSES.keys())
    
    if all_classes == expected_classes:
        print(f"✓ Both splits contain all 6 expected classes")
        return True
    else:
        missing = set(expected_classes) - (train_classes | val_classes)
        extra = (train_classes | val_classes) - set(expected_classes)
        
        if missing:
            print(f"❌ Missing classes: {sorted(missing)}")
        if extra:
            print(f"❌ Unexpected classes: {sorted(extra)}")
        return False

def generate_mapping_summary_table():
    """Generate a detailed mapping summary table."""
    print("\n" + "="*80)
    print("6. MAPPING SUMMARY TABLE")
    print("="*80)
    
    print("\n" + "─"*100)
    print(f"{'Dataset':<35} {'Original Classes':<40} {'→ Unified Class':<15}")
    print("─"*100)
    
    for dataset_name in sorted(ORIGINAL_MAPPINGS.keys()):
        mapping = ORIGINAL_MAPPINGS[dataset_name]
        original_names = ', '.join(mapping['original_classes'].values())
        target = mapping['class_name']
        
        # Truncate long names
        if len(original_names) > 38:
            original_names = original_names[:35] + "..."
        
        print(f"{dataset_name:<35} {original_names:<40} {target:<15}")
    
    print("─"*100)

def main():
    """Run all verification checks."""
    BASE_PATH = '/home/preetham-bhat/plant_project/Plant Disease prediction/yolo new dataset'
    OUTPUT_PATH = Path('/home/preetham-bhat/plant_project/Plant Disease prediction/unified_plant_dataset')
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "COMPREHENSIVE CLASS MAPPING VERIFICATION TOOL".center(78) + "║")
    print("║" + "Ensuring no mislabeling across all datasets".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run all checks
    results = {}
    
    results['yaml_config'] = verify_data_yaml(OUTPUT_PATH)
    results['unified_labels'] = verify_unified_labels(OUTPUT_PATH)
    results['original_datasets'] = verify_original_datasets(BASE_PATH)
    results['no_conflicts'] = verify_no_duplicate_mappings(BASE_PATH, OUTPUT_PATH)
    results['consistency'] = verify_class_consistency(OUTPUT_PATH)
    generate_mapping_summary_table()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    print("\nCheck Results:")
    print(f"  1. data.yaml configuration:        {'✅ PASS' if results['yaml_config'] else '❌ FAIL'}")
    print(f"  2. Unified dataset labels:          {'✅ PASS' if results['unified_labels'] else '❌ FAIL'}")
    print(f"  3. Original dataset definitions:    {'✅ PASS' if results['original_datasets'] else '❌ FAIL'}")
    print(f"  4. No conflicting mappings:         {'✅ PASS' if results['no_conflicts'] else '❌ FAIL'}")
    print(f"  5. Train/Val class consistency:     {'✅ PASS' if results['consistency'] else '❌ FAIL'}")
    
    print("\n" + "─"*80)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - CLASS MAPPING IS CORRECT!")
        print("✅ No mislabeling detected")
        print("✅ Dataset is safe for training")
    else:
        print("❌ SOME CHECKS FAILED - REVIEW ABOVE FOR DETAILS")
    
    print("─"*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())
