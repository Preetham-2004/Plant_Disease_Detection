#!/usr/bin/env python3
"""Verify YOLO format conversion across train and val splits."""

import os
from pathlib import Path
from collections import defaultdict

def verify_split(split_name, labels_dir):
    """Verify a single train/val split."""
    print(f"\n{'='*70}")
    print(f"Verifying {split_name.upper()} Split")
    print(f"{'='*70}")
    
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        print(f"❌ Directory not found: {labels_dir}")
        return False
    
    label_files = list(labels_path.glob('*.txt'))
    print(f"\nTotal files: {len(label_files):,}")
    
    # Statistics
    total_boxes = 0
    files_with_boxes = 0
    empty_files = 0
    class_dist = defaultdict(int)
    coord_issues = []
    
    # Verification checks
    invalid_format = 0
    out_of_range = 0
    
    print("\nVerifying format and coordinates...")
    
    for idx, label_file in enumerate(label_files):
        if (idx + 1) % max(1, len(label_files) // 10) == 0:
            print(f"  [{idx+1:,}/{len(label_files):,}] processed", end='\r')
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                empty_files += 1
                continue
            
            files_with_boxes += 1
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Check format: should have exactly 5 fields
                if len(parts) != 5:
                    invalid_format += 1
                    coord_issues.append(f"{label_file.name}:{line_num} - Expected 5 fields, got {len(parts)}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Check class ID
                    if not 0 <= class_id <= 5:
                        out_of_range += 1
                        coord_issues.append(f"{label_file.name}:{line_num} - Invalid class ID: {class_id}")
                        continue
                    
                    # Check coordinates in [0, 1] range
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        out_of_range += 1
                        coord_issues.append(f"{label_file.name}:{line_num} - Coordinates out of range: "
                                          f"center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
                        continue
                    
                    # Valid box
                    total_boxes += 1
                    class_dist[class_id] += 1
                    
                except ValueError as e:
                    invalid_format += 1
                    coord_issues.append(f"{label_file.name}:{line_num} - {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"\n❌ Error processing {label_file.name}: {e}")
            continue
    
    print(f"  [{'✓' if invalid_format == 0 else '✗'}] Format validation complete")
    
    # Print results
    print(f"\n{'─'*70}")
    print(f"Summary:")
    print(f"{'─'*70}")
    print(f"  Files processed:        {len(label_files):,}")
    print(f"  Files with boxes:       {files_with_boxes:,}")
    print(f"  Empty files:            {empty_files:,}")
    print(f"  Total boxes:            {total_boxes:,}")
    print(f"  Invalid format:         {invalid_format:,}")
    print(f"  Out of range coords:    {out_of_range:,}")
    
    if class_dist:
        print(f"\nClass Distribution ({split_name}):")
        for class_id in sorted(class_dist.keys()):
            count = class_dist[class_id]
            pct = 100 * count / total_boxes
            print(f"  Class {class_id}: {count:,} boxes ({pct:5.1f}%)")
    
    # Print issues if any
    if coord_issues:
        print(f"\nFirst 10 issues found:")
        for issue in coord_issues[:10]:
            print(f"  - {issue}")
    
    success = invalid_format == 0 and out_of_range == 0
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: All coordinates valid")
    
    return {
        'files': len(label_files),
        'boxes': total_boxes,
        'errors': invalid_format + out_of_range,
        'classes': class_dist
    }

if __name__ == '__main__':
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "YOLO Format Conversion Verification".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    train_stats = verify_split('train', 'unified_plant_dataset/train/labels')
    val_stats = verify_split('val', 'unified_plant_dataset/val/labels')
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    if train_stats and val_stats:
        total_files = train_stats['files'] + val_stats['files']
        total_boxes = train_stats['boxes'] + val_stats['boxes']
        total_errors = train_stats['errors'] + val_stats['errors']
        
        print(f"\nCombined Statistics:")
        print(f"  Total files:            {total_files:,}")
        print(f"  Total boxes:            {total_boxes:,}")
        print(f"  Format errors:          {total_errors:,}")
        
        # Merge class distributions
        all_classes = {}
        for class_id in set(list(train_stats.get('classes', {}).keys()) + 
                           list(val_stats.get('classes', {}).keys())):
            count = train_stats.get('classes', {}).get(class_id, 0) + \
                   val_stats.get('classes', {}).get(class_id, 0)
            all_classes[class_id] = count
        
        print(f"\nOverall Class Distribution:")
        class_names = [
            'eggplant_fruit', 'eggplant_leaf', 'potato_fruit',
            'potato_leaf', 'tomato_fruit', 'tomato_leaf'
        ]
        for class_id in sorted(all_classes.keys()):
            count = all_classes[class_id]
            pct = 100 * count / total_boxes
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"  Class {class_id}: {name:20} - {count:,} ({pct:5.1f}%)")
        
        print(f"\n{'✅ CONVERSION SUCCESSFUL' if total_errors == 0 else '⚠️  CONVERSION INCOMPLETE'}:")
        if total_errors == 0:
            print(f"  All {total_boxes:,} boxes successfully converted to YOLO format")
        else:
            print(f"  {total_errors:,} boxes had format/coordinate issues")
    
    print(f"\n{'='*70}\n")
