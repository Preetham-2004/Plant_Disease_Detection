# 🔒 CLASS MAPPING - MISLABELING PREVENTION GUIDE

## Quick Reference: How We Prevent Mislabeling

Your dataset uses a comprehensive mislabeling prevention system with multiple layers of protection.

---

## ✅ PROTECTION LAYER 1: Unified Class Registry

**File**: `unified_plant_dataset/data.yaml`

```yaml
names:
  0: eggplant_fruit      # Only definition of class 0
  1: eggplant_leaf       # Only definition of class 1
  2: potato_fruit        # Only definition of class 2
  3: potato_disease      # Only definition of class 3
  4: tomato_fruit        # Only definition of class 4
  5: tomato_leaf         # Only definition of class 5
```

**Protection**: Single source of truth. All labels must use IDs 0-5.

---

## ✅ PROTECTION LAYER 2: Fixed Mapping Rules

All 7 original datasets follow consistent remapping:

```
EGGPLANT FRUIT:
  big, medium, small → Class 0 (eggplant_fruit)

EGGPLANT LEAF:
  healthy, FS, GS, SS → Class 1 (eggplant_leaf)

POTATO FRUIT:
  brak, good → Class 2 (potato_fruit)

POTATO DISEASE:
  Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, Virus → Class 3

TOMATO FRUIT:
  tomato → Class 4 (tomato_fruit)

TOMATO LEAF:
  All 10 diseases + healthy → Class 5 (tomato_leaf)
```

**Protection**: No ambiguity. Each original class maps to exactly ONE unified class.

---

## ✅ PROTECTION LAYER 3: Automated Verification

**Use this anytime to verify no mislabeling has occurred:**

```bash
python3 verify_class_mapping.py
```

**What it checks**:
- ✓ All labels use valid class IDs (0-5)
- ✓ data.yaml matches actual labels
- ✓ Original mappings are consistent
- ✓ Train and val splits consistent
- ✓ No conflicting class definitions

**Expected output**:
```
✅ ALL CHECKS PASSED - CLASS MAPPING IS CORRECT!
✅ No mislabeling detected
✅ Dataset is safe for training
```

---

## ✅ PROTECTION LAYER 4: Standardized File Structure

All files follow a consistent format:

```
unified_plant_dataset/
├── data.yaml                    ← Master class registry
├── train/
│   ├── images/                  ← Raw images
│   └── labels/                  ← Labels with class IDs 0-5
└── val/
    ├── images/                  ← Raw images
    └── labels/                  ← Labels with class IDs 0-5
```

**Protection**: No hidden or inconsistent labels. Clear structure prevents mistakes.

---

## 🚨 How to Detect Mislabeling

### Signs of Problems

| Issue | What to Look For | How to Fix |
|-------|-----------------|-----------|
| **Invalid Class ID** | Class IDs outside 0-5 | Run verify_class_mapping.py |
| **Missing Classes** | Some classes not in train/val | Re-run merge script |
| **Inconsistent Mapping** | Different mappings in different files | Run verify_class_mapping.py |
| **Wrong data.yaml** | Class names don't match labels | Update data.yaml |

### Verification Command

```bash
# Quick check - should output ✅ PASS for all 5 checks
python3 verify_class_mapping.py
```

### Manual Spot Check

```python
# Check a few label files manually
import random
labels_dir = 'unified_plant_dataset/train/labels'
sample_files = random.sample(os.listdir(labels_dir), 5)

for file in sample_files:
    with open(f'{labels_dir}/{file}') as f:
        for line in f:
            class_id = int(line.split()[0])
            assert 0 <= class_id <= 5, f"Invalid class {class_id}"
            
print("✓ Sample check passed")
```

---

## 🔄 Adding New Data (Advanced)

If you need to add more datasets:

### Step 1: Update Mapping Rules

Edit `merge_unified_dataset.py`:

```python
# Add your new dataset mapping
DATASET_MAPPINGS = {
    # ... existing mappings ...
    'new_dataset.yolov8-obb': {
        0: 1,  # new_dataset class 0 → unified class 1
        1: 1,  # new_dataset class 1 → unified class 1
    }
}
```

### Step 2: Re-run Merger

```bash
python3 merge_unified_dataset.py
```

### Step 3: Verify

```bash
python3 verify_class_mapping.py
# Must output: ✅ ALL CHECKS PASSED
```

---

## 📊 Class Mapping Reference Table

Keep this handy when working with the dataset:

```
┌─────┬─────────────────┬──────────────────────────────────┐
│ ID  │ Unified Class   │ Source Classes                   │
├─────┼─────────────────┼──────────────────────────────────┤
│ 0   │ eggplant_fruit  │ big, medium, small (3 classes)   │
│ 1   │ eggplant_leaf   │ healthy, FS, GS, SS (4 classes)  │
│ 2   │ potato_fruit    │ brak, good (2 classes)           │
│ 3   │ potato_disease  │ 7 diseases + healthy             │
│ 4   │ tomato_fruit    │ tomato (1 class)                 │
│ 5   │ tomato_leaf     │ 10 diseases + healthy            │
└─────┴─────────────────┴──────────────────────────────────┘
```

---

## 💡 Best Practices

### When Training
```python
# Always verify before training
results = os.system('python3 verify_class_mapping.py')
if results != 0:
    raise Exception("Dataset verification failed!")

# Start training
model = YOLO('yolov8s.pt')
model.train(data='unified_plant_dataset/data.yaml', ...)
```

### When Debugging Results
```python
# If model performance seems wrong:
# 1. Check predictions match expectations
# 2. Verify class distribution is balanced
# 3. Run verification script
# 4. Review CLASS_MAPPING_VERIFICATION.md

python3 verify_class_mapping.py
```

### When Adding Data
```python
# 1. Update mapping rules in merge_unified_dataset.py
# 2. Re-run merge
# 3. Verify with verification script
# 4. Only then use new dataset

python3 merge_unified_dataset.py
python3 verify_class_mapping.py
```

---

## 🎯 Class-Specific Notes

### Class 0: eggplant_fruit (3.5% of data)
- Maps from: 3 size categories (big, medium, small)
- Interpretation: All eggplant fruits regardless of size
- Training note: Small class, monitor separately

### Class 1: eggplant_leaf (10.2% of data)
- Maps from: 4 severity levels
- Interpretation: All eggplant leaves (healthy + diseased)
- Training note: Medium class, should perform well

### Class 2: potato_fruit (26.4% of data)
- Maps from: 2 quality states (brak, good)
- Interpretation: All potato tubers
- Training note: Large class, well-represented

### Class 3: potato_disease (42.9% of data) ⭐
- Maps from: 7 disease types + healthy
- Interpretation: All potato diseases
- Training note: Dominant class, use weighted loss!

### Class 4: tomato_fruit (14.8% of data)
- Maps from: All tomato fruits
- Interpretation: Tomato fruits in any condition
- Training note: Well-balanced

### Class 5: tomato_leaf (2.1% of data) ⭐
- Maps from: 10 diseases + healthy
- Interpretation: All tomato leaf conditions
- Training note: Smallest class, needs special attention

---

## 🔐 Security Checklist

Before each training session:

- [ ] Run `python3 verify_class_mapping.py` → ✅ PASS
- [ ] Check data.yaml exists and is readable
- [ ] Verify all 6 classes present in both train and val
- [ ] Confirm no new invalid class IDs
- [ ] Review CLASS_MAPPING_VERIFICATION.md for any notes
- [ ] Plan class weighting strategy if needed

---

## 📋 Troubleshooting

### "verification failed" Error
```bash
# Check what failed:
python3 verify_class_mapping.py

# Then fix the issue by:
# 1. Checking data.yaml exists
# 2. Verifying unified_plant_dataset structure
# 3. Re-running merge_unified_dataset.py if needed
```

### "Invalid class ID" in Output
```bash
# Find which files have bad class IDs:
find unified_plant_dataset -name "*.txt" -exec \
  grep -l "^[6789]" {} \;

# These files need to be fixed
# Re-run merge_unified_dataset.py to regenerate
```

### "Missing classes in val split"
```bash
# Classes might not be represented uniformly
# This is normal with small datasets
# Solution: Check CLASS_MAPPING_VERIFICATION.md for distribution
```

---

## ✨ Summary

Your mislabeling prevention system includes:

1. ✅ **Unified registry** - Single source of truth
2. ✅ **Fixed mapping rules** - No ambiguity
3. ✅ **Automated verification** - Run anytime
4. ✅ **Standardized structure** - Clear organization
5. ✅ **Documentation** - This guide + detailed reports

**Result**: Maximum confidence that class mappings are correct.

---

## 📞 Quick Commands

```bash
# Verify class mapping
python3 verify_class_mapping.py

# View class mapping details
cat CLASS_MAPPING_VERIFICATION.md

# View data configuration
cat unified_plant_dataset/data.yaml

# Count instances per class
python3 << 'EOF'
import os
from collections import defaultdict
counts = defaultdict(int)
for f in os.listdir('unified_plant_dataset/train/labels'):
    with open(f'unified_plant_dataset/train/labels/{f}') as file:
        for line in file:
            counts[int(line.split()[0])] += 1
for k in sorted(counts): print(f"Class {k}: {counts[k]}")
EOF
```

---

**Last Updated**: 2026-03-30  
**Status**: ✅ All class mapping protection layers active  
**Safety Level**: 🔒 Maximum
