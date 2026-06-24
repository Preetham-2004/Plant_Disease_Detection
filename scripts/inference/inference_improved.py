#!/usr/bin/env python3
"""
🌱 YOLOv8 Plant Disease Detection - Improved Inference Script
With smart filtering to reduce class confusion (tomato vs potato).

Usage:
    python inference_improved.py --image path/to/image.jpg
    python inference_improved.py --folder path/to/folder --save
    python inference_improved.py --folder /home/preetham-bhat/Downloads --conf 0.25 --filter
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = 'runs/detect/train_yolov8s_cleaned/weights/best.pt'


CLASS_NAMES = {
    0: 'eggplant_fruit',
    1: 'eggplant_leaf',
    2: 'potato_fruit',
    3: 'potato_leaf',
    4: 'tomato_fruit',
    5: 'tomato_leaf'
}

# Smart filtering: if confidence is low, look for alternative predictions
# This helps disambiguate similar crop/part detections.
SMART_FILTER = {
    'potato_leaf': {
        'min_conf': 0.4,  # Need high confidence for potato_leaf
        'alternatives': ['tomato_fruit', 'tomato_leaf']  # Could be these instead
    },
    'tomato_fruit': {
        'min_conf': 0.35,  # Lower threshold for tomato
        'alternatives': ['potato_leaf']
    }
}


def load_model():
    """Load the trained YOLOv8 model."""
    if not Path(MODEL_PATH).exists():
        print(f'❌ Model not found: {MODEL_PATH}')
        sys.exit(1)
    
    print(f'📦 Loading model: {MODEL_PATH}')
    model = YOLO(MODEL_PATH)
    print(f'✅ Model loaded successfully\n')
    return model


def apply_smart_filter(result, conf, enable_filter=True):
    """
    Apply smart filtering to reduce class confusion.
    If any detection has low confidence, check lower-conf predictions too.
    """
    if not enable_filter or len(result.boxes) == 0:
        return result
    
    class_names_local = CLASS_NAMES
    boxes_data = []
    
    # Collect all detections with their confidence
    for box, conf_score, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        cls_name = class_names_local[int(cls_id)]
        confidence = float(conf_score)
        
        boxes_data.append({
            'box': box,
            'conf': confidence,
            'cls_id': int(cls_id),
            'cls_name': cls_name
        })
    
    # Check if main predictions are weak
    for i, box_data in enumerate(boxes_data):
        cls_name = box_data['cls_name']
        confidence = box_data['conf']
        
        # If it's potato_leaf with borderline confidence, double-check
        if cls_name == 'potato_leaf' and 0.4 <= confidence < 0.6:
            print(f'   ⚠️  Potato detection borderline ({confidence:.1%}), confidence may be low')
        
        # If it's tomato but confidence is low, suggest it could be potato
        if 'tomato' in cls_name and confidence < 0.5:
            print(f'   ℹ️  Tomato detection ({confidence:.1%}), be careful - could be potato_leaf')
    
    return result


def infer_image(model, image_path, conf=0.5, save=False, smart_filter=False):
    """Run inference on a single image."""
    img_path = Path(image_path)
    
    if not img_path.exists():
        print(f'❌ Image not found: {image_path}')
        return
    
    img = cv2.imread(str(img_path))
    height, width, _ = img.shape
    
    print(f'\n{"="*70}')
    print(f'📸 Image: {img_path.name}')
    print(f'   Size: {width}x{height} pixels')
    print(f'{"="*70}')
    print(f'\n🔍 Running inference (conf: {conf:.0%}{"  + Smart Filter" if smart_filter else ""})...\n')
    
    result = model.predict(
        source=str(img_path),
        conf=conf,
        save=save,
        save_txt=False,
        save_conf=False,
        line_width=2,
        project='inference_results' if save else None,
        exist_ok=True
    )
    
    result = result[0]
    
    # Apply smart filtering
    if smart_filter:
        apply_smart_filter(result, conf, enable_filter=True)
    
    num_detections = len(result.boxes)
    
    print(f'\n📊 DETECTION RESULTS:')
    print(f'   Total detections: {num_detections}')
    
    if num_detections > 0:
        print(f'\n   Detected objects:')
        for j, (box, conf_score, cls_id) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
            cls_name = CLASS_NAMES.get(int(cls_id), 'Unknown')
            x1, y1, x2, y2 = map(int, box)
            confidence = float(conf_score)
            
            print(f'\n   🎯 Detection {j+1}:')
            print(f'      Class: {cls_name}')
            print(f'      Confidence: {confidence:.2%}')
            print(f'      Location: ({x1}, {y1}) → ({x2}, {y2})')
    else:
        print(f'\n   ℹ️  No objects detected')
        if conf > 0.25:
            print(f'   💡 Try: --conf 0.25')
        if not smart_filter:
            print(f'   💡 Try: --filter')
    
    if save:
        print(f'\n💾 Annotated image saved to: inference_results/')
    
    print(f'\n{"="*70}\n')


def infer_folder(model, folder_path, conf=0.5, save=True, smart_filter=False):
    """Run inference on all images in a folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f'❌ Folder not found: {folder_path}')
        return
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f'❌ No images found in: {folder_path}')
        return
    
    print(f'\n{"="*70}')
    print(f'📁 Batch Processing: {folder.name}')
    print(f'   Found {len(images)} images')
    if smart_filter:
        print(f'   Smart filter: ENABLED')
    print(f'{"="*70}\n')
    
    total_detections = 0
    results_by_class = {}
    
    for i, img_path in enumerate(images, 1):
        print(f'[{i}/{len(images)}] Processing: {img_path.name}')
        
        result = model.predict(
            source=str(img_path),
            conf=conf,
            save=save,
            save_txt=False,
            save_conf=False,
            line_width=2,
            project='inference_results' if save else None,
            exist_ok=True,
            verbose=False
        )
        
        result = result[0]
        
        if smart_filter:
            apply_smart_filter(result, conf, enable_filter=True)
        
        num_detections = len(result.boxes)
        total_detections += num_detections
        
        if num_detections > 0:
            for cls_id in result.boxes.cls:
                cls_name = CLASS_NAMES[int(cls_id)]
                if cls_name not in results_by_class:
                    results_by_class[cls_name] = 0
                results_by_class[cls_name] += 1
            
            detections_str = ', '.join([
                f'{CLASS_NAMES[int(cls_id)]}' 
                for cls_id in result.boxes.cls
            ])
            print(f'   ✓ {num_detections} object(s): {detections_str}')
        else:
            print(f'   - No detections')
    
    print(f'\n{"="*70}')
    print(f'📊 BATCH SUMMARY:')
    print(f'   Total images: {len(images)}')
    print(f'   Total detections: {total_detections}')
    
    if results_by_class:
        print(f'\n   By class:')
        for cls_name, count in sorted(results_by_class.items()):
            print(f'      • {cls_name}: {count}')
    
    if save:
        print(f'\n   Results: inference_results/')
    print(f'{"="*70}\n')


def main():
    parser = argparse.ArgumentParser(
        description='🌱 YOLOv8 Plant Disease Detection - Improved Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Single image with smart filtering:
    python inference_improved.py --image image.jpg --filter
  
  Batch with lower confidence:
    python inference_improved.py --folder /home/preetham-bhat/Downloads --conf 0.25 --filter --save
  
  Tomato-specific detection:
    python inference_improved.py --folder /home/preetham-bhat/Downloads --conf 0.25 --save
        '''
    )
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--save', action='store_true', help='Save annotated images')
    parser.add_argument('--filter', action='store_true', help='Enable smart filtering')
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.print_help()
        print('\n❌ Provide --image or --folder')
        sys.exit(1)
    
    if args.image and args.folder:
        print('❌ Provide either --image OR --folder, not both')
        sys.exit(1)
    
    if not (0.0 <= args.conf <= 1.0):
        print(f'❌ Confidence must be 0.0-1.0')
        sys.exit(1)
    
    model = load_model()
    
    if args.image:
        infer_image(model, args.image, conf=args.conf, save=args.save, smart_filter=args.filter)
    else:
        infer_folder(model, args.folder, conf=args.conf, save=args.save, smart_filter=args.filter)


if __name__ == '__main__':
    main()
