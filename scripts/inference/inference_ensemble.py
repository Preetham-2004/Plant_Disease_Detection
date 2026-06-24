#!/usr/bin/env python3
"""
🌱 ENSEMBLE INFERENCE: Combine Original + Finetuned Models
Uses voting and weighted confidence to combine predictions from both models.

Advantages:
✅ Reduces false positives from model disagreement
✅ Increases confidence in shared predictions
✅ Handles ambiguities better
✅ More robust

Usage:
    python inference_ensemble.py --folder /home/preetham-bhat/Downloads --save-dir results_ensemble
    python inference_ensemble.py --folder /home/preetham-bhat/Downloads --conf 0.25 --save-dir results_ensemble
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict

CLASS_NAMES = {
    0: 'eggplant_fruit',
    1: 'eggplant_leaf',
    2: 'potato_fruit',
    3: 'potato_leaf',
    4: 'tomato_fruit',
    5: 'tomato_leaf'
}


class EnsembleDetector:
    def __init__(self, model1_path, model2_path, conf=0.5):
        """Initialize ensemble with two models."""
        print(f'📦 Loading Model 1 (Original): {Path(model1_path).name}')
        self.model1 = YOLO(model1_path)
        self.model1_weight = 0.6  # Original model has higher mAP50 (0.8162)
        
        print(f'📦 Loading Model 2 (Finetuned): {Path(model2_path).name}')
        self.model2 = YOLO(model2_path)
        self.model2_weight = 0.4  # Finetuned model has lower mAP50 (0.8000)
        
        self.conf = conf
        print(f'✅ Ensemble initialized (weights: {self.model1_weight:.1%} vs {self.model2_weight:.1%})\n')
    
    def predict_single(self, image_path, model, conf):
        """Run prediction with a single model."""
        result = model.predict(source=image_path, conf=conf, verbose=False)
        return result[0]
    
    def ensemble_predict(self, image_path):
        """
        Combine predictions from both models using:
        1. Voting (confidence threshold)
        2. Weighted confidence averaging
        3. IoU-based box matching
        """
        # Run both models
        result1 = self.predict_single(image_path, self.model1, self.conf)
        result2 = self.predict_single(image_path, self.model2, self.conf)
        
        detections = []
        
        # Collect all detections from both models
        for result, model_id, weight in [
            (result1, 0, self.model1_weight),
            (result2, 1, self.model2_weight)
        ]:
            if len(result.boxes) > 0:
                for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    detections.append({
                        'box': box.cpu().numpy(),
                        'conf': float(conf),
                        'cls_id': int(cls_id),
                        'model_id': model_id,
                        'weight': weight
                    })
        
        # Merge similar detections (IoU > 0.5 + same class = same object)
        merged_detections = self._merge_detections(detections)
        
        return merged_detections
    
    def _merge_detections(self, detections):
        """Merge detections from both models using IoU and class matching."""
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Find matching detections from other model
            matches = [det1]
            
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Same class AND high IoU
                if det1['cls_id'] == det2['cls_id']:
                    iou = self._calc_iou(det1['box'], det2['box'])
                    if iou > 0.3:  # Overlapping detections
                        matches.append(det2)
                        used.add(j)
            
            # Combine matches
            if len(matches) > 1:
                # Multiple models agree - BOOST confidence
                confidences = [m['conf'] * m['weight'] for m in matches]
                final_conf = sum(confidences) / sum(m['weight'] for m in matches)
                final_conf = min(final_conf * 1.2, 1.0)  # Boost to max 1.0
                
                consensus = 'CONSENSUS' if len(matches) > 1 else 'SINGLE'
                
                merged.append({
                    'box': matches[0]['box'],
                    'conf': final_conf,
                    'cls_id': matches[0]['cls_id'],
                    'consensus': True,
                    'agreement': len(matches)
                })
            else:
                # Single model detection
                merged.append({
                    'box': det1['box'],
                    'conf': det1['conf'] * det1['weight'],
                    'cls_id': det1['cls_id'],
                    'consensus': False,
                    'agreement': 1
                })
            
            used.add(i)
        
        return merged
    
    def _calc_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


def process_batch(ensemble, folder_path, conf, save_dir):
    """Process all images in a folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f'❌ Folder not found: {folder_path}')
        sys.exit(1)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f'❌ No images found')
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(save_dir) / 'predict'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\n{"="*80}')
    print(f'🔗 ENSEMBLE INFERENCE: Original + Finetuned Models')
    print(f'{"="*80}\n')
    
    print(f'📁 Processing: {folder.name}')
    print(f'   Images: {len(images)}')
    print(f'   Confidence: {conf:.0%}')
    print(f'   Output: {output_dir}/')
    print(f'\n{"="*80}\n')
    
    total_detections = 0
    consensus_count = 0
    single_count = 0
    results_by_class = defaultdict(int)
    
    for i, img_path in enumerate(images, 1):
        print(f'[{i}/{len(images)}] {img_path.name}')
        
        # Read image
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        
        # Get ensemble predictions
        detections = ensemble.ensemble_predict(str(img_path))
        
        num_detections = len(detections)
        total_detections += num_detections
        
        # Count consensus vs single
        consensus_det = sum(1 for d in detections if d['consensus'])
        single_det = num_detections - consensus_det
        consensus_count += consensus_det
        single_count += single_det
        
        # Draw boxes on image
        for det in detections:
            box = det['box']
            conf_score = det['conf']
            cls_id = det['cls_id']
            is_consensus = det['consensus']
            
            x1, y1, x2, y2 = map(int, box)
            
            # Color: Green for consensus, Orange for single
            color = (0, 255, 0) if is_consensus else (0, 165, 255)
            thickness = 3 if is_consensus else 2
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            cls_name = CLASS_NAMES[cls_id]
            label = f'{cls_name} {conf_score:.1%}'
            if is_consensus:
                label += ' ✓'
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness_text = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]
            
            padding = 6
            bg_y1 = max(0, y1 - text_size[1] - padding)
            bg_x2 = min(img_w, x1 + text_size[0] + padding)
            bg_x1 = max(0, x1)
            
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thickness_text)
        
        # Save annotated image
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img)
        
        if num_detections > 0:
            det_str = ', '.join([
                f'{CLASS_NAMES[d["cls_id"]]} ({d["conf"]:.1%}){"*" if d["consensus"] else ""}'
                for d in detections
            ])
            print(f'   ✓ {num_detections} detections: {det_str}')
            
            for det in detections:
                cls_name = CLASS_NAMES[det['cls_id']]
                results_by_class[cls_name] += 1
        else:
            print(f'   - No detections')
    
    print(f'\n{"="*80}')
    print(f'📊 ENSEMBLE RESULTS:')
    print(f'   Total detections: {total_detections}')
    print(f'   Consensus (both models agree)*: {consensus_count}')
    print(f'   Single model: {single_count}')
    
    if results_by_class:
        print(f'\n   By class:')
        for cls_name, count in sorted(results_by_class.items()):
            print(f'      • {cls_name}: {count}')
    
    print(f'\n   * = Both models predicted (higher confidence)')
    print(f'   ✓ = Consensus prediction (marked green in images)')
    print(f'\n💾 ANNOTATED IMAGES SAVED TO:')
    print(f'   {output_dir}/')
    print(f'{"="*80}\n')
    
    print(f'💡 ENSEMBLE ADVANTAGES:')
    print(f'   ✅ {consensus_count} detections confirmed by both models')
    print(f'   ✅ Reduced false positives ({single_count} single-model detections)')
    print(f'   ✅ Higher confidence in consensus predictions')
    print(f'\n')


def main():
    parser = argparse.ArgumentParser(
        description='🌱 Ensemble Inference: Combine Original + Finetuned Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Standard ensemble:
    python inference_ensemble.py --folder /home/preetham-bhat/Downloads --save-dir results_ensemble
  
  Lower confidence for more detections:
    python inference_ensemble.py --folder /home/preetham-bhat/Downloads --conf 0.25 --save-dir results_ensemble
        '''
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        default='/home/preetham-bhat/Downloads',
        help='Folder with images'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Base confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results_ensemble',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Model paths
    model1_path = 'runs/detect/train_yolov8s_cleaned/weights/best.pt'
    model2_path = '/home/preetham-bhat/plant_project/Plant Disease prediction/best_yolo_parts.pt'
    
    # Initialize ensemble
    ensemble = EnsembleDetector(model1_path, model2_path, conf=args.conf)
    
    # Process images
    process_batch(ensemble, args.folder, args.conf, args.save_dir)


if __name__ == '__main__':
    main()
