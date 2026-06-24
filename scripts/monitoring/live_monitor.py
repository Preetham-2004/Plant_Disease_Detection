#!/usr/bin/env python3
"""
Live Training Monitor - Real-time YOLOv8 Training Progress Display
Shows epoch progress as training happens
"""

import pandas as pd
import os
import time
import sys
from pathlib import Path
from datetime import datetime

def get_color(value, min_val, max_val, reverse=False):
    """Get ANSI color based on value (green=good, red=bad)"""
    if reverse:
        ratio = (max_val - value) / (max_val - min_val) if (max_val - min_val) != 0 else 0
    else:
        ratio = (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
    
    if ratio >= 0.7:
        return '\033[92m'  # Green
    elif ratio >= 0.4:
        return '\033[93m'  # Yellow
    else:
        return '\033[91m'  # Red

RESET = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'

def print_header():
    os.system('clear' if os.name == 'posix' else 'cls')
    print(f"{CYAN}{'='*140}{RESET}")
    print(f"{CYAN}{'='*140}{RESET}")
    print(f"{CYAN}  🎯 YOLOv8 LIVE TRAINING MONITOR - Real-time Progress{RESET}")
    print(f"{CYAN}{'='*140}{RESET}")
    print(f"{CYAN}{'='*140}{RESET}")

def monitor_training():
    results_file = Path("runs/detect/train/results.csv")
    last_epoch = 0
    
    print(f"{YELLOW}Waiting for training to start...{RESET}\n")
    
    while True:
        try:
            if not results_file.exists():
                print(f"{RED}❌ Training results file not found{RESET}")
                time.sleep(5)
                continue
            
            df = pd.read_csv(results_file)
            current_epoch = len(df)
            
            if current_epoch > last_epoch:
                print_header()
                
                # Overall progress
                progress_pct = min(current_epoch, 100)
                progress_bar = '█' * (progress_pct // 5) + '░' * ((100 - progress_pct) // 5)
                print(f"\n{BOLD}📊 OVERALL PROGRESS{RESET}")
                print(f"  Epochs: {current_epoch}/100 | {progress_bar} {progress_pct}%")
                print(f"  Time/Epoch: {df['time'].mean():.1f}s")
                print(f"  Total Time: {df['time'].sum()/3600:.2f}h")
                print(f"  Est. Remaining: {(100-current_epoch) * df['time'].mean() / 3600:.2f}h")
                
                # Best metrics
                best_map50_idx = df['metrics/mAP50(B)'].idxmax()
                best_map50 = df.loc[best_map50_idx, 'metrics/mAP50(B)']
                print(f"\n{BOLD}⭐ BEST METRICS{RESET}")
                print(f"  Best mAP50: {GREEN}{best_map50:.4f}{RESET} (Epoch {int(df.loc[best_map50_idx, 'epoch'])})")
                print(f"  Best mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")
                print(f"  Best Precision: {df['metrics/precision(B)'].max():.4f}")
                print(f"  Best Recall: {df['metrics/recall(B)'].max():.4f}")
                
                # Latest epochs (last 10)
                print(f"\n{BOLD}📈 RECENT EPOCHS (Last 10){RESET}")
                print(f"{BOLD}Epoch │ Time  │ Train Loss │ Val Loss │ mAP50  │ Precision │ Recall │ mAP50-95 │ Status{RESET}")
                print("─" * 140)
                
                start_idx = max(0, len(df) - 10)
                for idx in range(start_idx, len(df)):
                    row = df.iloc[idx]
                    epoch = int(row['epoch'])
                    time_s = row['time']
                    train_loss = row['train/box_loss']
                    val_loss = row['val/box_loss']
                    map50 = row['metrics/mAP50(B)']
                    precision = row['metrics/precision(B)']
                    recall = row['metrics/recall(B)']
                    map50_95 = row['metrics/mAP50-95(B)']
                    
                    # Color based on performance
                    loss_color = get_color(train_loss, df['train/box_loss'].max(), df['train/box_loss'].min(), reverse=True)
                    map_color = get_color(map50, df['metrics/mAP50(B)'].min(), df['metrics/mAP50(B)'].max())
                    
                    # Status indicator
                    if epoch == best_map50_idx + 1:
                        status = f"{GREEN}⭐ BEST{RESET}"
                    elif idx == len(df) - 1:
                        status = f"{CYAN}🔄 CURRENT{RESET}"
                    else:
                        status = "✓"
                    
                    print(f"{epoch:5} │ {time_s:5.0f} │ {loss_color}{train_loss:10.4f}{RESET} │ {val_loss:8.4f} │ {map_color}{map50:6.4f}{RESET} │ {precision:9.4f} │ {recall:6.4f} │ {map50_95:8.4f} │ {status}")
                
                # Current stats
                latest = df.iloc[-1]
                print(f"\n{BOLD}📌 CURRENT EPOCH #{int(latest['epoch'])}{RESET}")
                print(f"  Training Loss: {latest['train/box_loss']:.4f} (Box) | {latest['train/cls_loss']:.4f} (Class) | {latest['train/dfl_loss']:.4f} (DFL)")
                print(f"  Validation Loss: {latest['val/box_loss']:.4f} (Box) | {latest['val/cls_loss']:.4f} (Class) | {latest['val/dfl_loss']:.4f} (DFL)")
                print(f"  Validation Metrics:")
                print(f"    • mAP50: {GREEN}{latest['metrics/mAP50(B)']:.4f}{RESET}")
                print(f"    • mAP50-95: {latest['metrics/mAP50-95(B)']:.4f}")
                print(f"    • Precision: {latest['metrics/precision(B)']:.4f}")
                print(f"    • Recall: {latest['metrics/recall(B)']:.4f}")
                print(f"    • Learning Rate: {latest['lr/pg0']:.6f}")
                
                # Loss trend
                if len(df) > 1:
                    loss_trend = latest['train/box_loss'] - df.iloc[-2]['train/box_loss']
                    map_trend = latest['metrics/mAP50(B)'] - df.iloc[-2]['metrics/mAP50(B)']
                    
                    loss_trend_color = RED if loss_trend > 0 else GREEN
                    map_trend_color = GREEN if map_trend > 0 else RED
                    
                    print(f"\n{BOLD}📉 TRENDS (vs Previous Epoch){RESET}")
                    print(f"  Loss Change: {loss_trend_color}{loss_trend:+.4f}{RESET}")
                    print(f"  mAP50 Change: {map_trend_color}{map_trend:+.4f}{RESET}")
                
                # Health check
                print(f"\n{BOLD}🏥 SYSTEM HEALTH{RESET}")
                if latest['train/box_loss'] > 2.0:
                    print(f"  {RED}⚠️ High training loss - may need adjustment{RESET}")
                else:
                    print(f"  {GREEN}✅ Training loss normal{RESET}")
                
                if latest['val/box_loss'] > latest['train/box_loss'] * 1.5:
                    print(f"  {YELLOW}⚠️ High validation loss - possible overfitting{RESET}")
                else:
                    print(f"  {GREEN}✅ Validation loss healthy{RESET}")
                
                print(f"\n{YELLOW}Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
                print(f"{YELLOW}(Press Ctrl+C to stop monitoring){RESET}\n")
                
                last_epoch = current_epoch
            
            time.sleep(2)  # Update every 2 seconds
            
        except KeyboardInterrupt:
            print(f"\n\n{GREEN}✅ Monitoring stopped{RESET}")
            break
        except Exception as e:
            print(f"{RED}❌ Error: {e}{RESET}")
            time.sleep(5)

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print(f"\n{GREEN}✅ Monitor closed{RESET}")
        sys.exit(0)
