#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for DART-X vs. Dynse Baseline
Generates:
  1. results/figures/accuracy_over_time.png
  2. results/figures/per_location_accuracy.png
  3. results/figures/confusion_matrix.png
  4. results/figures/ablation_study.png
  5. results/figures/synthetic_drifts.png
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configurations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def read_csv_columns(filepath):
    cols = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in cols:
                    cols[k] = []
                try:
                    cols[k].append(float(v))
                except ValueError:
                    cols[k].append(v)
    for k in cols:
        cols[k] = np.array(cols[k])
    return cols

def rolling_mean(arr, window=3):
    result = np.zeros_like(arr, dtype=float)
    n = len(arr)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        result[i] = np.mean(arr[start:end])
    return result

def plot_accuracy_over_time():
    comp = read_csv_columns('results/comparison/pklot_comparison_daily.csv')
    
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    
    days = comp['day']
    dynse_acc = comp['dynse_accuracy']
    dartx_acc = comp['dartx_accuracy']
    
    # Raw traces with light alpha
    ax.plot(days, dynse_acc, color='#1f77b4', alpha=0.35, linewidth=1.2, linestyle=':')
    ax.plot(days, dartx_acc, color='#d62728', alpha=0.35, linewidth=1.2, linestyle=':')
    
    # Rolling 3-day smoothing for clear trend
    roll_dynse = rolling_mean(dynse_acc, window=3)
    roll_dartx = rolling_mean(dartx_acc, window=3)
    
    ax.plot(days, roll_dynse, color='#1f77b4', linewidth=2.4, label='Original Dynse Baseline (Mean: {:.2f}%)'.format(np.mean(dynse_acc)))
    ax.plot(days, roll_dartx, color='#d62728', linewidth=2.6, label='DART-X Proposed Framework (Mean: {:.2f}%)'.format(np.mean(dartx_acc)))
    
    # Vertical phase transition boundaries
    ax.axvline(x=22.5, color='#444444', linestyle='--', linewidth=1.8, alpha=0.8)
    ax.axvline(x=50.5, color='#444444', linestyle='--', linewidth=1.8, alpha=0.8)
    
    # Shaded location backgrounds
    ax.axvspan(1, 22.5, color='#f0f4f8', alpha=0.7, label='UFPR04 (Camera Setup 1)')
    ax.axvspan(22.5, 50.5, color='#fff9e6', alpha=0.7, label='UFPR05 (Sensor Drift: Camera Setup 2)')
    ax.axvspan(50.5, 82, color='#eef9f0', alpha=0.7, label='PUCPR (Environment Drift: New Campus)')
    
    # Text annotations for concept transitions
    ax.text(11, 62, 'UFPR04\n(Days 1–22)', ha='center', va='center', fontsize=11, fontweight='bold', color='#2b4c6f')
    ax.text(36.5, 62, 'UFPR05\n(Days 23–50)\n[Sensor Drift]', ha='center', va='center', fontsize=11, fontweight='bold', color='#855a00')
    ax.text(66, 62, 'PUCPR\n(Days 51–82)\n[Environment Drift]', ha='center', va='center', fontsize=11, fontweight='bold', color='#1e5622')
    
    ax.set_title('PKLot 82-Day Chronological Benchmark: Dynse Baseline vs. DART-X Framework', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Time Step / Chronological Day (Batch)', fontsize=12, labelpad=8)
    ax.set_ylabel('Test Classification Accuracy (%)', fontsize=12, labelpad=8)
    ax.set_xlim(1, 82)
    ax.set_ylim(60, 102)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#dddddd', fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig('results/figures/accuracy_over_time.png', dpi=300)
    plt.close()
    print('Generated results/figures/accuracy_over_time.png')

def plot_per_location_accuracy():
    comp = read_csv_columns('results/comparison/pklot_comparison_daily.csv')
    
    locs = ['UFPR04 (Days 1–22)', 'UFPR05 (Days 23–50)', 'PUCPR (Days 51–82)', 'Overall Benchmark']
    dynse_sub = [
        np.mean(comp['dynse_accuracy'][0:22]),
        np.mean(comp['dynse_accuracy'][22:50]),
        np.mean(comp['dynse_accuracy'][50:82]),
        np.mean(comp['dynse_accuracy'])
    ]
    dartx_sub = [
        np.mean(comp['dartx_accuracy'][0:22]),
        np.mean(comp['dartx_accuracy'][22:50]),
        np.mean(comp['dartx_accuracy'][50:82]),
        np.mean(comp['dartx_accuracy'])
    ]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(locs))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, dynse_sub, width, label='Original Dynse Baseline', color='#4575b4', edgecolor='#2c4d75', linewidth=1)
    rects2 = ax.bar(x + width/2, dartx_sub, width, label='DART-X Proposed Framework', color='#d73027', edgecolor='#8c1d17', linewidth=1)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1d3557')
        
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#b7094c')
        
    ax.set_title('Per-Location Classification Accuracy: Dynse vs. DART-X', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Mean Accuracy (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(locs, fontsize=10.5)
    ax.set_ylim(70, 102)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/figures/per_location_accuracy.png', dpi=300)
    plt.close()
    print('Generated results/figures/per_location_accuracy.png')

def plot_confusion_matrices():
    dynse_df = read_csv_columns('results/baseline/pklot_dynse_daily.csv')
    dartx_df = read_csv_columns('results/dartx/pklot_dartx_daily.csv')
    
    cm_dynse = np.array([
        [int(np.sum(dynse_df['tn'])), int(np.sum(dynse_df['fp']))],
        [int(np.sum(dynse_df['fn'])), int(np.sum(dynse_df['tp']))]
    ])
    cm_dartx = np.array([
        [int(np.sum(dartx_df['tn'])), int(np.sum(dartx_df['fp']))],
        [int(np.sum(dartx_df['fn'])), int(np.sum(dartx_df['tp']))]
    ])
    
    cm_dynse_norm = cm_dynse.astype(float) / cm_dynse.sum(axis=1, keepdims=True) * 100.0
    cm_dartx_norm = cm_dartx.astype(float) / cm_dartx.sum(axis=1, keepdims=True) * 100.0
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    
    for ax, cm, norm, title in zip(axes, [cm_dynse, cm_dartx], [cm_dynse_norm, cm_dartx_norm],
                                  ['Original Dynse Baseline', 'DART-X Framework']):
        im = ax.imshow(norm, cmap='Blues', vmin=75, vmax=100)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Empty (-1)', 'Occupied (+1)'], fontsize=10)
        ax.set_yticklabels(['Empty (-1)', 'Occupied (+1)'], fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=11, labelpad=6)
        ax.set_ylabel('True Label', fontsize=11, labelpad=6)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        for r in range(2):
            for c in range(2):
                raw = cm[r, c]
                pct = norm[r, c]
                color = 'white' if pct > 88 else '#222222'
                ax.text(c, r, f'{pct:.2f}%\n({raw:,})', ha='center', va='center',
                        fontsize=10.5, fontweight='bold', color=color)
                
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label='Class Accuracy (%)')
    plt.savefig('results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated results/figures/confusion_matrix.png')

def plot_ablation_study():
    abl_df = read_csv_columns('results/comparison/pklot_ablation_daily.csv')
    
    configs = [
        'Complete DART-X',
        'No Adaptive Window (Static M=4)',
        'No Hysteresis (h=0)',
        'No Reliability Scoring (Binary Hits)'
    ]
    means = [
        np.mean(abl_df['full_dartx']),
        np.mean(abl_df['no_adaptive_window']),
        np.mean(abl_df['no_hysteresis']),
        np.mean(abl_df['no_reliability'])
    ]
    stds = [
        np.std(abl_df['full_dartx']),
        np.std(abl_df['no_adaptive_window']),
        np.std(abl_df['no_hysteresis']),
        np.std(abl_df['no_reliability'])
    ]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    y_pos = np.arange(len(configs))
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
    
    bars = ax.barh(y_pos, means, xerr=stds, align='center', color=colors, alpha=0.85,
                   edgecolor='#333333', linewidth=1, capsize=5)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height()/2, f'{w:.2f}%',
                ha='left', va='center', fontsize=10, fontweight='bold', color='#111111')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Classification Accuracy across 82 Days (%)', fontsize=11, labelpad=8)
    ax.set_title('DART-X Component Ablation Study on PKLot Benchmark', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(85, 96)
    
    plt.tight_layout()
    plt.savefig('results/figures/ablation_study.png', dpi=300)
    plt.close()
    print('Generated results/figures/ablation_study.png')

def plot_synthetic_drifts():
    if not os.path.exists('results/comparison/synthetic_experiments.json'):
        return
    with open('results/comparison/synthetic_experiments.json', 'r') as f:
        data = json.load(f)
        
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    
    # (a) Abrupt Drift
    ax = axes[0, 0]
    ax.plot(data['abrupt']['steps'], data['abrupt']['dynse'], label='Dynse Baseline', color='#1f77b4', linewidth=1.8)
    ax.plot(data['abrupt']['steps'], data['abrupt']['dartx'], label='DART-X', color='#d62728', linewidth=2.0)
    ax.axvline(x=30, color='#666666', linestyle='--', alpha=0.7)
    ax.axvline(x=65, color='#666666', linestyle='--', alpha=0.7)
    ax.text(30.5, 50, 'Abrupt Drift 1', fontsize=9, color='#444444')
    ax.text(65.5, 50, 'Abrupt Drift 2', fontsize=9, color='#444444')
    ax.set_title('(a) Abrupt Concept Drift Resilience', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 102)
    ax.legend(loc='lower left', fontsize=9)
    
    # (b) Gradual Drift
    ax = axes[0, 1]
    ax.plot(data['gradual']['steps'], data['gradual']['dynse'], label='Dynse Baseline', color='#1f77b4', linewidth=1.8)
    ax.plot(data['gradual']['steps'], data['gradual']['dartx'], label='DART-X', color='#d62728', linewidth=2.0)
    ax.axvspan(30, 60, color='#ffffcc', alpha=0.6, label='Gradual Transition Window')
    ax.set_title('(b) Gradual Concept Drift Transition', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 102)
    ax.legend(loc='lower left', fontsize=9)
    
    # (c) Noise Stress Test
    ax = axes[1, 0]
    noise_pct = [n * 100 for n in data['noise']['levels']]
    ax.plot(noise_pct, data['noise']['dynse'], marker='o', label='Dynse Baseline', color='#1f77b4', linewidth=2)
    ax.plot(noise_pct, data['noise']['dartx'], marker='s', label='DART-X', color='#d62728', linewidth=2)
    ax.set_title('(c) Robustness to Increasing Label Noise', fontsize=11, fontweight='bold')
    ax.set_xlabel('Injected Label Noise (%)')
    ax.set_ylabel('Mean Accuracy (%)')
    ax.set_ylim(60, 95)
    ax.legend(loc='lower left', fontsize=9)
    
    # (d) Imbalance Stress Test
    ax = axes[1, 1]
    ratios = [r * 100 for r in data['imbalance']['ratios']]
    ax.plot(ratios, data['imbalance']['dynse'], marker='o', label='Dynse Baseline', color='#1f77b4', linewidth=2)
    ax.plot(ratios, data['imbalance']['dartx'], marker='s', label='DART-X', color='#d62728', linewidth=2)
    ax.set_title('(d) Stress Test: Severe Class Imbalance', fontsize=11, fontweight='bold')
    ax.set_xlabel('Minority Class Proportion (%)')
    ax.set_ylabel('Macro F1-Score (%)')
    ax.set_ylim(60, 95)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/synthetic_drifts.png', dpi=300)
    plt.close()
    print('Generated results/figures/synthetic_drifts.png')

if __name__ == '__main__':
    print('Rendering publication figures...')
    os.makedirs('results/figures', exist_ok=True)
    if os.path.exists('results/comparison/pklot_comparison_daily.csv'):
        plot_accuracy_over_time()
        plot_per_location_accuracy()
        plot_confusion_matrices()
        plot_ablation_study()
    plot_synthetic_drifts()
    print('All figures rendered successfully!')
