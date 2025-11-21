"""
Generate Comparison Plots: RandomForest vs LSTM Models
Creates visualizations comparing model performance, accuracy, and characteristics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import argparse
import pickle
import re
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_model_results(comparison_dir='./model_comparison'):
    """Load model comparison results from report or models"""
    report_path = os.path.join(comparison_dir, 'model_comparison_report.txt')
    
    results = {
        'rf_accuracy': None,
        'lstm_single_accuracy': None,
        'lstm_cv_mean': None,
        'lstm_cv_std': None,
        'lstm_cv_scores': [],
        'rf_test_samples': None,
        'lstm_test_samples': None,
        'rf_train_samples': None,
        'lstm_train_samples': None,
        'data_samples': None,
        'lod_distribution': {}
    }
    
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
            
            # Parse RandomForest accuracy
            rf_match = re.search(r'RANDOM FOREST.*?Accuracy: ([\d.]+)', content, re.DOTALL)
            if rf_match:
                results['rf_accuracy'] = float(rf_match.group(1))
            
            # Parse RandomForest samples
            rf_train_match = re.search(r'Training samples: (\d+)', content)
            rf_test_match = re.search(r'Test samples: (\d+)', content)
            if rf_train_match:
                # Find the first occurrence (RandomForest)
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'RANDOM FOREST' in line:
                        # Look for Training samples in next few lines
                        for j in range(i, min(i+10, len(lines))):
                            if 'Training samples:' in lines[j]:
                                match = re.search(r'Training samples: (\d+)', lines[j])
                                if match:
                                    results['rf_train_samples'] = int(match.group(1))
                            if 'Test samples:' in lines[j]:
                                match = re.search(r'Test samples: (\d+)', lines[j])
                                if match:
                                    results['rf_test_samples'] = int(match.group(1))
                                    break
                        break
            
            # Parse LSTM single split accuracy
            lstm_match = re.search(r'Single Split Accuracy: ([\d.]+)', content)
            if lstm_match:
                results['lstm_single_accuracy'] = float(lstm_match.group(1))
            
            # Parse LSTM CV results
            cv_mean_match = re.search(r'Mean Accuracy: ([\d.]+)', content)
            cv_std_match = re.search(r'Std Deviation: ([\d.]+)', content)
            if cv_mean_match:
                results['lstm_cv_mean'] = float(cv_mean_match.group(1))
            if cv_std_match:
                results['lstm_cv_std'] = float(cv_std_match.group(1))
            
            # Parse LSTM samples
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'LSTM RESULTS' in line:
                    # Look for samples in next few lines
                    for j in range(i, min(i+10, len(lines))):
                        if 'Training samples:' in lines[j]:
                            match = re.search(r'Training samples: (\d+)', lines[j])
                            if match:
                                results['lstm_train_samples'] = int(match.group(1))
                        if 'Test samples:' in lines[j]:
                            match = re.search(r'Test samples: (\d+)', lines[j])
                            if match:
                                results['lstm_test_samples'] = int(match.group(1))
                                break
                    break
            
            # Parse CV fold details
            fold_matches = re.findall(r'Fold \d+: ([\d.]+)', content)
            if fold_matches:
                results['lstm_cv_scores'] = [float(x) for x in fold_matches]
            
            # Parse data samples
            samples_match = re.search(r'Total samples: (\d+)', content)
            if samples_match:
                results['data_samples'] = int(samples_match.group(1))
    
    return results

def plot_accuracy_comparison(results, output_dir):
    """Plot accuracy comparison between models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Single Split Accuracy
    ax1 = axes[0]
    models = ['RandomForest', 'LSTM']
    accuracies = [results['rf_accuracy'], results['lstm_single_accuracy']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        if acc is not None:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Single Split Accuracy Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: LSTM Cross-Validation Results
    ax2 = axes[1]
    if results['lstm_cv_scores']:
        folds = [f'Fold {i+1}' for i in range(len(results['lstm_cv_scores']))]
        bars2 = ax2.bar(folds, results['lstm_cv_scores'], 
                       color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add mean line
        if results['lstm_cv_mean']:
            ax2.axhline(y=results['lstm_cv_mean'], color='green', 
                       linestyle='--', linewidth=2, label=f'Mean: {results["lstm_cv_mean"]:.3f}')
            if results['lstm_cv_std']:
                ax2.axhline(y=results['lstm_cv_mean'] + results['lstm_cv_std'], 
                           color='green', linestyle=':', linewidth=1, alpha=0.5, label='±1 Std')
                ax2.axhline(y=results['lstm_cv_mean'] - results['lstm_cv_std'], 
                           color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        # Add value labels
        for bar, score in zip(bars2, results['lstm_cv_scores']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('LSTM Cross-Validation Results', fontweight='bold', fontsize=14)
        ax2.set_ylim([0, 1.1])
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '1_accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def plot_cv_comparison(results, output_dir):
    """Plot cross-validation comparison"""
    if not results['lstm_cv_scores']:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # RandomForest accuracy (single value, shown as horizontal line)
    if results['rf_accuracy']:
        ax.axhline(y=results['rf_accuracy'], color='#3498db', 
                  linestyle='-', linewidth=3, label=f'RandomForest: {results["rf_accuracy"]:.3f}')
    
    # LSTM CV scores with error bars
    folds = np.arange(1, len(results['lstm_cv_scores']) + 1)
    ax.plot(folds, results['lstm_cv_scores'], 'o-', color='#e74c3c', 
           linewidth=2, markersize=10, label='LSTM (CV Folds)', alpha=0.8)
    
    # Mean and std
    if results['lstm_cv_mean']:
        ax.axhline(y=results['lstm_cv_mean'], color='#e74c3c', 
                  linestyle='--', linewidth=2, alpha=0.7, 
                  label=f'LSTM Mean: {results["lstm_cv_mean"]:.3f}')
        if results['lstm_cv_std']:
            ax.fill_between(folds, 
                           results['lstm_cv_mean'] - results['lstm_cv_std'],
                           results['lstm_cv_mean'] + results['lstm_cv_std'],
                           color='#e74c3c', alpha=0.2, label=f'±1 Std: {results["lstm_cv_std"]:.3f}')
    
    ax.set_xlabel('Cross-Validation Fold', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Model Accuracy: RandomForest vs LSTM (Cross-Validation)', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(folds)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '2_cv_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def plot_model_characteristics(results, output_dir):
    """Plot model characteristics comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Samples
    ax1 = axes[0, 0]
    if results.get('rf_train_samples') and results.get('lstm_train_samples'):
        models = ['RandomForest', 'LSTM']
        train_samples = [
            results['rf_train_samples'],
            results['lstm_train_samples']
        ]
        test_samples = [
            results.get('rf_test_samples', 0),
            results.get('lstm_test_samples', 0)
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, train_samples, width, label='Training', 
               color='#3498db', alpha=0.7, edgecolor='black')
        ax1.bar(x + width/2, test_samples, width, label='Test', 
               color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax1.set_ylabel('Number of Samples', fontweight='bold')
        ax1.set_title('Training/Test Split', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'Training/Test split data\nnot available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Training/Test Split', fontweight='bold')
    
    # Plot 2: Accuracy Range
    ax2 = axes[0, 1]
    if results['lstm_cv_scores'] and results['rf_accuracy']:
        rf_acc = results['rf_accuracy']
        lstm_min = min(results['lstm_cv_scores'])
        lstm_max = max(results['lstm_cv_scores'])
        lstm_mean = results['lstm_cv_mean']
        
        models = ['RandomForest', 'LSTM\n(Min)', 'LSTM\n(Mean)', 'LSTM\n(Max)']
        accs = [rf_acc, lstm_min, lstm_mean, lstm_max]
        colors = ['#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
        
        bars = ax2.bar(models, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, acc in zip(bars, accs):
            if acc is not None:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.3f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('Accuracy Range Comparison', fontweight='bold')
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Model Stability (CV Std Dev)
    ax3 = axes[1, 0]
    if results['lstm_cv_std']:
        models = ['RandomForest\n(Single Split)', 'LSTM\n(CV Std Dev)']
        values = [0, results['lstm_cv_std']]  # RF has no CV, so 0
        colors = ['#3498db', '#e74c3c']
        
        bars = ax3.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax3.set_ylabel('Standard Deviation', fontweight='bold')
        ax3.set_title('Model Stability (Lower is Better)', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Data Overview
    ax4 = axes[1, 1]
    info_text = f"""Dataset Information:

Total Samples: {results.get('data_samples', 'N/A')}

RandomForest:
• Accuracy: {results.get('rf_accuracy', 0):.3f}
• Train: {results.get('rf_train_samples', 'N/A')}
• Test: {results.get('rf_test_samples', 'N/A')}

LSTM:
• Single Split: {results.get('lstm_single_accuracy', 0):.3f}
• CV Mean: {results.get('lstm_cv_mean', 0):.3f}
• CV Std: {results.get('lstm_cv_std', 0):.3f}
• Train: {results.get('lstm_train_samples', 'N/A')}
• Test: {results.get('lstm_test_samples', 'N/A')}
"""
    ax4.text(0.1, 0.5, info_text, fontsize=10, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Summary Statistics', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3_model_characteristics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def plot_accuracy_distribution(results, output_dir):
    """Plot accuracy distribution for LSTM CV"""
    if not results['lstm_cv_scores']:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of CV scores
    ax.hist(results['lstm_cv_scores'], bins=10, color='#e74c3c', 
           alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add mean line
    if results['lstm_cv_mean']:
        ax.axvline(x=results['lstm_cv_mean'], color='green', 
                  linestyle='--', linewidth=2, label=f'Mean: {results["lstm_cv_mean"]:.3f}')
        if results['lstm_cv_std']:
            ax.axvline(x=results['lstm_cv_mean'] + results['lstm_cv_std'], 
                      color='green', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=results['lstm_cv_mean'] - results['lstm_cv_std'], 
                      color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    # Add RandomForest accuracy
    if results['rf_accuracy']:
        ax.axvline(x=results['rf_accuracy'], color='#3498db', 
                  linestyle='-', linewidth=2, label=f'RandomForest: {results["rf_accuracy"]:.3f}')
    
    ax.set_xlabel('Accuracy', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('LSTM Cross-Validation Accuracy Distribution', fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '4_accuracy_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def plot_accuracy_boxplot(results, output_dir):
    """Create box-and-whisker plot comparing model accuracies"""
    if not results['lstm_cv_scores']:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    # LSTM: Use CV scores (5 folds)
    # RandomForest: Create a list with single value repeated (for comparison)
    data_to_plot = []
    labels = []
    colors = []
    
    # Add LSTM CV scores
    if results['lstm_cv_scores']:
        data_to_plot.append(results['lstm_cv_scores'])
        labels.append('LSTM\n(5-Fold CV)')
        colors.append('#e74c3c')
    
    # Add RandomForest (repeat single value to match LSTM length for visual comparison)
    if results['rf_accuracy']:
        rf_data = [results['rf_accuracy']] * len(results['lstm_cv_scores'])
        data_to_plot.append(rf_data)
        labels.append('RandomForest\n(Single Split)')
        colors.append('#3498db')
    
    if not data_to_plot:
        return
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True,
                    medianprops=dict(linewidth=2, color='black'),
                    meanprops=dict(linewidth=2, color='green', linestyle='--'))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style the whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('red')
        flier.set_markersize(8)
        flier.set_alpha(0.5)
    
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Model Accuracy Comparison: Box-and-Whisker Plot', 
                fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_lines = []
    if results['lstm_cv_mean'] and results['lstm_cv_std']:
        stats_lines.append(f"LSTM: Mean={results['lstm_cv_mean']:.3f}, Std={results['lstm_cv_std']:.3f}")
    if results['rf_accuracy']:
        stats_lines.append(f"RandomForest: {results['rf_accuracy']:.3f}")
    
    if stats_lines:
        stats_text = "\n".join(stats_lines)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '5_accuracy_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def plot_metrics_boxplot_from_csv(csv_path, output_dir):
    """Create box plots from CSV data comparing metrics between models"""
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV file not found: {csv_path}")
        print("   Skipping metrics box plots. Collect runtime data first.")
        return
    
    try:
        df = pd.read_csv(csv_path)
        
        # Metrics to plot
        metrics = {
            'deviceFPS': 'FPS (Frames Per Second)',
            'latencyMs': 'Latency (ms)',
            'deviceGPULoad': 'GPU Load (%)',
            'memoryMB': 'Memory Usage (MB)',
            'fpsAfterDecision': 'FPS After Decision',
            'deviceCPULoad': 'CPU Load'
        }
        
        # Filter available metrics
        available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
        
        if not available_metrics:
            print("⚠️  No plottable metrics found in CSV")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(available_metrics.items()):
            ax = axes[idx]
            
            # Get data (remove NaN values)
            data = pd.to_numeric(df[metric], errors='coerce').dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data for\n{label}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label, fontweight='bold')
                continue
            
            # Create box plot
            bp = ax.boxplot([data], labels=['PARIMA'], patch_artist=True,
                          widths=0.6, showmeans=True, meanline=True,
                          medianprops=dict(linewidth=2, color='black'),
                          meanprops=dict(linewidth=2, color='green', linestyle='--'))
            
            # Style
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][0].set_edgecolor('black')
            bp['boxes'][0].set_linewidth(1.5)
            
            # Style whiskers and caps
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1.5)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.5)
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markerfacecolor('red')
                flier.set_markersize(6)
                flier.set_alpha(0.5)
            
            ax.set_ylabel(label, fontweight='bold')
            ax.set_title(f'{label}\n(n={len(data)} samples)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            stats_text = f"Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '6_runtime_metrics_boxplot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
        
    except Exception as e:
        print(f"⚠️  Error creating metrics box plots: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_metrics_boxplot_comparison(parima_csv_path, lstm_csv_path, output_dir):
    """Create box plots comparing runtime metrics between PARIMA and LSTM"""
    if not os.path.exists(parima_csv_path):
        print(f"⚠️  PARIMA CSV file not found: {parima_csv_path}")
        return
    if not os.path.exists(lstm_csv_path):
        print(f"⚠️  LSTM CSV file not found: {lstm_csv_path}")
        return
    
    try:
        parima_df = pd.read_csv(parima_csv_path)
        lstm_df = pd.read_csv(lstm_csv_path)
        
        # Metrics to plot
        metrics = {
            'deviceFPS': 'FPS (Frames Per Second)',
            'latencyMs': 'Latency (ms)',
            'deviceGPULoad': 'GPU Load (%)',
            'memoryMB': 'Memory Usage (MB)',
            'fpsAfterDecision': 'FPS After Decision',
            'deviceCPULoad': 'CPU Load'
        }
        
        # Filter available metrics (must exist in both CSVs)
        available_metrics = {}
        for k, v in metrics.items():
            if k in parima_df.columns and k in lstm_df.columns:
                available_metrics[k] = v
        
        if not available_metrics:
            print("⚠️  No common plottable metrics found in both CSVs")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(available_metrics.items()):
            ax = axes[idx]
            
            # Get data from both models (remove NaN values)
            parima_data = pd.to_numeric(parima_df[metric], errors='coerce').dropna()
            lstm_data = pd.to_numeric(lstm_df[metric], errors='coerce').dropna()
            
            if len(parima_data) == 0 and len(lstm_data) == 0:
                ax.text(0.5, 0.5, f'No data for\n{label}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label, fontweight='bold')
                continue
            
            # Create box plot with both models
            data_to_plot = [parima_data, lstm_data]
            labels = ['PARIMA', 'LSTM']
            colors = ['#3498db', '#e74c3c']  # Blue for PARIMA, Red for LSTM
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                          widths=0.6, showmeans=True, meanline=True,
                          medianprops=dict(linewidth=2, color='black'),
                          meanprops=dict(linewidth=2, color='green', linestyle='--'))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
            
            # Style whiskers and caps
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1.5)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.5)
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markerfacecolor('red')
                flier.set_markersize(6)
                flier.set_alpha(0.5)
            
            ax.set_ylabel(label, fontweight='bold')
            ax.set_title(f'{label}\nPARIMA: n={len(parima_data)}, LSTM: n={len(lstm_data)}', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            stats_lines = []
            if len(parima_data) > 0:
                stats_lines.append(f"PARIMA: μ={parima_data.mean():.2f}, M={parima_data.median():.2f}")
            if len(lstm_data) > 0:
                stats_lines.append(f"LSTM: μ={lstm_data.mean():.2f}, M={lstm_data.median():.2f}")
            
            stats_text = "\n".join(stats_lines)
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '6_runtime_metrics_boxplot_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_path}")
        
    except Exception as e:
        print(f"⚠️  Error creating comparison metrics box plots: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots: RandomForest vs LSTM')
    parser.add_argument('--comparison-dir', type=str, default='./model_comparison',
                       help='Directory containing model comparison results')
    parser.add_argument('--output-dir', type=str, default='./model_comparison_plots',
                       help='Output directory for plots')
    parser.add_argument('--csv-data', type=str, default=None,
                       help='Optional: Path to CSV file with runtime metrics for box plots')
    parser.add_argument('--parima-csv', type=str, default=None,
                       help='Path to PARIMA CSV file for runtime metrics comparison')
    parser.add_argument('--lstm-csv', type=str, default=None,
                       help='Path to LSTM CSV file for runtime metrics comparison')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("RandomForest vs LSTM Model Comparison Visualization")
    print("=" * 80)
    print()
    
    # Load results
    print("Loading model comparison results...")
    results = load_model_results(args.comparison_dir)
    
    if not results['rf_accuracy']:
        print("❌ Could not load model results. Make sure you've run compare_models.py first.")
        print(f"   Expected report at: {os.path.join(args.comparison_dir, 'model_comparison_report.txt')}")
        return
    
    print(f"✅ Loaded results:")
    print(f"   RandomForest Accuracy: {results['rf_accuracy']:.4f}")
    if results['lstm_single_accuracy']:
        print(f"   LSTM Single Split: {results['lstm_single_accuracy']:.4f}")
    if results['lstm_cv_mean']:
        print(f"   LSTM CV Mean: {results['lstm_cv_mean']:.4f} (±{results['lstm_cv_std']:.4f})")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    plot_accuracy_comparison(results, args.output_dir)
    plot_cv_comparison(results, args.output_dir)
    plot_model_characteristics(results, args.output_dir)
    plot_accuracy_distribution(results, args.output_dir)
    plot_accuracy_boxplot(results, args.output_dir)  # NEW: Box plot for accuracy
    
    # Optional: Box plots from CSV data
    csv_path = args.csv_data
    if not csv_path:
        # Try default path
        default_csv = '../data/training_logs/parima_decisions_log.csv'
        if os.path.exists(default_csv):
            csv_path = default_csv
            print(f"\n📊 Using default CSV path: {csv_path}")
    
    if csv_path:
        print(f"\n📊 Generating runtime metrics box plots from CSV...")
        plot_metrics_boxplot_from_csv(csv_path, args.output_dir)
    else:
        print(f"\n⚠️  No CSV data provided. Skipping runtime metrics box plots.")
        print("   Use --csv-data to specify CSV file with runtime metrics.")
    
    # NEW: Comparison box plots if both CSVs provided
    parima_csv = args.parima_csv
    lstm_csv = args.lstm_csv
    
    # Try default paths if not provided
    if not parima_csv:
        default_parima = '../data/training_logs/parima_decisions_log.csv'
        if os.path.exists(default_parima):
            parima_csv = default_parima
    
    if not lstm_csv:
        default_lstm = '../data/training_logs/lstm_decisions_log.csv'
        if os.path.exists(default_lstm):
            lstm_csv = default_lstm
    
    if parima_csv and lstm_csv:
        print(f"\n📊 Generating PARIMA vs LSTM runtime metrics comparison...")
        plot_metrics_boxplot_comparison(parima_csv, lstm_csv, args.output_dir)
    else:
        if not parima_csv and not lstm_csv:
            print(f"\n⚠️  No CSV data provided for comparison. Skipping runtime metrics comparison.")
            print("   Use --parima-csv and --lstm-csv to specify CSV files for comparison.")
        else:
            print(f"\n⚠️  Missing CSV file(s). Skipping comparison.")
            if not parima_csv:
                print("   PARIMA CSV not found. Use --parima-csv to specify.")
            if not lstm_csv:
                print("   LSTM CSV not found. Use --lstm-csv to specify.")
    
    print("\n" + "=" * 80)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

