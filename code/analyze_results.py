# Create analyze_results.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def load_results(group_dir, metric_keys=None):
    """Load results for an experiment group across all seeds"""
    if metric_keys is None:
        metric_keys = ['test_acc', 'test_auc', 'test_f1', 'robust_acc']
    
    results = {}
    
    for exp_name in os.listdir(group_dir):
        exp_dir = os.path.join(group_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
            
        metrics_file = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(metrics_file):
            continue
            
        # Parse experiment name to get config details
        parts = exp_name.split("_")
        config = {}
        seed = None
        
        for i, part in enumerate(parts):
            if part.startswith("seed"):
                seed = int(part[4:])
            elif i < len(parts) - 1:
                config[part] = parts[i+1]
        
        if seed is None:
            continue
            
        # Create a unique key for this configuration
        config_key = "_".join([f"{k}_{v}" for k, v in config.items()])
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        # Extract relevant metrics
        if config_key not in results:
            results[config_key] = {key: [] for key in metric_keys}
            results[config_key]['seeds'] = []
            results[config_key]['config'] = config
            
        # Store final values for each metric
        for key in metric_keys:
            if key in metrics and metrics[key]:
                if key.startswith('robust') and metrics[key][-1] is None:
                    results[config_key][key].append(None)
                else:
                    results[config_key][key].append(metrics[key][-1])
            else:
                results[config_key][key].append(None)
                
        results[config_key]['seeds'].append(seed)
    
    return results

def calculate_statistics(results, metric_keys=None):
    """Calculate statistics across seeds for each configuration"""
    if metric_keys is None:
        metric_keys = ['test_acc', 'test_auc', 'test_f1', 'robust_acc']
    
    stats_df = []
    
    for config_key, data in results.items():
        row = {'config_key': config_key}
        row.update(data['config'])
        
        for metric in metric_keys:
            values = [v for v in data[metric] if v is not None]
            if values:
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
                row[f'{metric}_min'] = np.min(values)
                row[f'{metric}_max'] = np.max(values)
                
                # Calculate 95% confidence interval
                if len(values) >= 2:
                    t_val = stats.t.ppf(0.975, len(values)-1)
                    ci = t_val * np.std(values) / np.sqrt(len(values))
                    row[f'{metric}_ci95'] = ci
                else:
                    row[f'{metric}_ci95'] = 0
            else:
                row[f'{metric}_mean'] = None
                row[f'{metric}_std'] = None
                row[f'{metric}_min'] = None
                row[f'{metric}_max'] = None
                row[f'{metric}_ci95'] = None
        
        # Calculate robustness gap if both metrics exist
        if 'test_acc_mean' in row and 'robust_acc_mean' in row and row['test_acc_mean'] is not None and row['robust_acc_mean'] is not None:
            row['robustness_gap'] = row['test_acc_mean'] - row['robust_acc_mean']
        
        stats_df.append(row)
    
    return pd.DataFrame(stats_df)

def create_comparison_plots(stats_df, group_name, output_dir):
    """Create publication-quality comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    
    # Sort by test accuracy
    sorted_df = stats_df.sort_values('test_acc_mean', ascending=False)
    
    # Create x-axis labels
    x_labels = []
    for _, row in sorted_df.iterrows():
        if 'defense_type' in row:
            defense = row['defense_type'].upper()
            x_labels.append(defense)
        else:
            x_labels.append(row['config_key'])
    
    # Plot accuracy bars
    x = np.arange(len(sorted_df))
    width = 0.35
    
    plt.bar(x - width/2, sorted_df['test_acc_mean'], width, 
            yerr=sorted_df['test_acc_ci95'],
            label='Test Accuracy', color='#1f77b4',
            capsize=5, edgecolor='black', linewidth=1)
    
    # Add robust accuracy if available
    if not sorted_df['robust_acc_mean'].isnull().all():
        plt.bar(x + width/2, sorted_df['robust_acc_mean'], width,
                yerr=sorted_df['robust_acc_ci95'],
                label='Robust Accuracy', color='#ff7f0e',
                capsize=5, edgecolor='black', linewidth=1)
    
    plt.xlabel('Defense Method')
    plt.ylabel('Accuracy')
    plt.title(f'Defense Performance Comparison - {group_name}')
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{group_name}_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Robustness Gap Comparison (if robust metrics exist)
    if 'robustness_gap' in sorted_df.columns and not sorted_df['robustness_gap'].isnull().all():
        plt.figure(figsize=(10, 6))
        
        # Sort by robustness gap (ascending - smaller is better)
        gap_df = sorted_df.dropna(subset=['robustness_gap']).sort_values('robustness_gap')
        
        # Create labels
        gap_labels = []
        for _, row in gap_df.iterrows():
            if 'defense_type' in row:
                defense = row['defense_type'].upper()
                gap_labels.append(defense)
            else:
                gap_labels.append(row['config_key'])
        
        # Create colormap based on gap size
        norm = plt.Normalize(0, gap_df['robustness_gap'].max() * 1.2)
        colors = plt.cm.RdYlGn_r(norm(gap_df['robustness_gap']))
        
        x = np.arange(len(gap_df))
        
        # Plot bars
        bars = plt.bar(x, gap_df['robustness_gap'], color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f"{height:.3f}", ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Defense Method')
        plt.ylabel('Robustness Gap (smaller is better)')
        plt.title(f'Attack Robustness Comparison - {group_name}')
        plt.xticks(x, gap_labels, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{group_name}_robustness_gap.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    
    # Extract group name from directory
    group_name = os.path.basename(args.results_dir)
    
    # Load results
    results = load_results(args.results_dir)
    
    # Calculate statistics
    stats_df = calculate_statistics(results)
    
    # Save statistics to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(args.output_dir, f"{group_name}_statistics.csv"), index=False)
    
    # Create comparison plots
    create_comparison_plots(stats_df, group_name, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")