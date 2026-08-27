# Create paper_figures.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def create_defense_comparison_figure(baselines_file, output_dir):
    """Create main defense comparison figure"""
    df = pd.read_csv(baselines_file)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter and sort data
    df['method'] = df.apply(lambda row: row['defense_type'].upper(), axis=1)
    df = df.sort_values('test_acc_mean', ascending=False)
    
    # 1. Accuracy and AUC subplot
    bar_width = 0.25
    x = np.arange(len(df))
    
    # Plot accuracy
    ax1.bar(x - bar_width, df['test_acc_mean'], bar_width, 
           yerr=df['test_acc_ci95'], capsize=4,
           label='Accuracy', color='#1f77b4', edgecolor='black', linewidth=1)
    
    # Plot AUC
    ax1.bar(x, df['test_auc_mean'], bar_width,
           yerr=df['test_auc_ci95'], capsize=4,
           label='AUC', color='#ff7f0e', edgecolor='black', linewidth=1)
    
    # Plot F1
    ax1.bar(x + bar_width, df['test_f1_mean'], bar_width,
           yerr=df['test_f1_ci95'], capsize=4,
           label='F1', color='#2ca02c', edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Defense Method')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['method'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Robustness gap subplot
    if 'robustness_gap' in df.columns:
        robust_df = df.dropna(subset=['robustness_gap']).sort_values('robustness_gap')
        
        # Create colormap based on gap size
        norm = plt.Normalize(0, robust_df['robustness_gap'].max() * 1.2)
        colors = plt.cm.RdYlGn_r(norm(robust_df['robustness_gap']))
        
        x2 = np.arange(len(robust_df))
        
        # Plot bars
        bars = ax2.bar(x2, robust_df['robustness_gap'], color=colors, 
                      edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f"{height:.3f}", ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Defense Method')
        ax2.set_ylabel('Robustness Gap (smaller is better)')
        ax2.set_title('Attack Robustness')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(robust_df['method'], rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Defense Methods Comparison', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'defense_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_attack_impact_figure(attack_scenarios_file, output_dir):
    """Create attack impact figure"""
    df = pd.read_csv(attack_scenarios_file)
    
    # Extract attack fraction from config
    df['attack_fraction'] = df.apply(
        lambda row: float(row['attack_fraction']) if 'attack_fraction' in row else 0.0, 
        axis=1
    )
    
    # Sort by attack fraction
    df = df.sort_values('attack_fraction')
    
    plt.figure(figsize=(10, 6))
    
    # Plot metrics vs attack fraction
    x = df['attack_fraction'].values
    
    plt.plot(x, df['test_acc_mean'], marker='o', label='Accuracy', 
            linewidth=2, color='#1f77b4')
    plt.fill_between(x, 
                    df['test_acc_mean'] - df['test_acc_std'],
                    df['test_acc_mean'] + df['test_acc_std'],
                    alpha=0.2, color='#1f77b4')
    
    plt.plot(x, df['test_auc_mean'], marker='s', label='AUC', 
            linewidth=2, color='#ff7f0e')
    plt.fill_between(x,
                    df['test_auc_mean'] - df['test_auc_std'],
                    df['test_auc_mean'] + df['test_auc_std'],
                    alpha=0.2, color='#ff7f0e')
    
    # Plot robustness gap if available
    if 'robustness_gap' in df.columns:
        plt.plot(x, df['robustness_gap'], marker='d', label='Robustness Gap',
                linewidth=2, color='#d62728', linestyle='--')
    
    plt.xlabel('Fraction of Malicious Clients')
    plt.ylabel('Score')
    plt.title('Impact of Attacker Presence on DP-FedGuard Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format x-axis as percentage
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'attack_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_privacy_utility_figure(privacy_file, output_dir):
    """Create privacy-utility tradeoff figure"""
    df = pd.read_csv(privacy_file)
    
    # Extract noise multiplier from config
    df['noise_multiplier'] = df.apply(
        lambda row: float(row['noise_multiplier']) if 'noise_multiplier' in row else 1.0,
        axis=1
    )
    
    # Sort by noise multiplier
    df = df.sort_values('noise_multiplier')
    
    plt.figure(figsize=(10, 6))
    
    # Primary y-axis for accuracy metrics
    ax1 = plt.gca()
    x = df['noise_multiplier'].values
    
    # Plot accuracy
    line1 = ax1.plot(x, df['test_acc_mean'], marker='o', label='Accuracy',
                    linewidth=2, color='#1f77b4')
    ax1.fill_between(x,
                    df['test_acc_mean'] - df['test_acc_std'],
                    df['test_acc_mean'] + df['test_acc_std'],
                    alpha=0.2, color='#1f77b4')
    
    # Plot AUC
    line2 = ax1.plot(x, df['test_auc_mean'], marker='s', label='AUC',
                    linewidth=2, color='#ff7f0e')
    ax1.fill_between(x,
                    df['test_auc_mean'] - df['test_auc_std'],
                    df['test_auc_mean'] + df['test_auc_std'],
                    alpha=0.2, color='#ff7f0e')
    
    # Secondary y-axis for noise level
    ax2 = ax1.twinx()
    line3 = ax2.plot(x, x, marker='d', label='Privacy Budget',
                    linewidth=2, color='#d62728', linestyle='--')
    
    # Combine lines from both axes in one legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    ax1.set_xlabel('Noise Multiplier')
    ax1.set_ylabel('Performance Score')
    ax2.set_ylabel('Privacy Budget (Noise Level)')
    
    plt.title('Privacy-Utility Tradeoff for DP-FedGuard')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'privacy_utility_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base directory containing all analysis results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save paper figures")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "paper_figures")
    
    # Generate main comparison figure
    baselines_file = os.path.join(args.results_dir, "baselines/analysis/baselines_statistics.csv")
    if os.path.exists(baselines_file):
        create_defense_comparison_figure(baselines_file, args.output_dir)
    
    # Generate attack impact figure
    attack_file = os.path.join(args.results_dir, "attack_scenarios/analysis/attack_scenarios_statistics.csv")
    if os.path.exists(attack_file):
        create_attack_impact_figure(attack_file, args.output_dir)
    
    # Generate privacy-utility figure
    privacy_file = os.path.join(args.results_dir, "privacy_utility/analysis/privacy_utility_statistics.csv")
    if os.path.exists(privacy_file):
        create_privacy_utility_figure(privacy_file, args.output_dir)
    
    print(f"Paper figures generated and saved to {args.output_dir}")