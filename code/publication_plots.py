import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Enhanced color palette for better color distinction and accessibility
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def load_metrics(results_dir):
    """Load metrics from the metrics.json file"""
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found at {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def create_defense_comparison_plot(defense_dirs, output_dir, metrics_to_plot=None):
    """
    Create publication-quality comparison plots for different defense methods
    
    Args:
        defense_dirs: Dictionary of {defense_name: results_dir}
        output_dir: Directory to save comparison plots
        metrics_to_plot: List of metrics to plot (default: test_acc, robust_acc, f1)
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['test_acc', 'robust_acc', 'test_f1']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics for each defense
    defense_metrics = {}
    for defense_name, result_dir in defense_dirs.items():
        metrics = load_metrics(result_dir)
        if metrics:
            defense_metrics[defense_name] = metrics
    
    if not defense_metrics:
        print("No metrics found for any defense method.")
        return
    
    # Determine maximum number of rounds
    max_rounds = max([len(metrics.get('test_acc', [])) for metrics in defense_metrics.values()])
    
    # Set up figure and axes
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        for j, (defense_name, metrics) in enumerate(defense_metrics.items()):
            if metric in metrics and metrics[metric]:
                # Get data and limit to max_rounds
                rounds = list(range(1, len(metrics[metric]) + 1))
                values = metrics[metric][:max_rounds]
                rounds = rounds[:max_rounds]
                
                # Plot with enhanced style
                ax.plot(rounds, values, marker='o', label=defense_name.replace('_', ' ').upper(),
                       color=colors[j % len(colors)], linewidth=2.5, markersize=8)
        
        # Set labels and title
        metric_name = metric.replace('_', ' ').title()
        if metric == 'test_acc':
            metric_name = 'Test Accuracy'
        elif metric == 'robust_acc':
            metric_name = 'Robust Accuracy'
        elif metric == 'test_f1':
            metric_name = 'F1 Score'
        elif metric == 'test_auc':
            metric_name = 'AUC'
            
        ax.set_xlabel('Federated Learning Rounds', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold')
        
        # Add grid and improve appearance
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis limits appropriately
        if metric.endswith('acc') or metric.endswith('auc') or metric.endswith('f1'):
            ax.set_ylim(0, 1.05)
        
        # Add legend to the first subplot only
        if i == 0:
            ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True, shadow=True)
    
    # Add overall title
    fig.suptitle('Comparison of Defense Methods for Federated Learning', fontweight='bold', y=1.05)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'defense_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'defense_comparison.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def create_robustness_gap_comparison(defense_dirs, output_dir):
    """Create robustness gap comparison for different defense methods"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average robustness gap for each defense
    gaps = {}
    for defense_name, result_dir in defense_dirs.items():
        metrics = load_metrics(result_dir)
        if not metrics or 'test_acc' not in metrics or 'robust_acc' not in metrics:
            continue
            
        # Calculate gap for each round
        acc_gaps = []
        for i in range(min(len(metrics['test_acc']), len(metrics['robust_acc']))):
            if metrics['robust_acc'][i] is not None:
                gap = metrics['test_acc'][i] - metrics['robust_acc'][i]
                acc_gaps.append(gap)
        
        if acc_gaps:
            # Calculate average gap
            avg_gap = np.mean(acc_gaps)
            gaps[defense_name] = avg_gap
    
    if not gaps:
        print("No robustness gap data available.")
        return
    
    # Sort by gap (smaller is better)
    sorted_defenses = sorted(gaps.items(), key=lambda x: x[1])
    defenses = [d[0].replace('_', ' ').upper() for d in sorted_defenses]
    gap_values = [d[1] for d in sorted_defenses]
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Create colormap based on gap size (smaller is better)
    norm = plt.Normalize(0, max(gap_values) * 1.2)
    colors = plt.cm.RdYlGn_r(norm(gap_values))
    
    bars = plt.bar(defenses, gap_values, color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f"{height:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.xlabel('Defense Method', fontweight='bold', fontsize=14)
    plt.ylabel('Average Robustness Gap (smaller is better)', fontweight='bold', fontsize=14)
    plt.title('Attack Robustness Comparison', fontweight='bold', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_gap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'robustness_gap_comparison.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def create_trust_evolution_plot(metrics, output_dir, experiment_name):
    """Create enhanced trust evolution plot"""
    if 'client_trust' not in metrics or not metrics['client_trust']:
        return
        
    trust_data = metrics['client_trust']
    num_rounds = len(trust_data)
    rounds = list(range(1, num_rounds + 1))
    
    plt.figure(figsize=(12, 8))
    
    # Create colormap for clients
    num_clients = len(trust_data[0])
    
    # Plot trust scores with enhanced style
    for client_idx in range(num_clients):
        client_trust = [round_trust[client_idx] for round_trust in trust_data]
        plt.plot(rounds, client_trust, marker='o', label=f'Client {client_idx+1}',
                linewidth=2.5, color=colors[client_idx % len(colors)])
    
    plt.title('Client Trust Scores Evolution', fontweight='bold', fontsize=16)
    plt.xlabel('Federated Learning Rounds', fontweight='bold', fontsize=14)
    plt.ylabel('Trust Score', fontweight='bold', fontsize=14)
    plt.ylim(0, 1.05)
    
    # Add threshold lines with better styling
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, 
               label='High Trust Threshold', linewidth=2)
    plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, 
               label='Low Trust Threshold', linewidth=2)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add enhanced legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, 
              fancybox=True, shadow=True)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(os.path.join(output_dir, 'trust_scores_evolution.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'trust_scores_evolution.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def create_performance_metrics_plot(metrics, output_dir, experiment_name):
    """Create enhanced performance metrics plot"""
    if 'test_acc' not in metrics or not metrics['test_acc']:
        return
        
    num_rounds = len(metrics['test_acc'])
    rounds = list(range(1, num_rounds + 1))
    
    plt.figure(figsize=(12, 8))
    
    # Plot metrics with enhanced styling
    if 'test_acc' in metrics:
        plt.plot(rounds, metrics['test_acc'], marker='o', label='Test Accuracy', 
                 color=colors[0], linewidth=2.5, markersize=8)
    
    if 'test_auc' in metrics:
        plt.plot(rounds, metrics['test_auc'], marker='s', label='AUC', 
                 color=colors[1], linewidth=2.5, markersize=8)
    
    if 'test_f1' in metrics:
        plt.plot(rounds, metrics['test_f1'], marker='^', label='F1 Score', 
                 color=colors[2], linewidth=2.5, markersize=8)
    
    if 'robust_acc' in metrics and metrics['robust_acc'][0] is not None:
        robust_acc = [acc if acc is not None else 0 for acc in metrics['robust_acc']]
        plt.plot(rounds, robust_acc, marker='d', linestyle='--', 
                label='Robust Accuracy', color=colors[3], linewidth=2.5, markersize=8)
    
    # Enhanced styling
    plt.title('Model Performance Metrics Over Rounds', fontweight='bold', fontsize=16)
    plt.xlabel('Federated Learning Rounds', fontweight='bold', fontsize=14)
    plt.ylabel('Performance Score', fontweight='bold', fontsize=14)
    plt.ylim(0, 1.05)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add enhanced legend
    plt.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True, shadow=True)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'performance_metrics.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def create_trust_heatmap(metrics, output_dir, experiment_name):
    """Create enhanced trust score heatmap"""
    if 'client_trust' not in metrics or not metrics['client_trust']:
        return
    
    # Create matrix for heatmap
    trust_matrix = np.array(metrics['client_trust'])
    num_rounds, num_clients = trust_matrix.shape
    
    plt.figure(figsize=(14, 8))
    
    # Create enhanced heatmap
    ax = plt.gca()
    im = ax.imshow(trust_matrix.T, cmap='RdYlGn', aspect='auto', interpolation='none',
                  vmin=0.0, vmax=1.0)
    
    # Configure axes with better styling
    ax.set_xlabel('Federated Learning Round', fontweight='bold', fontsize=14)
    ax.set_ylabel('Client ID', fontweight='bold', fontsize=14)
    ax.set_title('Trust Score Heatmap', fontweight='bold', fontsize=16)
    
    # Set ticks and labels
    round_ticks = np.arange(0, num_rounds, max(1, num_rounds // 10))
    ax.set_xticks(round_ticks)
    ax.set_xticklabels([str(r + 1) for r in round_ticks])
    
    ax.set_yticks(np.arange(num_clients))
    ax.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
    
    # Add grid for better readability
    for i in range(num_clients):
        ax.axhline(i + 0.5, color='white', linewidth=1)
    for i in range(num_rounds):
        ax.axvline(i + 0.5, color='white', linewidth=1)
    
    # Add colorbar with better styling
    cbar = plt.colorbar(im)
    cbar.set_label('Trust Score', fontweight='bold', fontsize=14)
    
    # Annotate attacks
    for client in range(num_clients):
        client_trust = trust_matrix[:, client]
        
        # Find significant drops in trust (potential attack points)
        for i in range(1, len(client_trust)):
            if client_trust[i] <= 0.3 and client_trust[i-1] >= 0.7:
                ax.plot(i, client, 'rx', markersize=10, markeredgewidth=3)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(os.path.join(output_dir, 'trust_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'trust_heatmap.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def create_privacy_utility_tradeoff(output_dir):
    """
    Create privacy-utility tradeoff plot showing model accuracy vs privacy budget (epsilon)
    """
    # Sample data (replace with your actual results)
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    # Accuracy values for different methods
    dpfedguard = [78.5, 83.2, 86.7, 89.3, 91.8, 93.2, 94.1]
    standard_fl = [65.2, 77.5, 83.9, 88.1, 91.3, 92.9, 93.8]
    central_dp = [72.1, 75.8, 79.3, 82.6, 85.2, 87.5, 89.2]
    local_dp = [68.7, 70.9, 73.2, 76.8, 80.3, 83.1, 85.7]
    
    # Create figure
    plt.figure(figsize=(12, 9))
    
    # Plot each method with enhanced styling
    plt.plot(epsilons, dpfedguard, 'o-', color=colors[0], linewidth=2.5, 
             markersize=8, label='DPFedGuard (Ours)')
    plt.plot(epsilons, standard_fl, 's-', color=colors[1], linewidth=2, 
             markersize=7, label='Standard FL')
    plt.plot(epsilons, central_dp, '^-', color=colors[2], linewidth=2, 
             markersize=7, label='Central DP-FL')
    plt.plot(epsilons, local_dp, 'D-', color=colors[3], linewidth=2, 
             markersize=7, label='Local DP-FL')
    
    # Configure the plot
    plt.xscale('log')  # Log scale for epsilon
    plt.xlabel('Privacy Budget (ε) - Lower is More Private', fontweight='bold')
    plt.ylabel('Model Accuracy (%)', fontweight='bold')
    plt.title('Privacy-Utility Tradeoff in Federated Learning', fontweight='bold')
    
    # Add annotation highlighting better performance
    plt.annotate('Better privacy-utility\ntradeoff with DPFedGuard',
                xy=(0.5, 83.2), xytext=(2.0, 75),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Set y-axis limits to better show differences
    plt.ylim(60, 100)
    
    # Remove top and right spines (consistent with your style)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add legend with enhanced styling
    plt.legend(loc='lower right', frameon=True, framealpha=0.9, 
              fancybox=True, shadow=True)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'privacy_utility_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'privacy_utility_tradeoff.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Privacy-utility tradeoff plot saved to {output_dir}/")

def create_attack_success_rate_plot(output_dir):
    """
    Create plot showing membership inference attack success rates
    """
    # Sample data (replace with your actual results)
    methods = ['DPFedGuard', 'Standard FL', 'Central DP-FL', 'Local DP-FL']
    
    # Attack success rate for different privacy budgets (lower is better)
    attack_success = {
        'ε=0.1': [52.1, 89.5, 58.3, 53.8],
        'ε=1.0': [58.4, 92.7, 67.5, 61.2],
        'ε=10.0': [67.8, 94.3, 78.9, 74.5]
    }
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(attack_success, index=methods)
    
    # Create figure
    plt.figure(figsize=(12, 9))
    
    # Create grouped bar chart with enhanced styling
    ax = df.plot(kind='bar', width=0.7, edgecolor='black', linewidth=1.2, ax=plt.gca())
    
    # Add a horizontal line at 50% (random guessing)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, 
               label='Random Guess (No Information)', linewidth=2)
    
    # Configure the plot
    plt.xlabel('Method', fontweight='bold')
    plt.ylabel('Membership Inference Attack Success Rate (%)', fontweight='bold')
    plt.title('Resistance to Membership Inference Attacks', fontweight='bold')
    plt.ylim(45, 100)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add annotation highlighting better defense
    plt.annotate('DPFedGuard provides best\ndefense against attacks',
                xy=(0, 52.1), xytext=(1.5, 47),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Enhanced legend
    plt.legend(title='Privacy Budget', loc='upper right', frameon=True, 
              framealpha=0.9, fancybox=True, shadow=True)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_success_rates.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'attack_success_rates.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Attack success rates plot saved to {output_dir}/")

def create_all_enhanced_visualizations(results_dir, output_dir=None):
    """Create all enhanced visualizations for a single experiment"""
    # Load metrics
    metrics = load_metrics(results_dir)
    if metrics is None:
        return
    
    # Create output directory
    experiment_name = os.path.basename(results_dir)
    if output_dir is None:
        output_dir = os.path.join(results_dir, "publication_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating enhanced visualizations for: {experiment_name}")
    
    # Create enhanced visualizations
    create_performance_metrics_plot(metrics, output_dir, experiment_name)
    create_trust_evolution_plot(metrics, output_dir, experiment_name)
    create_trust_heatmap(metrics, output_dir, experiment_name)
    
    # Create additional visualizations
    create_privacy_utility_tradeoff(output_dir)
    create_attack_success_rate_plot(output_dir)
    
    print(f"Enhanced visualizations saved to: {output_dir}")

def main():
    """Main function to generate all publications plots"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create publication-quality visualizations")
    parser.add_argument("--results_dir", type=str, default="../results", 
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save enhanced visualizations")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Specific experiment to visualize")
    parser.add_argument("--compare", action="store_true",
                        help="Create comparison plots for different defense methods")
    args = parser.parse_args()
    
    if args.experiment:
        # Create visualizations for a specific experiment
        exp_dir = os.path.join(args.results_dir, args.experiment)
        out_dir = args.output_dir or os.path.join(exp_dir, "publication_plots")
        create_all_enhanced_visualizations(exp_dir, out_dir)
    elif args.compare:
        # Create comparison visualizations for all defense methods
        defense_dirs = {}
        
        for dirname in os.listdir(args.results_dir):
            if dirname.startswith("final_"):
                full_path = os.path.join(args.results_dir, dirname)
                if os.path.isdir(full_path):
                    defense_name = dirname.replace("final_", "")
                    defense_dirs[defense_name] = full_path
        
        if defense_dirs:
            out_dir = args.output_dir or os.path.join(args.results_dir, "comparison_plots")
            create_defense_comparison_plot(defense_dirs, out_dir)
            create_robustness_gap_comparison(defense_dirs, out_dir)
            print(f"Comparison plots saved to: {out_dir}")
        else:
            print("No defense experiment directories found.")
    else:
        # Process all experiments in the results directory
        for dirname in os.listdir(args.results_dir):
            full_path = os.path.join(args.results_dir, dirname)
            if os.path.isdir(full_path):
                out_dir = args.output_dir or os.path.join(full_path, "publication_plots")
                create_all_enhanced_visualizations(full_path, out_dir)

if __name__ == "__main__":
    main()