import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import ticker

# Set style for publication-quality plots
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

def create_results_visualization(results_dir, output_dir=None, high_dpi=True):
    """
    Create comprehensive visualizations from saved metrics
    with publication-quality figures
    
    Args:
        results_dir: Directory containing the metrics.json file
        output_dir: Directory to save the plots (default: results_dir/plots)
        high_dpi: Whether to save high-resolution figures for publication
    """
    # Load metrics from file
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found at {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create plots directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure DPI for saving
    fig_dpi = 300 if high_dpi else 100
    
    # Extract experiment name from directory
    experiment_name = os.path.basename(results_dir)
    print(f"Creating visualizations for experiment: {experiment_name}")
    
    # ============ 1. Performance Metrics Plot ============
    create_performance_plot(metrics, output_dir, fig_dpi, experiment_name)
    
    # ============ 2. Trust Scores Analysis ============
    if 'client_trust' in metrics and len(metrics['client_trust']) > 0:
        create_trust_score_plots(metrics, output_dir, fig_dpi, experiment_name)
    
    # ============ 3. Combined Results Plot ============
    create_combined_results_plot(metrics, output_dir, fig_dpi, experiment_name)
    
    # ============ 4. Robustness Analysis ============
    if 'robust_acc' in metrics and metrics['robust_acc'][0] is not None:
        create_robustness_plot(metrics, output_dir, fig_dpi, experiment_name)
    
    # ============ 5. Create Summary Visualization ============
    create_summary_visualization(metrics, output_dir, fig_dpi, experiment_name)
    
    print(f"Visualizations created and saved to {output_dir}")

def create_performance_plot(metrics, output_dir, fig_dpi, experiment_name):
    """Create performance metrics plot with error bands"""
    if 'test_acc' not in metrics or not metrics['test_acc']:
        return
        
    num_rounds = len(metrics['test_acc'])
    rounds = list(range(1, num_rounds + 1))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot metrics with different styles and colors
    if 'test_acc' in metrics:
        plt.plot(rounds, metrics['test_acc'], marker='o', label='Test Accuracy', 
                 color='#1f77b4', linewidth=2, markersize=6)
    
    if 'test_auc' in metrics:
        plt.plot(rounds, metrics['test_auc'], marker='s', label='Test AUC', 
                 color='#ff7f0e', linewidth=2, markersize=6)
    
    if 'test_f1' in metrics:
        plt.plot(rounds, metrics['test_f1'], marker='^', label='Test F1', 
                 color='#2ca02c', linewidth=2, markersize=6)
    
    # Add confidence region (illustrative - in a real scenario you'd calculate this from multiple runs)
    if 'test_acc' in metrics:
        mean_acc = np.array(metrics['test_acc'])
        error_margin = 0.02  # Illustrative error margin
        plt.fill_between(rounds, mean_acc - error_margin, mean_acc + error_margin, 
                         color='#1f77b4', alpha=0.2)
    
    # Enhance plot appearance
    plt.title(f'Model Performance Metrics Over Rounds\n{experiment_name}')
    plt.xlabel('Federated Learning Rounds')
    plt.ylabel('Performance Score')
    plt.ylim([0.0, 1.05])  # Standard range for metrics
    plt.xlim([0.8, num_rounds + 0.2])  # Adjust x limits
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create legend
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    # Add annotations for key points
    if 'test_acc' in metrics:
        max_acc_round = np.argmax(metrics['test_acc']) + 1
        max_acc = max(metrics['test_acc'])
        plt.annotate(f'Max Acc: {max_acc:.3f}', 
                    xy=(max_acc_round, max_acc),
                    xytext=(max_acc_round, max_acc + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def create_trust_score_plots(metrics, output_dir, fig_dpi, experiment_name):
    """Create detailed trust score visualizations"""
    trust_data = metrics['client_trust']
    num_rounds = len(trust_data)
    rounds = list(range(1, num_rounds + 1))
    
    # ========== Trust Score Evolution Plot ==========
    plt.figure(figsize=(10, 6))
    
    # Create colormap for clients
    num_clients = len(trust_data[0])
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=num_clients-1)
    
    for client_idx in range(num_clients):
        client_trust = [round_trust[client_idx] for round_trust in trust_data]
        color = cmap(norm(client_idx))
        plt.plot(rounds, client_trust, marker='o', label=f'Client {client_idx+1}',
                linewidth=2, color=color)
    
    plt.title(f'Client Trust Scores Evolution\n{experiment_name}')
    plt.xlabel('Federated Learning Rounds')
    plt.ylabel('Trust Score')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal lines for trust thresholds
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Trust Threshold')
    plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Low Trust Threshold')
    
    # Improve legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trust_scores.png'), dpi=fig_dpi, bbox_inches='tight')
    plt.close()
    
    # ========== Trust Score Heatmap ==========
    plt.figure(figsize=(12, 6))
    
    # Create matrix for heatmap
    trust_matrix = np.array(trust_data)
    
    # Calculate metrics for annotation
    mean_trust = np.mean(trust_matrix, axis=0)
    min_trust = np.min(trust_matrix, axis=0)
    
    # Create heatmap
    ax = plt.gca()
    im = ax.imshow(trust_matrix.T, cmap='RdYlGn', aspect='auto', interpolation='none',
                  vmin=0.0, vmax=1.0)
    
    # Configure axes
    ax.set_xlabel('Federated Learning Round')
    ax.set_ylabel('Client ID')
    ax.set_title(f'Trust Score Heatmap\n{experiment_name}')
    
    # Set ticks
    ax.set_xticks(np.arange(0, num_rounds, max(1, num_rounds // 10)))
    ax.set_xticklabels(np.arange(1, num_rounds+1, max(1, num_rounds // 10)))
    ax.set_yticks(np.arange(num_clients))
    ax.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Trust Score')
    
    # Add annotations
    for client in range(num_clients):
        # Add mean and min annotations on the right side
        ax.text(num_rounds + 0.5, client, f'Avg: {mean_trust[client]:.2f}', 
               ha='left', va='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trust_heatmap.png'), dpi=fig_dpi, bbox_inches='tight')
    plt.close()
    
    # ========== Trust Distribution Over Time ==========
    if num_rounds >= 5:  # Only create if we have enough rounds
        plt.figure(figsize=(10, 6))
        
        # Create data for violin plot
        data = [[] for _ in range(min(num_rounds, 10))]
        step = max(1, num_rounds // 10)
        idx = 0
        
        for round_idx in range(0, num_rounds, step):
            if round_idx < num_rounds and idx < len(data):
                data[idx] = trust_data[round_idx]
                idx += 1
        
        # Fill in the last round if needed
        if idx < len(data):
            data[idx] = trust_data[-1]
        
        # Create violin plot
        positions = list(range(1, len(data) + 1))
        violins = plt.violinplot(data, positions=positions, showmeans=True, 
                                showmedians=True, showextrema=True)
        
        # Customize violin plot
        for i, violin in enumerate(violins['bodies']):
            violin.set_facecolor('lightblue')
            violin.set_edgecolor('black')
            violin.set_alpha(0.7)
        
        # Set labels and title
        plt.title(f'Client Trust Distribution Evolution\n{experiment_name}')
        plt.xlabel('Sampling of Rounds')
        plt.ylabel('Trust Score Distribution')
        plt.ylim(0, 1.05)
        
        # Set x-ticks to show actual round numbers
        round_labels = list(range(0, num_rounds, step))[:len(data)]
        if len(round_labels) < len(data):
            round_labels.append(num_rounds - 1)
        round_labels = [r + 1 for r in round_labels]  # 1-indexed rounds
        
        plt.xticks(positions, round_labels)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trust_distribution.png'), dpi=fig_dpi, bbox_inches='tight')
        plt.close()

def create_combined_results_plot(metrics, output_dir, fig_dpi, experiment_name):
    """Create comprehensive combined results plot"""
    plt.figure(figsize=(15, 12))
    
    # Use GridSpec for more flexible layout
    gs = GridSpec(3, 2, figure=plt.gcf(), height_ratios=[1, 1, 1])
    
    # Plot 1: Accuracy Metrics
    ax1 = plt.subplot(gs[0, 0])
    rounds = list(range(1, len(metrics['test_acc']) + 1))
    
    if 'test_acc' in metrics:
        ax1.plot(rounds, metrics['test_acc'], marker='o', label='Test Accuracy',
                color='#1f77b4', linewidth=2)
    
    if 'robust_acc' in metrics and metrics['robust_acc'][0] is not None:
        robust_acc = [acc if acc is not None else 0 for acc in metrics['robust_acc']]
        ax1.plot(rounds, robust_acc, marker='s', linestyle='--', 
                label='Robust Accuracy', color='#d62728', linewidth=2)
    
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right')
    
    # Plot 2: AUC and F1
    ax2 = plt.subplot(gs[0, 1])
    
    if 'test_auc' in metrics:
        ax2.plot(rounds, metrics['test_auc'], marker='o', label='AUC',
                color='#ff7f0e', linewidth=2)
    
    if 'test_f1' in metrics:
        ax2.plot(rounds, metrics['test_f1'], marker='^', label='F1',
                color='#2ca02c', linewidth=2)
    
    if 'robust_auc' in metrics and metrics['robust_auc'][0] is not None:
        robust_auc = [auc if auc is not None else 0 for auc in metrics['robust_auc']]
        ax2.plot(rounds, robust_auc, marker='s', linestyle='--',
                label='Robust AUC', color='#9467bd', linewidth=2)
    
    ax2.set_title('AUC and F1 Metrics')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    
    # Plot 3: Training Loss
    ax3 = plt.subplot(gs[1, 0])
    
    if 'train_loss' in metrics:
        loss_rounds = list(range(1, len(metrics['train_loss']) + 1))
        ax3.plot(loss_rounds, metrics['train_loss'], marker='o', 
                color='#d62728', linewidth=2)
    
    if 'test_loss' in metrics:
        test_loss_rounds = list(range(1, len(metrics['test_loss']) + 1))
        ax3.plot(test_loss_rounds, metrics['test_loss'], marker='s',
                color='#9467bd', linestyle='--', linewidth=2,
                label='Test Loss')
        ax3.legend()
    
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Rounds')
    ax3.set_ylabel('Loss')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Client Trust Scores
    ax4 = plt.subplot(gs[1, 1])
    
    if 'client_trust' in metrics and len(metrics['client_trust']) > 0:
        trust_rounds = list(range(1, len(metrics['client_trust']) + 1))
        client_trust_data = metrics['client_trust']
        num_clients = len(client_trust_data[0])
        
        # Create colormap for clients
        cmap = plt.cm.tab10
        
        for client_idx in range(num_clients):
            client_trust = [round_trust[client_idx] for round_trust in client_trust_data]
            color = cmap(client_idx % 10)
            ax4.plot(trust_rounds, client_trust, marker='o', 
                    label=f'Client {client_idx+1}', linewidth=2, color=color)
        
        ax4.set_title('Client Trust Scores')
        ax4.set_xlabel('Rounds')
        ax4.set_ylabel('Trust Score')
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Plot 5: Precision and Recall
    ax5 = plt.subplot(gs[2, 0])
    
    if 'test_precision' in metrics:
        ax5.plot(rounds, metrics['test_precision'], marker='o',
                label='Precision', color='#e377c2', linewidth=2)
    
    if 'test_recall' in metrics:
        ax5.plot(rounds, metrics['test_recall'], marker='^',
                label='Recall', color='#7f7f7f', linewidth=2)
    
    ax5.set_title('Precision and Recall')
    ax5.set_xlabel('Rounds')
    ax5.set_ylabel('Score')
    ax5.set_ylim(0, 1.05)
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend(loc='lower right')
    
    # Plot 6: Robustness Gap (if available)
    ax6 = plt.subplot(gs[2, 1])
    
    if 'test_acc' in metrics and 'robust_acc' in metrics and metrics['robust_acc'][0] is not None:
        robustness_gap = []
        for i in range(len(metrics['test_acc'])):
            if i < len(metrics['robust_acc']) and metrics['robust_acc'][i] is not None:
                gap = metrics['test_acc'][i] - metrics['robust_acc'][i]
            else:
                gap = None
            robustness_gap.append(gap)
        
        # Filter out None values
        valid_indices = [i for i, g in enumerate(robustness_gap) if g is not None]
        valid_rounds = [rounds[i] for i in valid_indices]
        valid_gaps = [robustness_gap[i] for i in valid_indices]
        
        if valid_gaps:
            ax6.plot(valid_rounds, valid_gaps, marker='o', color='#8c564b',
                    linewidth=2, label='Accuracy Gap')
            ax6.fill_between(valid_rounds, [0] * len(valid_rounds), valid_gaps,
                           alpha=0.2, color='#8c564b')
            
            ax6.set_title('Robustness Gap (Test Acc - Robust Acc)')
            ax6.set_xlabel('Rounds')
            ax6.set_ylabel('Gap (smaller is better)')
            ax6.grid(True, linestyle='--', alpha=0.7)
            ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    else:
        # If no robustness data, show privacy budget or round time
        if 'round_time' in metrics:
            time_rounds = list(range(1, len(metrics['round_time']) + 1))
            ax6.plot(time_rounds, metrics['round_time'], marker='o',
                    color='#17becf', linewidth=2)
            ax6.set_title('Round Execution Time')
            ax6.set_xlabel('Rounds')
            ax6.set_ylabel('Time (seconds)')
            ax6.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle(f'Federated Learning Results: {experiment_name}', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, 'combined_results.png'), dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def create_robustness_plot(metrics, output_dir, fig_dpi, experiment_name):
    """Create detailed robustness analysis plot"""
    plt.figure(figsize=(12, 8))
    
    # Create a 2x1 subplot layout
    gs = GridSpec(2, 1, figure=plt.gcf(), height_ratios=[1.5, 1])
    
    # Plot 1: Accuracy comparison
    ax1 = plt.subplot(gs[0])
    rounds = list(range(1, len(metrics['test_acc']) + 1))
    
    # Filter out None values in robust metrics
    valid_robust_acc = np.array([acc if acc is not None else np.nan for acc in metrics['robust_acc']])
    valid_robust_auc = np.array([auc if auc is not None else np.nan for auc in metrics['robust_auc']])
    
    # Plot standard metrics
    ln1 = ax1.plot(rounds, metrics['test_acc'], marker='o', label='Test Accuracy',
                 linewidth=2.5, color='#1f77b4')
    ln2 = ax1.plot(rounds, metrics['test_auc'], marker='s', label='Test AUC',
                 linewidth=2.5, color='#ff7f0e')
    
    # Plot robust metrics with dashed lines
    ln3 = ax1.plot(rounds, valid_robust_acc, marker='o', linestyle='--', 
                 label='Robust Accuracy', linewidth=2.5, color='#1f77b4', alpha=0.7)
    ln4 = ax1.plot(rounds, valid_robust_auc, marker='s', linestyle='--',
                 label='Robust AUC', linewidth=2.5, color='#ff7f0e', alpha=0.7)
    
    # Highlight the gap with shading
    ax1.fill_between(rounds, metrics['test_acc'], valid_robust_acc, 
                    where=~np.isnan(valid_robust_acc),
                    color='#1f77b4', alpha=0.2, label='Accuracy Gap')
    
    ax1.fill_between(rounds, metrics['test_auc'], valid_robust_auc,
                    where=~np.isnan(valid_robust_auc),
                    color='#ff7f0e', alpha=0.2, label='AUC Gap')
    
    # Configure axes
    ax1.set_title(f'Attack Impact Analysis\n{experiment_name}')
    ax1.set_xlabel('Federated Learning Rounds')
    ax1.set_ylabel('Performance Score')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create combined legend
    lines = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', ncol=2)
    
    # Plot 2: Direct robustness gap
    ax2 = plt.subplot(gs[1])
    
    # Calculate accuracy and AUC gaps
    acc_gap = metrics['test_acc'] - valid_robust_acc
    auc_gap = metrics['test_auc'] - valid_robust_auc
    
    # Create bar chart for gaps
    bar_width = 0.35
    index = np.arange(len(rounds))
    
    # Plot only valid (non-NaN) gaps
    valid_acc_indices = ~np.isnan(acc_gap)
    valid_auc_indices = ~np.isnan(auc_gap)
    
    if np.any(valid_acc_indices):
        ax2.bar(index[valid_acc_indices] - bar_width/2, 
               acc_gap[valid_acc_indices], bar_width,
               label='Accuracy Gap', color='#1f77b4', alpha=0.8)
    
    if np.any(valid_auc_indices):
        ax2.bar(index[valid_auc_indices] + bar_width/2, 
               auc_gap[valid_auc_indices], bar_width,
               label='AUC Gap', color='#ff7f0e', alpha=0.8)
    
    # Configure axes
    ax2.set_title('Robustness Gap (smaller is better)')
    ax2.set_xlabel('Federated Learning Rounds')
    ax2.set_ylabel('Gap Size')
    ax2.set_xticks(index)
    ax2.set_xticklabels(rounds)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add average gap annotation
    mean_acc_gap = np.nanmean(acc_gap)
    mean_auc_gap = np.nanmean(auc_gap)
    
    ax2.annotate(f'Avg Acc Gap: {mean_acc_gap:.4f}', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold', color='#1f77b4')
    
    ax2.annotate(f'Avg AUC Gap: {mean_auc_gap:.4f}', 
                xy=(0.02, 0.87), xycoords='axes fraction',
                fontsize=10, fontweight='bold', color='#ff7f0e')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_analysis.png'), dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def create_summary_visualization(metrics, output_dir, fig_dpi, experiment_name):
    """Create a publication-ready summary visualization"""
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid layout with trust score heatmap spanning right column
    gs = GridSpec(2, 3, figure=plt.gcf(), width_ratios=[1, 1, 1.2])
    
    # 1. Performance Summary (top left)
    ax1 = plt.subplot(gs[0, 0])
    
    # Calculate key statistics
    if 'test_acc' in metrics and metrics['test_acc']:
        final_acc = metrics['test_acc'][-1]
        max_acc = max(metrics['test_acc'])
        max_acc_round = metrics['test_acc'].index(max_acc) + 1
        
        # Calculate average of last 3 rounds for stability measure
        if len(metrics['test_acc']) >= 3:
            last_3_avg = np.mean(metrics['test_acc'][-3:])
            last_3_std = np.std(metrics['test_acc'][-3:])
        else:
            last_3_avg = final_acc
            last_3_std = 0
        
        # Create a horizontal bar chart for metrics
        metric_names = ['Final Accuracy', 'Max Accuracy', 'Avg Last 3']
        metric_values = [final_acc, max_acc, last_3_avg]
        
        # Sort by value
        sorted_indices = np.argsort(metric_values)
        metric_names = [metric_names[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]
        
        y_pos = np.arange(len(metric_names))
        ax1.barh(y_pos, metric_values, align='center', 
                color=plt.cm.Blues(np.linspace(0.5, 0.9, len(metric_names))))
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(metric_names)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_xlabel('Score')
        ax1.set_xlim(0, 1.0)
        
        # Add value labels
        for i, v in enumerate(metric_values):
            ax1.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        ax1.set_title('Performance Summary')
    
    # 2. Client Trust Analysis (top middle)
    ax2 = plt.subplot(gs[0, 1])
    
    if 'client_trust' in metrics and metrics['client_trust']:
        trust_data = np.array(metrics['client_trust'])
        num_clients = trust_data.shape[1]
        
        # Calculate statistics for each client
        client_names = [f'Client {i+1}' for i in range(num_clients)]
        mean_trust = np.mean(trust_data, axis=0)
        min_trust = np.min(trust_data, axis=0)
        
        # Sort clients by mean trust
        sort_idx = np.argsort(mean_trust)
        mean_trust = mean_trust[sort_idx]
        min_trust = min_trust[sort_idx]
        client_names = [client_names[i] for i in sort_idx]
        
        # Plot
        x = np.arange(len(client_names))
        width = 0.35
        
        ax2.bar(x - width/2, mean_trust, width, label='Mean Trust', color='#2ca02c')
        ax2.bar(x + width/2, min_trust, width, label='Min Trust', color='#d62728')
        
        ax2.set_ylabel('Trust Score')
        ax2.set_title('Client Trust Summary')
        ax2.set_xticks(x)
        ax2.set_xticklabels(client_names, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1.05)
    
    # 3. Trust Score Evolution Heatmap (right column)
    ax3 = plt.subplot(gs[:, 2])
    
    if 'client_trust' in metrics and metrics['client_trust']:
        # Create matrix for heatmap
        trust_matrix = np.array(metrics['client_trust'])
        
        # Create heatmap
        im = ax3.imshow(trust_matrix.T, cmap='RdYlGn', aspect='auto', 
                       interpolation='none', vmin=0.0, vmax=1.0)
        
        # Configure axes
        ax3.set_xlabel('Federated Learning Round')
        ax3.set_ylabel('Client ID')
        ax3.set_title('Trust Score Evolution')
        
        # Set tick frequency based on number of rounds
        num_rounds = trust_matrix.shape[0]
        tick_step = max(1, num_rounds // 10)
        ax3.set_xticks(np.arange(0, num_rounds, tick_step))
        ax3.set_xticklabels(np.arange(1, num_rounds+1, tick_step))
        
        ax3.set_yticks(np.arange(num_clients))
        ax3.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Trust Score')
        
        # Add annotations for suspicious patterns
        for client in range(num_clients):
            client_trust = trust_matrix[:, client]
            
            # Find significant drops in trust (potential attack points)
            for i in range(1, len(client_trust)):
                if client_trust[i] <= 0.3 and client_trust[i-1] >= 0.7:
                    ax3.plot(i, client, 'rx', markersize=8, markeredgewidth=2)
    
    # 4. Robustness Analysis (bottom left span 2)
    ax4 = plt.subplot(gs[1, :2])
    
    if 'test_acc' in metrics and 'robust_acc' in metrics and metrics['robust_acc'][0] is not None:
        rounds = list(range(1, len(metrics['test_acc']) + 1))
        
        # Filter out None values
        valid_robust_acc = [acc if acc is not None else np.nan for acc in metrics['robust_acc']]
        
        # Calculate accuracy gap
        acc_gap = [a - r if r is not None else np.nan for a, r in zip(metrics['test_acc'], metrics['robust_acc'])]
        
        # Line plot for accuracy gap
        ax4.plot(rounds, acc_gap, marker='o', linestyle='-', 
                color='#d62728', linewidth=2.5, label='Robustness Gap')
        
        # Add shading
        ax4.fill_between(rounds, [0] * len(rounds), acc_gap, 
                        alpha=0.2, color='#d62728')
        
        # Calculate statistics
        valid_gaps = [g for g in acc_gap if not np.isnan(g)]
        if valid_gaps:
            mean_gap = np.mean(valid_gaps)
            max_gap = np.max(valid_gaps)
            min_gap = np.min(valid_gaps)
            
            # Add annotation with statistics
            ax4.annotate(f'Mean Gap: {mean_gap:.3f}\nMax Gap: {max_gap:.3f}\nMin Gap: {min_gap:.3f}',
                        xy=(0.02, 0.85), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=9)
        
        ax4.set_title('Robustness Gap Analysis (Test Acc - Robust Acc)')
        ax4.set_xlabel('Federated Learning Rounds')
        ax4.set_ylabel('Gap Size (smaller is better)')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add overall title
    plt.suptitle(f'Experiment Summary: {experiment_name}', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, 'experiment_summary.png'), dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def analyze_client_behavior(metrics, output_dir, fig_dpi, experiment_name):
    """
    Create a detailed analysis of client behavior patterns
    Particularly useful for understanding attack patterns
    """
    if 'client_trust' not in metrics or not metrics['client_trust']:
        return
        
    # Convert trust data to numpy array
    trust_data = np.array(metrics['client_trust'])
    num_rounds, num_clients = trust_data.shape
    
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # 1. Trust Volatility Analysis (top left)
    ax1 = plt.subplot(gs[0, 0])
    
    # Calculate trust volatility (standard deviation of differences)
    volatility = []
    for client in range(num_clients):
        client_trust = trust_data[:, client]
        if len(client_trust) > 1:
            diffs = np.diff(client_trust)
            vol = np.std(diffs)
        else:
            vol = 0
        volatility.append(vol)
    
    # Create bar chart of volatility
    client_names = [f'Client {i+1}' for i in range(num_clients)]
    ax1.bar(client_names, volatility, color=plt.cm.viridis(np.linspace(0, 0.8, num_clients)))
    
    # Add threshold line for suspicious volatility
    threshold = 0.2
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    
    ax1.set_title('Trust Volatility by Client')
    ax1.set_xlabel('Client')
    ax1.set_ylabel('Volatility (std of trust changes)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add annotation for potentially malicious clients
    suspicious = [i for i, v in enumerate(volatility) if v > threshold]
    if suspicious:
        suspicious_str = ', '.join([f'Client {i+1}' for i in suspicious])
        ax1.annotate(f'Suspicious: {suspicious_str}',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # 2. Trust Correlation Matrix (top right)
    ax2 = plt.subplot(gs[0, 1])
    
    # Calculate correlation between client trust patterns
    corr_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            corr_matrix[i, j] = np.corrcoef(trust_data[:, i], trust_data[:, j])[0, 1]
    
    # Create heatmap
    im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add client labels
    ax2.set_xticks(np.arange(num_clients))
    ax2.set_yticks(np.arange(num_clients))
    ax2.set_xticklabels(client_names)
    ax2.set_yticklabels(client_names)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Correlation')
    
    ax2.set_title('Client Trust Pattern Correlation')
    
    # Highlight potential collusion (high positive correlation)
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            if corr_matrix[i, j] > 0.8:  # High correlation threshold
                ax2.plot(j, i, 'rs', markersize=8)  # Mark with red square
    
    # 3. Client Trust Timeline (bottom left)
    ax3 = plt.subplot(gs[1, 0])
    
    # Create a detailed timeline plot
    rounds = np.arange(1, num_rounds + 1)
    
    # Plot each client's trust
    for client in range(num_clients):
        ax3.plot(rounds, trust_data[:, client], label=f'Client {client+1}',
                linewidth=2, marker='o' if volatility[client] > threshold else None)
    
    ax3.set_title('Client Trust Evolution')
    ax3.set_xlabel('Federated Learning Round')
    ax3.set_ylabel('Trust Score')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Add trust threshold lines
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
    
    # Add legend with smaller font
    ax3.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 4. Attack Pattern Detection (bottom right)
    ax4 = plt.subplot(gs[1, 1])
    
    # Define patterns to look for
    patterns = {
        'sharp_drop': [],      # Sudden trust drop
        'oscillation': [],     # Trust oscillating up and down
        'always_low': [],      # Consistently low trust
        'recovery': []         # Recovery after drop
    }
    
    for client in range(num_clients):
        client_trust = trust_data[:, client]
        
        # Pattern 1: Sharp drop
        for i in range(1, len(client_trust)):
            if client_trust[i] < client_trust[i-1] - 0.4:  # Drop of 0.4 or more
                patterns['sharp_drop'].append(client)
                break
        
        # Pattern 2: Oscillation
        if len(client_trust) >= 4:
            diffs = np.diff(client_trust)
            direction_changes = np.sum(diffs[1:] * diffs[:-1] < 0)  # Count sign changes
            if direction_changes >= len(diffs) / 2.5:  # Frequent direction changes
                patterns['oscillation'].append(client)
        
        # Pattern 3: Always low
        if np.mean(client_trust) < 0.4 and np.max(client_trust) < 0.6:
            patterns['always_low'].append(client)
        
        # Pattern 4: Recovery
        for i in range(2, len(client_trust)):
            if (client_trust[i-1] < 0.3 and client_trust[i] > 0.7 and 
                client_trust[i-2] > 0.7):
                patterns['recovery'].append(client)
                break
    
    # Create pattern presence matrix
    pattern_matrix = np.zeros((len(patterns), num_clients))
    for i, (pattern, clients) in enumerate(patterns.items()):
        for client in clients:
            pattern_matrix[i, client] = 1
    
    # Create heatmap
    im = ax4.imshow(pattern_matrix, cmap='binary', aspect='auto')
    
    # Add labels
    ax4.set_yticks(np.arange(len(patterns)))
    ax4.set_yticklabels([p.replace('_', ' ').title() for p in patterns.keys()])
    ax4.set_xticks(np.arange(num_clients))
    ax4.set_xticklabels(client_names)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add counts
    for i, pattern in enumerate(patterns.keys()):
        ax4.text(-0.5, i, f"({len(patterns[pattern])})", va='center', ha='right')
    
    ax4.set_title('Suspicious Behavior Patterns')
    
    # Add overall title
    plt.suptitle(f'Client Behavior Analysis: {experiment_name}', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, 'client_behavior_analysis.png'), dpi=fig_dpi, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import sys
    
    # Check if a directory is provided as argument
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        create_results_visualization(results_dir, output_dir, high_dpi=True)
    else:
        print("Usage: python create_plots.py <results_dir> [output_dir]")
        print("Example: python create_plots.py ../results/experiment1 ../results/experiment1/plots")