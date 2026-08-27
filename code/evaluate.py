import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import pandas as pd
import os
import json
from tqdm import tqdm
import time
import seaborn as sns

def evaluate_model(model, test_loader, device):
    """
    Evaluate a model on a test dataset
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    # Track predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []  # Store raw logits for debugging
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            # Store raw logits for analysis
            all_logits.extend(outputs.cpu().numpy())
            
            # Get predictions and probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Dynamic threshold selection instead of fixed 0.5
            # Find optimal threshold based on validation set performance
            # For now, use 0.5 but in a real implementation, you'd tune this
            preds = (probs > 0.5).astype(np.float32)
            
            # Store predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)
    
    # Print diagnostic information
    unique_preds = np.unique(all_preds, return_counts=True)
    unique_labels = np.unique(all_labels, return_counts=True)
    
    print(f"Prediction distribution: {dict(zip(unique_preds[0], unique_preds[1]))}")
    print(f"Label distribution: {dict(zip(unique_labels[0], unique_labels[1]))}")
    print(f"Logits range: [{np.min(all_logits):.4f}, {np.max(all_logits):.4f}], Mean: {np.mean(all_logits):.4f}")
    print(f"Probabilities range: [{np.min(all_probs):.4f}, {np.max(all_probs):.4f}], Mean: {np.mean(all_probs):.4f}")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Robust handling of case when all predictions are the same class
    if len(np.unique(all_preds)) < 2:
        print("WARNING: All predictions are the same class!")
        # If all predictions are the same, calculate metrics appropriately
        if np.unique(all_preds)[0] == 0:  # All negative predictions
            precision = 0  # No positive predictions, so precision is 0
            # Recall is 0 if there are any positive labels, 1 if there are none
            recall = 0 if np.sum(all_labels) > 0 else 1
            f1 = 0  # F1 is 0 with precision=0
        else:  # All positive predictions
            # Precision is the proportion of true positives
            precision = np.sum(all_labels) / len(all_labels)
            recall = 1  # All positives were found
            f1 = 2 * precision / (precision + 1) if precision > 0 else 0
    else:
        # Normal calculation when we have both positive and negative predictions
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
    
    # AUC calculation with robust handling of edge cases
    if len(np.unique(all_labels)) < 2:
        print("WARNING: Only one class present in ground truth!")
        auc = 0.5  # Default value when only one class is present
    else:
        try:
            # Try to calculate AUC
            auc = roc_auc_score(all_labels, all_probs)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            # Check if we have extreme probability values
            if np.all(np.isclose(all_probs, all_probs[0])):
                print("All probability scores are identical!")
                auc = 0.5
            else:
                # Try a different approach - add small noise to break ties
                perturbed_probs = all_probs + np.random.normal(0, 1e-5, size=len(all_probs))
                try:
                    auc = roc_auc_score(all_labels, perturbed_probs)
                    print(f"AUC calculated with perturbed probabilities: {auc:.4f}")
                except:
                    auc = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'logits': all_logits  # Include raw logits for further analysis
    }
    
    return metrics

def evaluate_attack_impact(models, test_loader, attacked_test_loader, device, labels=None):
    """
    Evaluate the impact of attacks on model performance
    
    Args:
        models: Dictionary of models to evaluate
        test_loader: DataLoader with clean test data
        attacked_test_loader: DataLoader with attacked test data
        device: Device to run evaluation on
        labels: Labels for the models
        
    Returns:
        results: Dictionary with evaluation results
    """
    if labels is None:
        labels = list(models.keys())
    
    results = {}
    
    for name, model in models.items():
        # Evaluate on clean data
        clean_metrics = evaluate_model(model, test_loader, device)
        
        # Evaluate on attacked data
        attacked_metrics = evaluate_model(model, attacked_test_loader, device)
        
        # Store results
        results[name] = {
            'clean': clean_metrics,
            'attacked': attacked_metrics,
            'robustness_gap': clean_metrics['accuracy'] - attacked_metrics['accuracy']
        }
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Bar width
    bar_width = 0.35
    
    # Positions for bars
    indices = np.arange(len(models))
    
    # Plot bars
    plt.bar(indices - bar_width/2, [results[name]['clean']['accuracy'] for name in models], 
            bar_width, label='Clean Accuracy')
    plt.bar(indices + bar_width/2, [results[name]['attacked']['accuracy'] for name in models], 
            bar_width, label='Attacked Accuracy')
    
    # Labels and title
    plt.xlabel('Defense Method')
    plt.ylabel('Accuracy')
    plt.title('Attack Impact on Different Defense Methods')
    plt.xticks(indices, [name.replace('_', ' ').upper() for name in models.keys()])
    plt.legend()
    plt.grid(axis='y')
    
    # Save plot
    os.makedirs('../results/attack_impact', exist_ok=True)
    plt.savefig('../results/attack_impact/attack_impact_comparison.png')
    plt.close()
    
    return results

def evaluate_privacy_utility_tradeoff(models, test_loaders, privacy_budgets, device, 
                                    metric='accuracy', title=None):
    """
    Evaluate the privacy-utility tradeoff
    
    Args:
        models: List of models with different privacy settings
        test_loaders: List of test DataLoaders
        privacy_budgets: List of epsilon values for each model
        device: Device to run evaluation on
        metric: Metric to use for utility ('accuracy', 'f1', 'auc')
        title: Title for the plot
        
    Returns:
        results: DataFrame with privacy-utility results
    """
    if len(models) != len(privacy_budgets):
        raise ValueError("Number of models must match number of privacy budgets")
    
    if len(models) != len(test_loaders) and len(test_loaders) != 1:
        raise ValueError("Number of test loaders must be 1 or match number of models")
    
    # If only one test loader is provided, use it for all models
    if len(test_loaders) == 1:
        test_loaders = [test_loaders[0]] * len(models)
    
    results = []
    
    for i, (model, loader, epsilon) in enumerate(zip(models, test_loaders, privacy_budgets)):
        # Evaluate model
        metrics = evaluate_model(model, loader, device)
        
        # Store results
        results.append({
            'model_idx': i,
            'epsilon': epsilon,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'auc': metrics['auc']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Sort by privacy budget
    results_df = results_df.sort_values('epsilon')
    
    # Plot privacy-utility curve
    utility_metric = results_df[metric].values
    privacy_values = results_df['epsilon'].values
    
    plt.plot(privacy_values, utility_metric, marker='o', linestyle='-')
    
    # Labels and title
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel(f'Utility ({metric.capitalize()})')
    if title:
        plt.title(title)
    else:
        plt.title('Privacy-Utility Tradeoff')
    plt.grid(True)
    
    # Save plot
    os.makedirs('../results/privacy_utility', exist_ok=True)
    plt.savefig(f'../results/privacy_utility/privacy_utility_{metric}.png')
    plt.close()
    
    return results_df

def plot_metrics_over_rounds(metrics_history, save_dir=None, title_prefix='', figsize=(15, 10)):
    """
    Plot metrics over federated learning rounds
    
    Args:
        metrics_history: Dictionary with metrics history
        save_dir: Directory to save plots (if None, plots are displayed)
        title_prefix: Prefix for plot titles
        figsize: Figure size
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Number of rounds
    num_rounds = len(metrics_history['accuracy']) if 'accuracy' in metrics_history else 0
    if num_rounds == 0:
        print("Warning: No metrics to plot")
        return
    
    rounds = list(range(1, num_rounds + 1))
    
    # Create a 2x2 plot
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Accuracy and Loss
    if 'accuracy' in metrics_history:
        ax1 = axs[0, 0]
        ax1.plot(rounds, metrics_history['accuracy'], marker='o', label='Accuracy')
        if 'robust_accuracy' in metrics_history:
            ax1.plot(rounds, metrics_history['robust_accuracy'], marker='s', 
                    linestyle='--', label='Robust Accuracy')
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{title_prefix}Accuracy')
        ax1.grid(True)
        ax1.legend()
    
    # Plot 2: Loss
    if 'loss' in metrics_history:
        ax2 = axs[0, 1]
        ax2.plot(rounds, metrics_history['loss'], marker='o', label='Loss')
        ax2.set_xlabel('Rounds')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{title_prefix}Loss')
        ax2.grid(True)
        ax2.legend()
    
    # Plot 3: AUC and F1
    if 'auc' in metrics_history or 'f1' in metrics_history:
        ax3 = axs[1, 0]
        if 'auc' in metrics_history:
            ax3.plot(rounds, metrics_history['auc'], marker='o', label='AUC')
        if 'f1' in metrics_history:
            ax3.plot(rounds, metrics_history['f1'], marker='s', label='F1')
        ax3.set_xlabel('Rounds')
        ax3.set_ylabel('Score')
        ax3.set_title(f'{title_prefix}AUC and F1')
        ax3.grid(True)
        ax3.legend()
    
    # Plot 4: Privacy Budget
    if 'privacy_budget' in metrics_history:
        ax4 = axs[1, 1]
        privacy_budget = np.array(metrics_history['privacy_budget'])
        cumulative_budget = np.cumsum(privacy_budget)
        ax4.plot(rounds, cumulative_budget, marker='o', label='Cumulative ε')
        ax4.set_xlabel('Rounds')
        ax4.set_ylabel('Privacy Budget (ε)')
        ax4.set_title(f'{title_prefix}Privacy Budget')
        ax4.grid(True)
        ax4.legend()
    elif 'trust_scores' in metrics_history and metrics_history['trust_scores']:
        # If no privacy budget but trust scores available, plot those instead
        ax4 = axs[1, 1]
        num_clients = len(metrics_history['trust_scores'][0])
        for client_idx in range(num_clients):
            client_trust = [round_scores[client_idx] for round_scores in metrics_history['trust_scores']]
            ax4.plot(rounds, client_trust, marker='o', label=f'Client {client_idx+1}')
        ax4.set_xlabel('Rounds')
        ax4.set_ylabel('Trust Score')
        ax4.set_title(f'{title_prefix}Client Trust Scores')
        ax4.grid(True)
        ax4.legend()
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics_over_rounds.png'))
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm, classes=None, save_path=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class names
        save_path: Path to save the plot
        title: Title for the plot
    """
    if classes is None:
        if cm.shape[0] == 2:
            classes = ['Negative', 'Positive']
        else:
            classes = [str(i) for i in range(cm.shape[0])]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, auc_score, save_path=None, title='ROC Curve'):
    """
    Plot a ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(precision, recall, save_path=None, title='Precision-Recall Curve'):
    """
    Plot a precision-recall curve
    
    Args:
        precision: Precision values
        recall: Recall values
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_model_detailed(model, test_loader, device, save_dir=None, title_prefix=''):
    """
    Perform detailed evaluation of a model including ROC and precision-recall curves
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        save_dir: Directory to save plots
        title_prefix: Prefix for plot titles
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get predictions and metrics
    metrics = evaluate_model(model, test_loader, device)
    
    # Extract values
    y_true = metrics['labels']
    y_pred = metrics['predictions']
    y_prob = metrics['probabilities']
    
    # Create classification report
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
    metrics['report'] = report
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics['precision_curve'] = precision
    metrics['recall_curve'] = recall
    
    # Plot confusion matrix
    if save_dir:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            classes=['Negative', 'Positive'],
            save_path=os.path.join(save_dir, f'{title_prefix}confusion_matrix.png'),
            title=f'{title_prefix}Confusion Matrix'
        )
    
    # Plot ROC curve
    if save_dir:
        plot_roc_curve(
            fpr, tpr, metrics['auc'],
            save_path=os.path.join(save_dir, f'{title_prefix}roc_curve.png'),
            title=f'{title_prefix}ROC Curve'
        )
    
    # Plot precision-recall curve
    if save_dir:
        plot_precision_recall_curve(
            precision, recall,
            save_path=os.path.join(save_dir, f'{title_prefix}precision_recall_curve.png'),
            title=f'{title_prefix}Precision-Recall Curve'
        )
    
    # Save metrics as JSON
    if save_dir:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'report': report
        }
        
        with open(os.path.join(save_dir, f'{title_prefix}metrics.json'), 'w') as f:
            json.dump(json_metrics, f, indent=2)
    
    return metrics

def compare_models(models, test_loader, device, model_names=None, save_dir=None, title='Model Comparison'):
    """
    Compare multiple models on the same test dataset
    
    Args:
        models: Dictionary or list of models to compare
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        model_names: Names for the models (if models is a list)
        save_dir: Directory to save plots
        title: Title for the comparison plots
        
    Returns:
        results: DataFrame with comparison results
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Convert list to dictionary if needed
    if isinstance(models, list):
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(len(models))]
        models = {name: model for name, model in zip(model_names, models)}
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, test_loader, device)
        results[name] = metrics
    
    # Create comparison table
    comparison = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'AUC': []
    }
    
    for name, metrics in results.items():
        comparison['Model'].append(name)
        comparison['Accuracy'].append(metrics['accuracy'])
        comparison['Precision'].append(metrics['precision'])
        comparison['Recall'].append(metrics['recall'])
        comparison['F1'].append(metrics['f1'])
        comparison['AUC'].append(metrics['auc'])
    
    results_df = pd.DataFrame(comparison)
    
    # Save results
    if save_dir:
        results_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Get metrics to plot
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    # Plot grouped bar chart
    bar_width = 0.15
    x = np.arange(len(results_df))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + (i - len(metrics_to_plot)/2 + 0.5) * bar_width, 
                results_df[metric], bar_width, label=metric)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, results_df['Model'])
    plt.legend()
    plt.grid(axis='y')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
        plt.close()
    else:
        plt.show()
    
    return results_df

def evaluate_attack_impact_detailed(models, test_loader, attacked_test_loader, 
                                  device, save_dir=None, model_names=None,
                                  title='Attack Impact Analysis'):
    """
    Detailed evaluation of attack impact on different models
    
    Args:
        models: Dictionary or list of models to evaluate
        test_loader: DataLoader with clean test data
        attacked_test_loader: DataLoader with attacked test data
        device: Device to run evaluation on
        save_dir: Directory to save results
        model_names: Names for the models (if models is a list)
        title: Title for the plots
        
    Returns:
        results: Dictionary with detailed evaluation results
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Convert list to dictionary if needed
    if isinstance(models, list):
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(len(models))]
        models = {name: model for name, model in zip(model_names, models)}
    
    # Evaluate each model on clean and attacked data
    results = {}
    for name, model in models.items():
        model_dir = os.path.join(save_dir, name) if save_dir else None
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Evaluate on clean data
        clean_metrics = evaluate_model_detailed(
            model, test_loader, device,
            save_dir=model_dir, title_prefix='Clean_'
        )
        
        # Evaluate on attacked data
        attacked_metrics = evaluate_model_detailed(
            model, attacked_test_loader, device,
            save_dir=model_dir, title_prefix='Attacked_'
        )
        
        # Calculate robustness gap
        robustness_gap = {
            'accuracy': clean_metrics['accuracy'] - attacked_metrics['accuracy'],
            'precision': clean_metrics['precision'] - attacked_metrics['precision'],
            'recall': clean_metrics['recall'] - attacked_metrics['recall'],
            'f1': clean_metrics['f1'] - attacked_metrics['f1'],
            'auc': clean_metrics['auc'] - attacked_metrics['auc']
        }
        
        # Store results
        results[name] = {
            'clean': clean_metrics,
            'attacked': attacked_metrics,
            'robustness_gap': robustness_gap
        }
    
    # Create comparison table
    comparison = {
        'Model': [],
        'Clean Accuracy': [],
        'Attacked Accuracy': [],
        'Accuracy Gap': [],
        'Clean AUC': [],
        'Attacked AUC': [],
        'AUC Gap': [],
        'Clean F1': [],
        'Attacked F1': [],
        'F1 Gap': []
    }
    
    for name, metrics in results.items():
        comparison['Model'].append(name)
        comparison['Clean Accuracy'].append(metrics['clean']['accuracy'])
        comparison['Attacked Accuracy'].append(metrics['attacked']['accuracy'])
        comparison['Accuracy Gap'].append(metrics['robustness_gap']['accuracy'])
        comparison['Clean AUC'].append(metrics['clean']['auc'])
        comparison['Attacked AUC'].append(metrics['attacked']['auc'])
        comparison['AUC Gap'].append(metrics['robustness_gap']['auc'])
        comparison['Clean F1'].append(metrics['clean']['f1'])
        comparison['Attacked F1'].append(metrics['attacked']['f1'])
        comparison['F1 Gap'].append(metrics['robustness_gap']['f1'])
    
    results_df = pd.DataFrame(comparison)
    
    # Save results
    if save_dir:
        results_df.to_csv(os.path.join(save_dir, 'attack_impact_comparison.csv'), index=False)
    
    # Create visualization
    # Accuracy comparison
    plt.figure(figsize=(12, 8))
    
    bar_width = 0.35
    x = np.arange(len(results_df))
    
    plt.bar(x - bar_width/2, results_df['Clean Accuracy'], bar_width, label='Clean')
    plt.bar(x + bar_width/2, results_df['Attacked Accuracy'], bar_width, label='Attacked')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xticks(x, results_df['Model'])
    plt.legend()
    plt.grid(axis='y')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'attack_impact_accuracy.png'))
        plt.close()
    
    # AUC comparison
    plt.figure(figsize=(12, 8))
    
    plt.bar(x - bar_width/2, results_df['Clean AUC'], bar_width, label='Clean')
    plt.bar(x + bar_width/2, results_df['Attacked AUC'], bar_width, label='Attacked')
    
    plt.xlabel('Model')
    plt.ylabel('AUC')
    plt.title(f'{title} - AUC')
    plt.xticks(x, results_df['Model'])
    plt.legend()
    plt.grid(axis='y')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'attack_impact_auc.png'))
        plt.close()
    
    # Robustness gap comparison
    plt.figure(figsize=(12, 8))
    
    plt.bar(x - bar_width, results_df['Accuracy Gap'], bar_width, label='Accuracy Gap')
    plt.bar(x, results_df['AUC Gap'], bar_width, label='AUC Gap')
    plt.bar(x + bar_width, results_df['F1 Gap'], bar_width, label='F1 Gap')
    
    plt.xlabel('Model')
    plt.ylabel('Robustness Gap')
    plt.title(f'{title} - Robustness Gap (smaller is better)')
    plt.xticks(x, results_df['Model'])
    plt.legend()
    plt.grid(axis='y')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'attack_impact_gap.png'))
        plt.close()
    
    return results

if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Run the main script to train and evaluate the federated learning models.")