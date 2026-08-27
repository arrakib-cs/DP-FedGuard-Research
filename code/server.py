import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import logging
import json
from datetime import datetime
from dataset import evaluate_with_threshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dp_fedguard.log")
    ]
)
logger = logging.getLogger("dp_fedguard_server")

# Import our own modules
from models import create_model, model_distance
from client import train_client
from defenses import (
    calculate_client_trust_scores, dp_fedguard_aggregate, fedavg, 
    fed_median, trimmed_mean, krum, adaptive_trust_aggregation, 
    create_defense_ensemble, dynamic_dp_noise
)

class FederatedServer:
    """
    Enhanced server class for federated learning with DP-FedGuard
    """
    def __init__(self, model_name='densenet121', num_classes=1, device='cuda',
                defense_type='dp_fedguard', use_dp=True, results_dir='../results',
                noise_multiplier=1.0, max_grad_norm=1.0, early_stopping=False):
        """
        Initialize the federated server
        
        Args:
            model_name: Name of the model architecture to use
            num_classes: Number of output classes (1 for binary)
            device: Device to run on (cuda/cpu)
            defense_type: Type of defense to use
                        'none': Standard FedAvg
                        'dp_fedguard': DP-FedGuard with trust-based aggregation
                        'median': Coordinate-wise median
                        'trimmed_mean': Trimmed mean aggregation
                        'krum': Krum or Multi-Krum
                        'adaptive': Adaptive defense strategy
            use_dp: Whether to use differential privacy
            results_dir: Directory to save results
            noise_multiplier: Base noise multiplier for DP
            max_grad_norm: Maximum gradient norm for DP
            early_stopping: Whether to use early stopping
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.defense_type = defense_type
        self.use_dp = use_dp
        self.results_dir = results_dir
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.early_stopping = early_stopping
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize the global model
        self.global_model = self.initialize_model()
        logger.info(f"Initialized global model: {model_name}")
        
        # Track metrics
        self.metrics = defaultdict(list)
        
        # Track privacy budget
        self.privacy_budget = 0.0
        
        # Set default parameters
        self.default_params = {
            'local_epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm
        }
        
        # For early stopping
        self.best_accuracy = 0.0
        self.best_model_state = None
        self.patience = 5
        self.patience_counter = 0
        
        # Initialize client history for trust tracking
        self.client_history = None
    
    def initialize_model(self):
        """Initialize and return a new global model"""
        model = create_model(model_name=self.model_name, num_classes=self.num_classes)
        model = model.to(self.device)
        return model
    
    def reset_global_model(self):
        """Reset the global model to initial state"""
        self.global_model = self.initialize_model()
        logger.info("Global model reset to initial state")
    
    def get_client_sample_sizes(self, client_train_loaders):
        """Get the number of samples for each client"""
        return [len(loader.dataset) for loader in client_train_loaders]
    
    def get_client_label_distributions(self, client_train_loaders):
        """
        Get the label distribution for each client
        
        Returns:
            list of dicts with information about label distribution
        """
        distributions = []
        
        for loader in client_train_loaders:
            # Count positive and negative samples
            positive_count = 0
            total_count = 0
            
            # Sample up to 100 batches to estimate distribution (for efficiency)
            sample_count = 0
            max_samples = 100
            
            for _, labels in loader:
                positive_count += torch.sum(labels).item()
                total_count += len(labels)
                sample_count += 1
                
                if sample_count >= max_samples:
                    break
            
            # Calculate positive ratio
            positive_ratio = positive_count / total_count if total_count > 0 else 0.5
            
            distributions.append({
                'positive_count': positive_count,
                'total_count': total_count,
                'positive_ratio': positive_ratio
            })
        
        return distributions
    
    def run_federated_round(self, client_train_loaders, attacker_ids=None, 
                          epochs=None, lr=None, noise_multiplier=None,
                          max_grad_norm=None, verbose=True):
        """
        Run a single round of federated learning with enhanced defenses
        
        Args:
            client_train_loaders: List of training data loaders for each client
            attacker_ids: List of client IDs that are attackers
            epochs: Number of local training epochs
            lr: Learning rate for client training
            noise_multiplier: DP noise multiplier
            max_grad_norm: Maximum gradient norm for DP
            verbose: Whether to print progress information
            
        Returns:
            metrics: Dictionary with metrics for this round
        """
        start_time = time.time()
        
        # Use default parameters if not specified
        if epochs is None:
            epochs = self.default_params['local_epochs']
        if lr is None:
            lr = self.default_params['learning_rate']
        if noise_multiplier is None:
            noise_multiplier = self.default_params['noise_multiplier']
        if max_grad_norm is None:
            max_grad_norm = self.default_params['max_grad_norm']
        
        # Default: no attackers
        if attacker_ids is None:
            attacker_ids = []
        
        num_clients = len(client_train_loaders)
        
        if verbose:
            logger.info("--- Starting Federated Round ---")
            
        # Get client sample sizes for weighted averaging
        client_sample_sizes = self.get_client_sample_sizes(client_train_loaders)
        
        # Get label distributions for trust calculation
        label_distributions = self.get_client_label_distributions(client_train_loaders)
        
        # Calculate trust scores from previous round (use equal trust for first round)
        if len(self.metrics['client_trust']) > 0:
            prev_trust_scores = self.metrics['client_trust'][-1]
            current_round = len(self.metrics['client_trust']) + 1
        else:
            prev_trust_scores = [1.0] * num_clients
            current_round = 1
        
        # Calculate dynamic noise levels for clients based on trust
        if self.defense_type in ['dp_fedguard', 'adaptive'] and self.use_dp:
            noise_levels = dynamic_dp_noise(
                prev_trust_scores, 
                base_noise=noise_multiplier,
                max_noise=noise_multiplier * 3.0  # Higher noise for untrusted clients
            )
        else:
            noise_levels = [noise_multiplier] * num_clients
            
        if verbose:
            logger.info(f"Training {num_clients} clients...")
            
        # Train each client's model
        client_models = []
        client_losses = []
        client_epsilons = []
        client_metrics = []
        
        for client_id, train_loader in enumerate(client_train_loaders):
            # Create a copy of the global model for this client
            client_model = copy.deepcopy(self.global_model)
            
            # Special handling for attackers
            if client_id in attacker_ids:
                if verbose:
                    logger.info(f"  Client {client_id+1}: Attacker (poisoning model)")
                
                # Model poisoning attack
                from models import initialize_model_poisoning, perform_model_poisoning
                
                # Add model poisoning (more sophisticated poisoning)
                client_model = initialize_model_poisoning(
                    client_model, 
                    poisoning_type='targeted',
                    target_class=0  # Target normal class to cause false negatives
                )
                
                # Simulate training - attackers might use better optimization
                local_epochs = max(1, epochs)  # Full training for stronger attack
                model, metrics = train_client(
                    model=client_model,
                    data_loader=train_loader,
                    device=self.device,
                    epochs=local_epochs,
                    lr=lr * 2.0,  # Higher learning rate for stronger attack
                    use_dp=False,  # Attackers don't use DP
                    verbose=False
                )
                
                # Apply sophisticated poisoning after training
                poisoning_type = random.choice(['scale', 'targeted'])  # Randomly select attack type
                client_model = perform_model_poisoning(
                    model, 
                    self.global_model, 
                    poisoning_type=poisoning_type
                )
                
                # Fake metrics
                metrics = {
                    'loss': metrics['loss'],
                    'time': metrics['time'],
                    'epsilon': None,
                    'epochs': local_epochs
                }
            else:
                # Regular training for honest clients
                if verbose:
                    logger.debug(f"  Client {client_id+1}: Training with {len(train_loader.dataset)} samples, " +
                          f"noise={noise_levels[client_id]:.2f}")
                
                # Train the client with trust-based noise level
                client_model, metrics = train_client(
                    model=client_model,
                    data_loader=train_loader,
                    device=self.device,
                    epochs=epochs,
                    lr=lr,
                    use_dp=self.use_dp,
                    noise_multiplier=noise_levels[client_id],
                    max_grad_norm=max_grad_norm,
                    verbose=False
                )
            
            # Save client model and metrics
            client_models.append(client_model)
            client_losses.append(metrics['loss'])
            client_metrics.append(metrics)
            
            if metrics['epsilon'] is not None:
                client_epsilons.append(metrics['epsilon'])
        
        # Calculate enhanced trust scores with history tracking
        if self.client_history is None:
            # Initialize client history on first round
            self.client_history = {
                'update_norms': [[] for _ in range(num_clients)],
                'cosine_similarities': [[] for _ in range(num_clients)],
                'suspected_attacks': [0] * num_clients
            }
            
        trust_scores, self.client_history = calculate_client_trust_scores(
            client_models, 
            self.global_model, 
            label_distributions,
            prev_trust_scores,
            current_round,
            self.client_history
        )
        
        if verbose:
            logger.info("Client trust scores:")
            for i, trust in enumerate(trust_scores):
                status = "Attacker" if i in attacker_ids else "Honest"
                logger.info(f"  Client {i+1}: {trust:.4f} ({status})")
        
        # Aggregate client models based on defense type
        if self.defense_type == 'none':
            # Standard FedAvg
            self.global_model = fedavg(client_models, client_sample_sizes)
            logger.info("Aggregated with standard FedAvg")
            
        elif self.defense_type == 'dp_fedguard':
            # DP-FedGuard
            self.global_model = dp_fedguard_aggregate(
                client_models, 
                self.global_model, 
                trust_scores,
                aggregation_type='weighted_averaging',
                round_num=current_round
            )
            logger.info("Aggregated with DP-FedGuard (weighted_averaging)")
            
        elif self.defense_type == 'median':
            # Coordinate-wise median
            self.global_model = fed_median(client_models, self.global_model)
            logger.info("Aggregated with coordinate-wise median")
            
        elif self.defense_type == 'trimmed_mean':
            # Trimmed mean
            trim_ratio = 0.2  # Remove 20% from each end
            self.global_model = trimmed_mean(client_models, self.global_model, trim_ratio=trim_ratio)
            logger.info(f"Aggregated with trimmed mean (trim_ratio={trim_ratio})")
            
        elif self.defense_type == 'krum':
            # Krum
            num_attackers = len(attacker_ids) if attacker_ids else max(1, num_clients // 5)
            self.global_model = krum(
                client_models, 
                self.global_model, 
                num_attackers=num_attackers,
                multi_krum=True,
                num_to_select=max(1, num_clients - num_attackers)
            )
            logger.info(f"Aggregated with Multi-Krum (num_attackers={num_attackers})")
            
        elif self.defense_type == 'adaptive':
            # Adaptive defense strategy
            self.global_model = adaptive_trust_aggregation(
                client_models,
                self.global_model,
                trust_scores,
                round_num=current_round
            )
            logger.info("Aggregated with adaptive trust-based strategy")
            
        elif self.defense_type == 'ensemble':
            # Ensemble of methods
            self.global_model = dp_fedguard_aggregate(
                client_models,
                self.global_model,
                trust_scores,
                aggregation_type='ensemble',
                round_num=current_round
            )
            logger.info("Aggregated with ensemble defense")
            
        else:
            raise ValueError(f"Unknown defense type: {self.defense_type}")
        
        # Calculate total privacy spent (simplified)
        if client_epsilons:
            round_epsilon = max(client_epsilons)
            self.privacy_budget += round_epsilon
        else:
            round_epsilon = 0.0
        
        # Calculate round time
        round_time = time.time() - start_time
        
        # Collect round metrics
        round_metrics = {
            'train_loss': np.mean(client_losses),
            'client_trust': trust_scores,
            'privacy_spent': round_epsilon,
            'round_time': round_time
        }
        
        # Save metrics
        self.metrics['train_loss'].append(round_metrics['train_loss'])
        self.metrics['client_trust'].append(round_metrics['client_trust'])
        self.metrics['privacy_spent'].append(round_metrics['privacy_spent'])
        self.metrics['round_time'].append(round_metrics['round_time'])
        
        if verbose:
            logger.info(f"Round completed in {round_time:.2f} seconds")
            logger.info(f"Average client loss: {round_metrics['train_loss']:.4f}")
            if round_epsilon > 0:
                # Fix Unicode issue by using "epsilon" instead of "ε"
                logger.info(f"Privacy spent (epsilon): {round_epsilon:.2f} (total: {self.privacy_budget:.2f})")
            logger.info("-----------------------------------")
        
        return round_metrics
    
    def evaluate(self, test_loader, attacked_test_loader=None, verbose=True):
        """
        Evaluate the global model with detailed metrics
        
        Args:
            test_loader: DataLoader with test data
            attacked_test_loader: DataLoader with attacked test data for robustness evaluation
            verbose: Whether to print progress information
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        # Set up loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Evaluate on normal test data
        from models import evaluate_model
        test_metrics = evaluate_with_threshold(self.global_model, test_loader, self.device)

        
        # Save metrics
        self.metrics['test_loss'].append(test_metrics['loss'])
        self.metrics['test_acc'].append(test_metrics['accuracy'])
        self.metrics['test_auc'].append(test_metrics['auc'])
        self.metrics['test_f1'].append(test_metrics['f1'])
        self.metrics['test_precision'].append(test_metrics['precision'])
        self.metrics['test_recall'].append(test_metrics['recall'])
        
        # Evaluate on attacked test data if provided
        if attacked_test_loader is not None:
            robust_metrics = evaluate_model(self.global_model, attacked_test_loader, criterion, self.device)
            robust_acc = robust_metrics['accuracy']
            self.metrics['robust_acc'].append(robust_acc)
            self.metrics['robust_auc'].append(robust_metrics['auc'])
            self.metrics['robust_f1'].append(robust_metrics['f1'])
        elif 'robust_acc' in self.metrics and len(self.metrics['robust_acc']) > 0:
            # If no attacked data, use last known robust accuracy
            robust_acc = self.metrics['robust_acc'][-1]
            self.metrics['robust_auc'].append(self.metrics['robust_auc'][-1])
            self.metrics['robust_f1'].append(self.metrics['robust_f1'][-1])
        else:
            robust_acc = None
            self.metrics['robust_acc'].append(None)
            self.metrics['robust_auc'].append(None)
            self.metrics['robust_f1'].append(None)
        
        if verbose:
            logger.info("\n--- Evaluation Results ---")
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
            if robust_acc is not None:
                logger.info(f"Robust Accuracy: {robust_acc:.4f}")
                logger.info(f"Robust AUC: {self.metrics['robust_auc'][-1]:.4f}")
                logger.info(f"Robust F1: {self.metrics['robust_f1'][-1]:.4f}")
            logger.info("---------------------------")
        
        # Check for early stopping
        if self.early_stopping:
            current_acc = test_metrics['accuracy']
            
            if current_acc > self.best_accuracy:
                self.best_accuracy = current_acc
                self.best_model_state = copy.deepcopy(self.global_model.state_dict())
                self.patience_counter = 0
                logger.info(f"New best model with accuracy: {current_acc:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement for {self.patience_counter} rounds (patience: {self.patience})")
        
        return test_metrics
    
    def run_experiment(self, client_train_loaders, test_loader, attacked_test_loader=None,
                    num_rounds=10, epochs=None, attacker_ids=None, 
                    eval_interval=1, save_interval=None, verbose=True,
                    experiment_name="federated_experiment"):
        """
        Run a complete federated learning experiment with enhanced monitoring
        
        Args:
            client_train_loaders: List of training data loaders for each client
            test_loader: DataLoader with test data
            attacked_test_loader: DataLoader with attacked test data for robustness evaluation
            num_rounds: Number of federated rounds to run
            epochs: Number of local training epochs per round
            attacker_ids: List of client IDs that are attackers
            eval_interval: How often to evaluate the model (in rounds)
            save_interval: How often to save the model (in rounds)
            verbose: Whether to print progress information
            experiment_name: Name of the experiment
            
        Returns:
            metrics: Dictionary with all metrics from the experiment
        """
        # Reset metrics
        self.metrics = defaultdict(list)
        
        # Reset privacy budget
        self.privacy_budget = 0.0
        
        # Reset client history
        self.client_history = None
        
        # Reset early stopping variables
        self.best_accuracy = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Create experiment directory
        experiment_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save experiment configuration
        config = {
            "model": self.model_name,
            "defense": self.defense_type,
            "use_dp": self.use_dp,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "num_rounds": num_rounds,
            "num_clients": len(client_train_loaders),
            "num_attackers": len(attacker_ids) if attacker_ids else 0,
            "early_stopping": self.early_stopping,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Initial evaluation
        if verbose:
            logger.info("\n=== Starting Experiment ===")
            logger.info(f"Defense: {self.defense_type}")
            logger.info(f"Differential Privacy: {'Enabled' if self.use_dp else 'Disabled'}")
            logger.info(f"Number of Clients: {len(client_train_loaders)}")
            logger.info(f"Attackers: {attacker_ids if attacker_ids else 'None'}")
            logger.info("\nInitial evaluation:")
            
        self.evaluate(test_loader, attacked_test_loader, verbose)
        
        # Training loop
        stopped_early = False
        for round_idx in range(num_rounds):
            if verbose:
                logger.info(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            # Run one federated round
            round_metrics = self.run_federated_round(
                client_train_loaders=client_train_loaders,
                attacker_ids=attacker_ids,
                epochs=epochs,
                verbose=verbose
            )
            
            # Evaluate at specified intervals
            if (round_idx + 1) % eval_interval == 0 or round_idx == num_rounds - 1:
                if verbose:
                    logger.info(f"\nEvaluating at round {round_idx + 1}:")
                self.evaluate(test_loader, attacked_test_loader, verbose)
            
            # Save model at specified intervals
            if save_interval and (round_idx + 1) % save_interval == 0:
                save_path = os.path.join(experiment_dir, f"model_round_{round_idx+1}.pth")
                torch.save(self.global_model.state_dict(), save_path)
                if verbose:
                    logger.info(f"Model saved to {save_path}")
            
            # Check for early stopping
            if self.early_stopping and self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {round_idx + 1} rounds")
                # Restore best model
                if self.best_model_state is not None:
                    self.global_model.load_state_dict(self.best_model_state)
                    logger.info(f"Restored best model with accuracy: {self.best_accuracy:.4f}")
                stopped_early = True
                break
        
        # Final evaluation
        if verbose:
            logger.info("\n=== Experiment Complete ===")
            if stopped_early:
                logger.info(f"Stopped early after {round_idx + 1} rounds")
            logger.info("Final evaluation:")
        final_metrics = self.evaluate(test_loader, attacked_test_loader, verbose)
        
        # Plot and save results
        self.plot_results(experiment_dir, experiment_name)
        
        # Save final model
        final_model_path = os.path.join(experiment_dir, "final_model.pth")
        torch.save(self.global_model.state_dict(), final_model_path)
        
        # If using early stopping, also save best model
        if self.early_stopping and self.best_model_state is not None:
            best_model_path = os.path.join(experiment_dir, "best_model.pth")
            torch.save(self.best_model_state, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
        
        # Save metrics
        self.save_metrics(experiment_dir)
        
        if verbose:
            logger.info(f"\nExperiment results saved to {experiment_dir}")
        
        return self.metrics
    
    def save_metrics(self, save_dir):
        """Save metrics to files"""
        # Convert metrics to serializable format
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], list):
                # Handle lists of lists (e.g., client_trust)
                serializable_metrics[key] = [[float(v) if v is not None else None for v in sublist] for sublist in value]
            elif isinstance(value, list):
                # Handle simple lists
                serializable_metrics[key] = [float(v) if v is not None else None for v in value]
            else:
                serializable_metrics[key] = value
        
        # Save metrics as JSON
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        # Also save as CSV for easier analysis
        csv_path = os.path.join(save_dir, "metrics.csv")
        with open(csv_path, 'w') as f:
            # Write header
            scalar_metrics = ['train_loss', 'test_loss', 'test_acc', 'test_auc', 'test_f1', 
                             'test_precision', 'test_recall', 'robust_acc', 'robust_auc', 
                             'robust_f1', 'privacy_spent', 'round_time']
            f.write("round," + ",".join(scalar_metrics) + "\n")
            
            # Write data
            num_rounds = len(self.metrics['train_loss'])
            for i in range(num_rounds):
                values = [str(i+1)]
                for metric in scalar_metrics:
                    if metric in self.metrics and i < len(self.metrics[metric]):
                        values.append(str(self.metrics[metric][i] if self.metrics[metric][i] is not None else "NA"))
                    else:
                        values.append("NA")
                f.write(",".join(values) + "\n")
    
    def plot_results(self, save_dir, title_prefix=""):
        """Plot and save experiment results with enhanced visualizations"""
        # Ensure all arrays have the same length
        num_rounds = len(self.metrics['train_loss'])
        rounds = list(range(1, num_rounds + 1))

        # Truncate all metric arrays to the same length
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = self.metrics[key][:num_rounds]
        
        # Create enhanced visualizations
        self.plot_accuracy_curves(rounds, save_dir, title_prefix)
        self.plot_loss_curves(rounds, save_dir, title_prefix)
        self.plot_auc_f1_curves(rounds, save_dir, title_prefix)
        self.plot_trust_scores(rounds, save_dir, title_prefix)
        self.plot_privacy_budget(rounds, save_dir, title_prefix)
        self.plot_performance_vs_privacy(save_dir, title_prefix)
        
        # Create combined results plot
        self.plot_combined_results(save_dir, title_prefix)
    
    def plot_accuracy_curves(self, rounds, save_dir, title_prefix=""):
        """Plot accuracy curves"""
        plt.figure(figsize=(10, 6))
        
        if self.metrics['test_acc']:
            plt.plot(rounds, self.metrics['test_acc'], marker='o', 
                    label='Test Accuracy', linewidth=2)
            
            if 'robust_acc' in self.metrics and self.metrics['robust_acc'][0] is not None:
                plt.plot(rounds, self.metrics['robust_acc'], marker='s', 
                        label='Robust Accuracy', linewidth=2, linestyle='--')
                
                # Also plot gap between test and robust accuracy
                accuracy_gap = [test - robust if robust is not None else None 
                               for test, robust in zip(self.metrics['test_acc'], self.metrics['robust_acc'])]
                plt.plot(rounds, accuracy_gap, marker='^', 
                        label='Accuracy Gap', linewidth=1.5, linestyle=':', alpha=0.7)
        
        plt.title(f"{title_prefix} Accuracy Metrics" if title_prefix else "Accuracy Metrics")
        plt.xlabel('Federated Rounds')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'), dpi=300)
        plt.close()
    
    def plot_loss_curves(self, rounds, save_dir, title_prefix=""):
        """Plot loss curves"""
        plt.figure(figsize=(10, 6))
        
        if self.metrics['train_loss'] and self.metrics['test_loss']:
            plt.plot(rounds, self.metrics['train_loss'], marker='o', 
                    label='Training Loss', linewidth=2)
            plt.plot(rounds, self.metrics['test_loss'], marker='s', 
                    label='Test Loss', linewidth=2)
        
        plt.title(f"{title_prefix} Loss Curves" if title_prefix else "Loss Curves")
        plt.xlabel('Federated Rounds')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300)
        plt.close()
    
    def plot_auc_f1_curves(self, rounds, save_dir, title_prefix=""):
        """Plot AUC and F1 curves"""
        plt.figure(figsize=(10, 6))
        
        if self.metrics['test_auc'] and self.metrics['test_f1']:
            plt.plot(rounds, self.metrics['test_auc'], marker='o', 
                    label='Test AUC', linewidth=2)
            plt.plot(rounds, self.metrics['test_f1'], marker='s', 
                    label='Test F1', linewidth=2)
            
            if 'robust_auc' in self.metrics and self.metrics['robust_auc'][0] is not None:
                plt.plot(rounds, self.metrics['robust_auc'], marker='d', 
                        label='Robust AUC', linewidth=2, linestyle='--')
            
            if 'robust_f1' in self.metrics and self.metrics['robust_f1'][0] is not None:
                plt.plot(rounds, self.metrics['robust_f1'], marker='x', 
                        label='Robust F1', linewidth=2, linestyle='--')
        
        plt.title(f"{title_prefix} AUC and F1 Metrics" if title_prefix else "AUC and F1 Metrics")
        plt.xlabel('Federated Rounds')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'auc_f1_curves.png'), dpi=300)
        plt.close()
    
    def plot_trust_scores(self, rounds, save_dir, title_prefix=""):
        """Plot client trust scores"""
        if not self.metrics['client_trust'] or not self.metrics['client_trust'][0]:
            return
        
        plt.figure(figsize=(10, 6))
        
        num_clients = len(self.metrics['client_trust'][0])
        for client_idx in range(num_clients):
            client_trust = [round_scores[client_idx] for round_scores in self.metrics['client_trust']]
            plt.plot(rounds, client_trust, marker='o', 
                    label=f'Client {client_idx+1}', linewidth=2)
        
        plt.title(f"{title_prefix} Client Trust Scores" if title_prefix else "Client Trust Scores")
        plt.xlabel('Federated Rounds')
        plt.ylabel('Trust Score')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'trust_scores.png'), dpi=300)
        plt.close()
    
    def plot_privacy_budget(self, rounds, save_dir, title_prefix=""):
        """Plot privacy budget used"""
        if not self.metrics['privacy_spent'] or not self.metrics['privacy_spent'][0]:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Cumulative privacy budget
        cumulative_budget = np.cumsum([eps for eps in self.metrics['privacy_spent'] if eps is not None])
        
        # Per-round privacy budget
        per_round_budget = [eps for eps in self.metrics['privacy_spent'] if eps is not None]
        
        plt.plot(rounds[:len(cumulative_budget)], cumulative_budget, marker='o', 
                label='Cumulative Privacy Budget', linewidth=2)
        plt.bar(rounds[:len(per_round_budget)], per_round_budget, alpha=0.3, 
               label='Per-Round Privacy Budget')
        
        plt.title(f"{title_prefix} Privacy Budget Used (epsilon)" if title_prefix else "Privacy Budget Used (epsilon)")
        plt.xlabel('Federated Rounds')
        plt.ylabel('Privacy Budget (epsilon)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'privacy_budget.png'), dpi=300)
        plt.close()
    
    def plot_performance_vs_privacy(self, save_dir, title_prefix=""):
        """Plot performance vs privacy trade-off"""
        if not self.metrics['test_acc'] or not self.metrics['privacy_spent'] or not self.metrics['privacy_spent'][0]:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Cumulative privacy budget
        cumulative_budget = np.cumsum([eps for eps in self.metrics['privacy_spent'] if eps is not None])
        
        # Test metrics for corresponding rounds
        accuracy = self.metrics['test_acc'][:len(cumulative_budget)]
        
        if self.metrics['test_auc'] and len(self.metrics['test_auc']) >= len(cumulative_budget):
            auc = self.metrics['test_auc'][:len(cumulative_budget)]
            plt.plot(cumulative_budget, auc, marker='s', 
                    label='AUC vs Privacy', linewidth=2)
        
        plt.plot(cumulative_budget, accuracy, marker='o', 
                label='Accuracy vs Privacy', linewidth=2)
        
        plt.title(f"{title_prefix} Performance vs Privacy Trade-off" if title_prefix else "Performance vs Privacy Trade-off")
        plt.xlabel('Cumulative Privacy Budget (epsilon)')
        plt.ylabel('Performance Metric')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'performance_vs_privacy.png'), dpi=300)
        plt.close()
    
    def plot_combined_results(self, save_dir, title_prefix=""):
        """Create a combined results plot"""
        plt.figure(figsize=(15, 10))
        
        # Create a 2x2 plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Accuracy and Loss
        ax1 = axs[0, 0]
        rounds = list(range(1, len(self.metrics['test_acc']) + 1))
        
        ax1.plot(rounds, self.metrics['test_acc'], marker='o', label='Test Accuracy')
        if 'robust_acc' in self.metrics and self.metrics['robust_acc'][0] is not None:
            ax1.plot(rounds, self.metrics['robust_acc'], marker='s', 
                    linestyle='--', label='Robust Accuracy')
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{title_prefix} Accuracy' if title_prefix else 'Accuracy')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Loss
        ax2 = axs[0, 1]
        ax2.plot(rounds, self.metrics['train_loss'], marker='o', label='Train Loss')
        ax2.plot(rounds, self.metrics['test_loss'], marker='s', label='Test Loss')
        ax2.set_xlabel('Rounds')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{title_prefix} Loss' if title_prefix else 'Loss')
        ax2.grid(True)
        ax2.legend()
        
        # Plot 3: AUC and F1
        ax3 = axs[1, 0]
        ax3.plot(rounds, self.metrics['test_auc'], marker='o', label='AUC')
        ax3.plot(rounds, self.metrics['test_f1'], marker='s', label='F1')
        ax3.set_xlabel('Rounds')
        ax3.set_ylabel('Score')
        ax3.set_title(f'{title_prefix} AUC and F1' if title_prefix else 'AUC and F1')
        ax3.grid(True)
        ax3.legend()
        
        # Plot 4: Privacy Budget or Trust Scores
        ax4 = axs[1, 1]
        if self.metrics['privacy_spent'] and self.metrics['privacy_spent'][0]:
            privacy_budget = np.array([eps if eps is not None else 0 for eps in self.metrics['privacy_spent']])
            cumulative_budget = np.cumsum(privacy_budget)
            ax4.plot(rounds, cumulative_budget, marker='o', label='Cumulative Privacy Budget (epsilon)')
            ax4.set_xlabel('Rounds')
            ax4.set_ylabel('Privacy Budget (epsilon)')
            ax4.set_title(f'{title_prefix} Privacy Budget' if title_prefix else 'Privacy Budget')
        elif self.metrics['client_trust'] and self.metrics['client_trust'][0]:
            # If no privacy budget but trust scores available, plot those instead
            num_clients = len(self.metrics['client_trust'][0])
            for client_idx in range(num_clients):
                client_trust = [round_scores[client_idx] for round_scores in self.metrics['client_trust']]
                ax4.plot(rounds, client_trust, marker='o', label=f'Client {client_idx+1}')
            ax4.set_xlabel('Rounds')
            ax4.set_ylabel('Trust Score')
            ax4.set_title(f'{title_prefix} Client Trust Scores' if title_prefix else 'Client Trust Scores')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, 'results.png'), dpi=300)
        plt.close()

def create_defense_comparison_plots(results, save_dir, title="Defense Comparison"):
    """
    Create visualizations comparing different defense methods
    
    Args:
        results: Dictionary with results for each defense
        save_dir: Directory to save plots
        title: Title for the plots
    """
    # Extract defense names and metrics
    defenses = list(results.keys())
    accuracy = [results[d]['final_acc'] for d in defenses]
    auc = [results[d]['final_auc'] for d in defenses]
    f1 = [results[d]['final_f1'] for d in defenses]
    robust_acc = [results[d]['final_robust_acc'] if results[d]['final_robust_acc'] is not None else 0 for d in defenses]
    robustness_gap = [results[d]['robustness_gap'] if results[d]['robustness_gap'] is not None else 0 for d in defenses]
    privacy = [results[d]['privacy_budget'] for d in defenses]
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(defenses))
    
    plt.bar(index - bar_width/2, accuracy, bar_width, label='Test Accuracy', color='royalblue')
    plt.bar(index + bar_width/2, robust_acc, bar_width, label='Robust Accuracy', color='darkorange')
    
    plt.xlabel('Defense Method')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy Comparison')
    plt.xticks(index, [d.upper() for d in defenses])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(accuracy):
        plt.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    for i, v in enumerate(robust_acc):
        if v > 0:  # Only add labels for non-zero values
            plt.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # Plot AUC and F1 comparison
    plt.figure(figsize=(12, 8))
    
    plt.bar(index - bar_width/2, auc, bar_width, label='AUC', color='purple')
    plt.bar(index + bar_width/2, f1, bar_width, label='F1 Score', color='teal')
    
    plt.xlabel('Defense Method')
    plt.ylabel('Score')
    plt.title(f'{title} - AUC and F1 Comparison')
    plt.xticks(index, [d.upper() for d in defenses])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(f1):
        plt.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'auc_f1_comparison.png'), dpi=300)
    plt.close()
    
    # Plot robustness gap (smaller is better)
    plt.figure(figsize=(12, 8))
    
    # Use a color gradient based on gap size (smaller gap = better = greener)
    colors = []
    for gap in robustness_gap:
        # Normalize gap to [0, 1] range for color mapping
        normalized_gap = min(1.0, gap / 0.5) if gap is not None else 1.0
        # Interpolate between green (small gap) and red (large gap)
        r = normalized_gap
        g = 1 - normalized_gap * 0.7  # Keep some green even for large gaps
        b = 0.2
        colors.append((r, g, b))
    
    plt.bar(index, robustness_gap, color=colors)
    
    plt.xlabel('Defense Method')
    plt.ylabel('Robustness Gap (smaller is better)')
    plt.title(f'{title} - Robustness Gap')
    plt.xticks(index, [d.upper() for d in defenses])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(robustness_gap):
        if v > 0:  # Only add labels for non-zero values
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_gap.png'), dpi=300)
    plt.close()
    
    # Plot privacy budget
    plt.figure(figsize=(12, 8))
    
    plt.bar(index, privacy, color='darkred')
    
    plt.xlabel('Defense Method')
    plt.ylabel('Privacy Budget Used (epsilon)')
    plt.title(f'{title} - Privacy Budget Comparison')
    plt.xticks(index, [d.upper() for d in defenses])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(privacy):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'privacy_comparison.png'), dpi=300)
    plt.close()

def run_privacy_utility_analysis(model_name='densenet121', num_classes=1, device='cuda',
                               defense_type='dp_fedguard', base_noise=1.0, 
                               client_train_loaders=None, test_loader=None,
                               results_dir='../results'):
    """
    Run an analysis of privacy-utility tradeoff with varying privacy levels
    
    Args:
        model_name: Model architecture to use
        num_classes: Number of output classes
        device: Device to run on (cuda/cpu)
        defense_type: Type of defense to use
        base_noise: Base noise multiplier for DP
        client_train_loaders: List of training DataLoaders for each client
        test_loader: DataLoader with test data
        results_dir: Directory to save results
        
    Returns:
        results: DataFrame with privacy-utility results
    """
    # Import necessary modules
    import pandas as pd
    import logging
    
    # Get logger
    logger = logging.getLogger("dp_fedguard_server")
    
    # Set device
    device = torch.device(device)
    
    # Create experiment directory
    experiment_dir = os.path.join(results_dir, f"privacy_utility_{defense_type}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Define noise multipliers to test
    noise_multipliers = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Results list
    results = []
    
    logger.info("\n=== Starting Privacy-Utility Analysis ===")
    logger.info(f"Defense: {defense_type}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Noise multipliers: {noise_multipliers}")
    
    for noise_multiplier in noise_multipliers:
        logger.info(f"\n--- Testing noise multiplier: {noise_multiplier} ---")
        
        # Create server with this noise level
        server = FederatedServer(
            model_name=model_name,
            num_classes=num_classes,
            device=device,
            defense_type=defense_type,
            use_dp=True,
            noise_multiplier=noise_multiplier,
            results_dir=os.path.join(experiment_dir, f"noise_{noise_multiplier}")
        )
        
        # Run a shorter experiment (fewer rounds)
        num_rounds = 5
        
        metrics = server.run_experiment(
            client_train_loaders=client_train_loaders,
            test_loader=test_loader,
            num_rounds=num_rounds,
            epochs=1,
            eval_interval=1,
            verbose=True,
            experiment_name=f"noise_{noise_multiplier}"
        )
        
        # Extract key metrics
        result = {
            'noise_multiplier': noise_multiplier,
            'privacy_budget': sum(eps for eps in metrics['privacy_spent'] if eps is not None),
            'accuracy': metrics['test_acc'][-1],
            'auc': metrics['test_auc'][-1],
            'f1': metrics['test_f1'][-1]
        }
        
        results.append(result)
        
        logger.info(f"Results for noise={noise_multiplier}:")
        logger.info(f"  Privacy Budget (epsilon): {result['privacy_budget']:.2f}")
        logger.info(f"  Accuracy: {result['accuracy']:.4f}")
        logger.info(f"  AUC: {result['auc']:.4f}")
        logger.info(f"  F1: {result['f1']:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(experiment_dir, 'privacy_utility_results.csv'), index=False)
    
    # Create privacy-utility curve
    plt.figure(figsize=(12, 8))
    
    # Sort by privacy budget
    results_df = results_df.sort_values('privacy_budget')
    
    # Plot curves for different metrics
    plt.plot(results_df['privacy_budget'], results_df['accuracy'], 
            marker='o', label='Accuracy', linewidth=2, color='royalblue')
    plt.plot(results_df['privacy_budget'], results_df['auc'], 
            marker='s', label='AUC', linewidth=2, color='darkorange')
    plt.plot(results_df['privacy_budget'], results_df['f1'], 
            marker='^', label='F1', linewidth=2, color='forestgreen')
    
    # Add noise multiplier annotations
    for i, row in results_df.iterrows():
        plt.annotate(f"noise={row['noise_multiplier']}", 
                   (row['privacy_budget'], row['accuracy']),
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Privacy Budget (epsilon)')
    plt.ylabel('Performance Metric')
    plt.title(f'Privacy-Utility Tradeoff for {defense_type.upper()}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(experiment_dir, 'privacy_utility_curve.png'), dpi=300)
    plt.close()
    
    return results_df

if __name__ == "__main__":
    # Simple test for the server functions
    logger.info("Testing FederatedServer class...")
    
    # Test creating a server
    server = FederatedServer(
        model_name='densenet121',
        num_classes=1,
        device='cpu',  # Use CPU for testing
        defense_type='dp_fedguard',
        use_dp=True
    )
    
    logger.info("Server initialized successfully")
    logger.info(f"Model: {server.model_name}")
    logger.info(f"Defense: {server.defense_type}")
    logger.info(f"DP enabled: {server.use_dp}")
    
    logger.info("Server implementation test completed!")