import os
import torch
import numpy as np
import random
import argparse
import json
import logging
from datetime import datetime
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy
import sys
from collections import defaultdict
from opacus.validators import ModuleValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dp_fedguard.log")
    ]
)
logger = logging.getLogger("dp_fedguard")

# Import our modules
from dataset import prepare_dataset, create_non_iid_distribution, create_federated_datasets, simulate_attacks
from models import create_model
from client import train_client
from defenses import (
    calculate_client_trust_scores, dp_fedguard_aggregate, fedavg, 
    fed_median, trimmed_mean, krum, adaptive_trust_aggregation,
    enhanced_outlier_detection, adaptive_aggregation_framework
)
from evaluate import evaluate_model, plot_metrics_over_rounds
from enhanced_dp_fedguard import EnhancedDPFedGuard

# Check if we need to fix BatchNorm layers for DP compatibility
def ensure_dp_compatible_model(model, model_name, use_dp=False):
    """
    Ensure the model is compatible with DP training by replacing BatchNorm layers
    with GroupNorm layers if needed
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
        use_dp: Whether DP will be used (only convert if True)
        
    Returns:
        model: DP-compatible model
    """
    if not use_dp:
        # No need to convert if not using DP
        return model
    
    # Check if model is already DP-compatible
    is_valid = ModuleValidator.validate(model)
    
    if is_valid:
        logger.info(f"Model is already DP-compatible")
        return model
    
    # Model needs conversion. Check if pre-converted model exists
    fixed_model_path = f"fixed_{model_name}.pth"
    if os.path.exists(fixed_model_path):
        logger.info(f"Loading pre-converted DP-compatible model from {fixed_model_path}")
        try:
            model.load_state_dict(torch.load(fixed_model_path))
            
            # Verify model is now DP-compatible
            is_valid = ModuleValidator.validate(model)
            if is_valid:
                logger.info("Successfully loaded DP-compatible model")
                return model
            else:
                logger.warning("Loaded model is still not DP-compatible")
        except Exception as e:
            logger.error(f"Error loading pre-converted model: {e}")
    
    # Try to convert model using ModuleValidator.fix
    logger.info("Converting model to be DP-compatible using ModuleValidator.fix...")
    try:
        model = ModuleValidator.fix(model)
        is_valid = ModuleValidator.validate(model)
        
        if is_valid:
            logger.info("Successfully converted model with ModuleValidator.fix")
            # Save the converted model for future use
            torch.save(model.state_dict(), fixed_model_path)
            logger.info(f"Saved converted model to {fixed_model_path}")
            return model
    except Exception as e:
        logger.error(f"Error converting model with ModuleValidator.fix: {e}")
    
    # If we get here, we failed to convert or load a compatible model
    logger.error("Failed to create a DP-compatible model.")
    logger.error("Please run fix_batchnorm.py first to create a compatible model.")
    
    return model

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_experiment_name(args):
    """Create a descriptive name for the experiment"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    defense = args.defense_type
    dp_str = "with_dp" if args.use_dp else "no_dp"
    attack_str = f"attack_{args.attack_type}" if args.attack_fraction > 0 else "no_attack"
    balanced_str = "balanced" if args.balanced else "imbalanced"
    
    name = f"{defense}_{dp_str}_{attack_str}_{balanced_str}_{timestamp}"
    
    # Add additional descriptive elements
    if args.use_dp:
        name += f"_noise{args.noise_multiplier}"
    
    if args.attack_fraction > 0:
        name += f"_att{args.attack_fraction}"
    
    return name

def log_experiment_config(config, save_dir):
    """Save experiment configuration to a file"""
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Also log to console
    logger.info("Experiment configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

def main(args):
    """Main function to run the DP-FedGuard experiments with enhanced techniques"""
    # Start timing the experiment
    start_time = time.time()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment name and directory
    experiment_name = create_experiment_name(args) if not args.experiment_name else args.experiment_name
    experiment_dir = os.path.join(args.results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Log configuration
    config = vars(args)
    log_experiment_config(config, experiment_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("\n=== Preparing Dataset ===")
    # Prepare the NIH Chest X-ray dataset
    df = prepare_dataset(args.csv_path, args.img_dir, 
                       subset_size=args.subset_size if not args.debug else 200,
                       verbose=True, balanced=args.balanced)
    
    # Create non-IID data distribution
    client_dfs = create_non_iid_distribution(
        df,
        num_clients=args.num_clients,
        distribution_type=args.distribution_type
    )
    
    # Create federated datasets with balanced sampling
    client_train_loaders, client_test_loaders, global_test_loader = create_federated_datasets(
        client_dfs,
        test_size=args.test_size,
        batch_size=args.batch_size,
        balanced_sampling=args.balanced_sampling  # Use balanced sampling if requested
    )
    
    # Simulate attacks if needed
    if args.attack_fraction > 0:
        logger.info(f"\n=== Simulating {args.attack_type} Attack ===")
        
        # Determine which clients are attackers
        num_attackers = max(1, int(args.num_clients * args.attack_fraction))
        attacker_ids = random.sample(range(args.num_clients), num_attackers)
        
        logger.info(f"Selected {num_attackers} attackers: Clients {[i+1 for i in attacker_ids]}")
        
        # Simulate attacks
        attacked_train_loaders = simulate_attacks(
            client_dfs,
            client_train_loaders,
            attack_type=args.attack_type,
            attacker_ids=attacker_ids,
            attack_fraction=1.0  # Attack 100% of the attacker's data
        )
        
        # Create an attacked test dataset for robustness evaluation
        attacked_test_loader = None
        if args.attack_type == 'label_flipping':
            # Get the test dataset
            test_dataset = global_test_loader.dataset
            
            # Clone the dataset
            from copy import deepcopy
            attacked_test_dataset = deepcopy(test_dataset)
            
            # For our custom ChestXrayDataset, we need to flip the labels directly in the dataframe
            for i in range(len(attacked_test_dataset.dataframe)):
                # Flip the label (0 -> 1, 1 -> 0)
                original_label = attacked_test_dataset.dataframe.iloc[i]['label']
                attacked_test_dataset.dataframe.iloc[i, attacked_test_dataset.dataframe.columns.get_loc('label')] = 1.0 - original_label
            
            # Create a dataloader with the attacked test dataset
            attacked_test_loader = DataLoader(
                attacked_test_dataset,
                batch_size=global_test_loader.batch_size,
                shuffle=False,
                num_workers=global_test_loader.num_workers
            )
            
            logger.info(f"Created attacked test dataset for robustness evaluation")
    else:
        attacked_train_loaders = client_train_loaders
        attacker_ids = []
        attacked_test_loader = None
    
    # Initialize the enhanced DP-FedGuard
    logger.info(f"\n=== Initializing Enhanced {args.defense_type.upper()} Defense ===")
    
    dp_fedguard = EnhancedDPFedGuard(
        num_clients=len(client_train_loaders),
        device=device,
        use_dp=args.use_dp,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm
    )
    
    # Create global model
    logger.info(f"Creating global model: {args.model}")
    global_model = create_model(args.model)
    
    # If using DP, ensure the model is compatible by converting BatchNorm to GroupNorm
    if args.use_dp:
        logger.info("Ensuring model is compatible with differential privacy...")
        global_model = ensure_dp_compatible_model(global_model, args.model, args.use_dp)
    
    # Move model to the appropriate device
    global_model = global_model.to(device)
    
    # Metrics tracking
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
        'test_auc': [],
        'test_f1': [],
        'test_precision': [],
        'test_recall': [],
        'robust_acc': [],
        'robust_auc': [],
        'robust_f1': [],
        'client_trust': [],
        'round_time': [],
        'reputation_scores': [],  # Track client reputation over time
        'suspected_attacks': []   # Track detected attack attempts
    }
    
    # Initial evaluation
    logger.info("\n=== Starting Experiment ===")
    logger.info(f"Defense: {args.defense_type}")
    logger.info(f"Differential Privacy: {'Enabled' if args.use_dp else 'Disabled'}")
    logger.info(f"Number of Clients: {args.num_clients}")
    logger.info(f"Attackers: {attacker_ids}")
    logger.info("\nInitial evaluation:")
    
    from dataset import evaluate_with_threshold
    initial_metrics = evaluate_with_threshold(global_model, global_test_loader, device)
    
    # Store initial metrics
    metrics['test_loss'].append(initial_metrics['loss'])
    metrics['test_acc'].append(initial_metrics['accuracy'])
    metrics['test_auc'].append(initial_metrics['auc'])
    metrics['test_f1'].append(initial_metrics['f1'])
    metrics['test_precision'].append(initial_metrics['precision'])
    metrics['test_recall'].append(initial_metrics['recall'])
    
    # Initial robust evaluation
    if attacked_test_loader:
        initial_robust_metrics = evaluate_with_threshold(global_model, attacked_test_loader, device)
        metrics['robust_acc'].append(initial_robust_metrics['accuracy'])
        metrics['robust_auc'].append(initial_robust_metrics['auc'])
        metrics['robust_f1'].append(initial_robust_metrics['f1'])
    else:
        metrics['robust_acc'].append(None)
        metrics['robust_auc'].append(None)
        metrics['robust_f1'].append(None)
    
    # Initialize client reputation (start with neutral reputation)
    reputation_scores = [0.5] * args.num_clients
    metrics['reputation_scores'].append(reputation_scores)
    metrics['suspected_attacks'].append([0] * args.num_clients)
    metrics['client_trust'].append([1.0] * args.num_clients)  # Initial full trust
    
    logger.info(f"Test Loss: {initial_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {initial_metrics['accuracy']:.4f}")
    logger.info(f"Test AUC: {initial_metrics['auc']:.4f}")
    logger.info(f"Test F1 Score: {initial_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {initial_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {initial_metrics['recall']:.4f}")
    
    if attacked_test_loader:
        logger.info(f"Robust Accuracy: {initial_robust_metrics['accuracy']:.4f}")
        logger.info(f"Robust AUC: {initial_robust_metrics['auc']:.4f}")
        logger.info(f"Robust F1: {initial_robust_metrics['f1']:.4f}")
    logger.info("---------------------------")
    
    # Create client history tracker for detection analysis
    client_history = {
        'update_norms': [[] for _ in range(args.num_clients)],
        'cosine_similarities': [[] for _ in range(args.num_clients)],
        'suspected_attacks': [0] * args.num_clients,
        'trust_scores': [[] for _ in range(args.num_clients)],
        'layer_norms': [{} for _ in range(args.num_clients)]
    }
    
    # For early stopping
    best_accuracy = initial_metrics['accuracy']
    best_model_state = copy.deepcopy(global_model.state_dict())
    patience_counter = 0
    
    # Training loop
    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"\n=== Round {round_idx + 1}/{args.rounds} ===")
        logger.info("--- Starting Federated Round ---")
        
        # Calculate dynamic noise levels based on trust scores
        noise_levels = None
        if args.use_dp and round_idx > 0:
            # Only apply dynamic noise after first round when we have trust scores
            prev_trust_scores = metrics['client_trust'][-1]
            noise_levels = []
            
            for client_idx, trust in enumerate(prev_trust_scores):
                # Dynamic noise scaling: lower trust → higher noise
                if trust > 0.8:  # High trust
                    noise = args.noise_multiplier * 0.7  # Reduce noise
                elif trust < 0.3:  # Low trust
                    noise = args.noise_multiplier * 2.0  # Increase noise
                else:  # Medium trust
                    # Linear interpolation
                    factor = (trust - 0.3) / 0.5  # Maps 0.3->0.0, 0.8->1.0
                    noise = args.noise_multiplier * (2.0 - 1.3 * factor)
                
                noise_levels.append(noise)
                
            logger.info(f"Using dynamic noise levels: min={min(noise_levels):.2f}, max={max(noise_levels):.2f}")
        
        # Train each client's model
        logger.info(f"Training {len(attacked_train_loaders)} clients...")
        client_models = []
        client_losses = []
        client_epsilons = []
        client_accuracies = []
        client_metrics = []
        
        for client_idx, train_loader in enumerate(attacked_train_loaders):
            # Progress indicator
            logger.info(f"  Training client {client_idx+1}/{len(attacked_train_loaders)}...")
            
            # Create a copy of the global model for this client
            client_model = copy.deepcopy(global_model)
            
            # Handle attackers differently
            if client_idx in attacker_ids:
                logger.info(f"  Client {client_idx+1}: Attacker (poisoning model)")
                
                # Model poisoning attack
                from models import initialize_model_poisoning, perform_model_poisoning
                
                # Add model poisoning (more sophisticated poisoning)
                client_model = initialize_model_poisoning(
                    client_model, 
                    poisoning_type='targeted',
                    target_class=0  # Target normal class to cause false negatives
                )
                
                # Simulate training - attackers might use better optimization
                local_epochs = max(1, args.epochs)  # Full training for stronger attack
                model, metrics_data = train_client(
                    model=client_model,
                    data_loader=train_loader,
                    device=device,
                    epochs=local_epochs,
                    lr=args.lr * 2.0,  # Higher learning rate for stronger attack
                    use_dp=False,  # Attackers don't use DP
                    verbose=False
                )
                
                # Apply sophisticated poisoning after training
                poisoning_type = random.choice(['scale', 'targeted'])  # Randomly select attack type
                if round_idx > 0:
                    # Adaptive attack: if previously detected (low trust), use more subtle attack
                    prev_trust = metrics['client_trust'][-1][client_idx]
                    if prev_trust < 0.3:
                        poisoning_type = 'subtle'  # More subtle attack to avoid detection
                        logger.info(f"    Attacker adapting strategy due to low trust ({prev_trust:.2f})")
                
                client_model = perform_model_poisoning(
                    model, 
                    global_model, 
                    poisoning_type=poisoning_type
                )
                
                # Fake metrics
                metrics_data = {
                    'loss': metrics_data['loss'],
                    'time': metrics_data['time'],
                    'epsilon': None,
                    'epochs': local_epochs
                }
            else:
                # Regular training for honest clients
                if noise_levels is not None:
                    client_noise = noise_levels[client_idx]
                    logger.debug(f"  Client {client_idx+1}: Using noise multiplier {client_noise:.2f}")
                else:
                    client_noise = args.noise_multiplier
                
                # Train the client with trust-based noise level
                client_model, metrics_data = train_client(
                    model=client_model,
                    data_loader=train_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    use_dp=args.use_dp,
                    noise_multiplier=client_noise if noise_levels is not None else args.noise_multiplier,
                    max_grad_norm=args.max_grad_norm,
                    verbose=False
                )
            
            # Save client model and metrics
            client_models.append(client_model)
            client_losses.append(metrics_data['loss'])
            client_metrics.append(metrics_data)
            
            # Evaluate client model performance on local test data
            if client_idx < len(client_test_loaders):
                client_test_metrics = evaluate_with_threshold(
                    client_model, 
                    client_test_loaders[client_idx],
                    device
                )
                client_accuracies.append(client_test_metrics['accuracy'])
            else:
                client_accuracies.append(0.0)
            
            if metrics_data['epsilon'] is not None:
                client_epsilons.append(metrics_data['epsilon'])
        
        # Calculate enhanced trust scores with history tracking
        trust_scores, client_history = calculate_client_trust_scores(
            client_models, 
            global_model, 
            None,  # No label distributions for now
            metrics['client_trust'][-1] if round_idx > 0 else None,
            round_idx + 1,
            client_history
        )
        
        logger.info("Client trust scores:")
        for i, trust in enumerate(trust_scores):
            attacker_status = "Attacker" if i in attacker_ids else "Honest"
            logger.info(f"  Client {i+1}: {trust:.4f} ({attacker_status})")
            
            # Update reputation scores
            old_rep = reputation_scores[i]
            if trust > 0.7:  # Good behavior
                reputation_scores[i] = min(1.0, old_rep + 0.05)
            elif trust < 0.3:  # Bad behavior
                reputation_scores[i] = max(0.0, old_rep - 0.1)
                client_history['suspected_attacks'][i] += 1
        
        # Store updated metrics
        metrics['client_trust'].append(trust_scores.copy())
        metrics['reputation_scores'].append(reputation_scores.copy())
        metrics['suspected_attacks'].append(client_history['suspected_attacks'].copy())
        
        # Determine effective attacker detection
        detected_attackers = [i for i, trust in enumerate(trust_scores) if trust < 0.3]
        true_positives = len([i for i in detected_attackers if i in attacker_ids])
        false_positives = len([i for i in detected_attackers if i not in attacker_ids])
        false_negatives = len([i for i in attacker_ids if i not in detected_attackers])
        
        if attacker_ids:
            detection_precision = true_positives / max(1, len(detected_attackers))
            detection_recall = true_positives / len(attacker_ids)
            logger.info(f"Attacker detection: precision={detection_precision:.2f}, recall={detection_recall:.2f}")
            logger.info(f"False positives: {false_positives}, False negatives: {false_negatives}")
        
        # Aggregate client models based on defense type
        if args.defense_type == 'none':
            # Standard FedAvg
            global_model = fedavg(client_models)
            logger.info("Aggregated with standard FedAvg")
            
        elif args.defense_type == 'dp_fedguard':
            # Use our enhanced DP-FedGuard implementation
            global_model = dp_fedguard.aggregate(client_models, global_model)
            logger.info("Aggregated with Enhanced DP-FedGuard")
            
        elif args.defense_type == 'median':
            # Coordinate-wise median
            global_model = fed_median(client_models, global_model)
            logger.info("Aggregated with coordinate-wise median")
            
        elif args.defense_type == 'trimmed_mean':
            # Trimmed mean
            trim_ratio = 0.2  # Remove 20% from each end
            global_model = trimmed_mean(client_models, global_model, trim_ratio=trim_ratio)
            logger.info(f"Aggregated with trimmed mean (trim_ratio={trim_ratio})")
            
        elif args.defense_type == 'krum':
            # Krum
            num_attackers = len(attacker_ids) if attacker_ids else max(1, args.num_clients // 5)
            global_model = krum(
                client_models, 
                global_model, 
                num_attackers=num_attackers,
                multi_krum=True,
                num_to_select=max(1, args.num_clients - num_attackers)
            )
            logger.info(f"Aggregated with Multi-Krum (num_attackers={num_attackers})")
            
        elif args.defense_type == 'adaptive':
            # Advanced adaptive strategy
            global_model = adaptive_aggregation_framework(
                client_models,
                global_model,
                trust_scores,
                round_num=round_idx + 1,
                metrics=metrics
            )
            logger.info("Aggregated with advanced adaptive framework")
            
        else:
            raise ValueError(f"Unknown defense type: {args.defense_type}")
        
        # Calculate round time
        round_time = time.time() - round_start_time
        metrics['round_time'].append(round_time)
        
        # Track average client loss
        avg_client_loss = np.mean(client_losses)
        metrics['train_loss'].append(avg_client_loss)
        
        logger.info(f"Round completed in {round_time:.2f} seconds")
        logger.info(f"Average client loss: {avg_client_loss:.4f}")
        logger.info(f"Average client accuracy: {np.mean(client_accuracies):.4f}")
        logger.info("-----------------------------------")
        
        # Evaluate global model
        logger.info(f"\nEvaluating at round {round_idx + 1}:")
        test_metrics = evaluate_with_threshold(global_model, global_test_loader, device)
        
        # Store metrics
        metrics['test_loss'].append(test_metrics['loss'])
        metrics['test_acc'].append(test_metrics['accuracy'])
        metrics['test_auc'].append(test_metrics['auc'])
        metrics['test_f1'].append(test_metrics['f1'])
        metrics['test_precision'].append(test_metrics['precision'])
        metrics['test_recall'].append(test_metrics['recall'])
        
        # Evaluate on attacked test dataset for robustness if available
        if attacked_test_loader:
            robust_metrics = evaluate_with_threshold(global_model, attacked_test_loader, device)
            metrics['robust_acc'].append(robust_metrics['accuracy'])
            metrics['robust_auc'].append(robust_metrics['auc'])
            metrics['robust_f1'].append(robust_metrics['f1'])
        else:
            metrics['robust_acc'].append(None)
            metrics['robust_auc'].append(None)
            metrics['robust_f1'].append(None)
        
        logger.info("\n--- Evaluation Results ---")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        
        if attacked_test_loader:
            logger.info(f"Robust Accuracy: {robust_metrics['accuracy']:.4f}")
            logger.info(f"Robust AUC: {robust_metrics['auc']:.4f}")
            logger.info(f"Robust F1: {robust_metrics['f1']:.4f}")
        logger.info("---------------------------")
        
        # Check for early stopping
        if args.early_stopping:
            current_acc = test_metrics['accuracy']
            
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_model_state = copy.deepcopy(global_model.state_dict())
                patience_counter = 0
                logger.info(f"New best model with accuracy: {current_acc:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} rounds (patience: {args.patience})")
                
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered after {round_idx+1} rounds")
                    break
        
        # Save checkpoint if requested
        if args.save_interval > 0 and (round_idx + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(experiment_dir, f"model_round_{round_idx+1}.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
    
    # Calculate total experiment time
    total_time = time.time() - start_time
    logger.info(f"\nTotal experiment time: {total_time:.2f} seconds")
    
    # If using early stopping, restore best model
    if args.early_stopping and best_model_state is not None:
        global_model.load_state_dict(best_model_state)
        logger.info(f"Restored best model with accuracy: {best_accuracy:.4f}")
    
    # Final evaluation
    logger.info("\n=== Final Evaluation ===")
    final_metrics = evaluate_with_threshold(global_model, global_test_loader, device)
    
    # Evaluate attack impact if we have an attacked test set
    if attacked_test_loader:
        logger.info("\n=== Evaluating Attack Impact ===")
        robust_metrics = evaluate_with_threshold(global_model, attacked_test_loader, device)
        
        # Calculate robustness gap
        acc_gap = final_metrics['accuracy'] - robust_metrics['accuracy']
        auc_gap = final_metrics['auc'] - robust_metrics['auc']
        
        logger.info(f"Robustness gaps:")
        logger.info(f"  Accuracy gap: {acc_gap:.4f}")
        logger.info(f"  AUC gap: {auc_gap:.4f}")
    
    # Save metrics to file with proper processing for JSON serialization
    logger.info(f"Saving metrics to {experiment_dir}")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    
    # Convert numpy values to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            if key in ['client_trust', 'reputation_scores', 'suspected_attacks']:
                # Handle nested lists
                serializable_metrics[key] = [[float(v) for v in round_data] for round_data in value]
            else:
                # Handle simple lists
                serializable_metrics[key] = [float(v) if v is not None else None for v in value]
        else:
            serializable_metrics[key] = value
            
    # Save metrics with proper indentation
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    # Explicitly create visualizations
    try:
        logger.info(f"Creating visualizations in {experiment_dir}")
        from create_plots import create_results_visualization
        create_results_visualization(experiment_dir)
        logger.info(f"Visualizations created in {experiment_dir}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    # Save final model
    final_model_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save(global_model.state_dict(), final_model_path)
    
    # Save best model if using early stopping
    if args.early_stopping and best_model_state is not None:
        best_model_path = os.path.join(experiment_dir, "best_model.pth")
        torch.save(best_model_state, best_model_path)
    
    logger.info(f"\n=== Experiment completed successfully! Results saved to: {experiment_dir} ===")
    logger.info(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Final test AUC: {final_metrics['auc']:.4f}")
    logger.info(f"Final test F1: {final_metrics['f1']:.4f}")
    
    if attacked_test_loader:
        logger.info(f"Final robust accuracy: {robust_metrics['accuracy']:.4f}")
    
    return metrics, global_model

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DP-FedGuard: Privacy-Preserving and Adversarially Robust Federated Learning")
    
    # Dataset parameters
    parser.add_argument("--csv_path", type=str, default="../data/Data_Entry_2017.csv", 
                      help="Path to the NIH Chest X-ray CSV file")
    parser.add_argument("--img_dir", type=str, default="../data/images", 
                      help="Path to the directory containing the X-ray images")
    parser.add_argument("--subset_size", type=int, default=5000, 
                      help="Number of images to use from the dataset")
    parser.add_argument("--test_size", type=float, default=0.2, 
                      help="Fraction of data to use for testing")
    parser.add_argument("--balanced", action="store_true", 
                      help="Create a balanced dataset")
    # NEW: Add balanced sampling argument
    parser.add_argument("--balanced_sampling", action="store_true",
                      help="Use balanced sampling to handle class imbalance")
    
    # Client and data distribution parameters
    parser.add_argument("--num_clients", type=int, default=5, 
                      help="Number of clients for federated learning")
    parser.add_argument("--distribution_type", type=str, default="practical", 
                      choices=["pathological", "practical", "balanced"],
                      help="Type of non-IID data distribution")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="densenet121", 
                      choices=["densenet121", "resnet18", "mobilenet"],
                      help="Model architecture to use")
    
    # Training parameters
    parser.add_argument("--rounds", type=int, default=16, 
                      help="Number of federated learning rounds")
    parser.add_argument("--epochs", type=int, default=2, 
                      help="Number of local training epochs")
    parser.add_argument("--batch_size", type=int, default=64,  
                      help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, 
                      help="Learning rate for training")
    
    # Defense parameters
    parser.add_argument("--defense_type", type=str, default="dp_fedguard", 
                      choices=["none", "dp_fedguard", "median", "trimmed_mean", "krum", "adaptive"],
                      help="Type of defense to use")
    parser.add_argument("--use_dp", action="store_true", 
                      help="Use differential privacy for training")
    parser.add_argument("--noise_multiplier", type=float, default=1.0,  
                      help="Base noise multiplier for differential privacy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,  
                      help="Maximum gradient norm for differential privacy")
    
    # Attack parameters
    parser.add_argument("--attack_type", type=str, default="label_flipping", 
                      choices=["label_flipping", "data_poisoning", "model_poisoning", "none"],
                      help="Type of attack to simulate")
    parser.add_argument("--attack_fraction", type=float, default=0.2, 
                      help="Fraction of clients that are attackers")
    
    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, default="", 
                      help="Name for the experiment (default: auto-generated)")
    parser.add_argument("--results_dir", type=str, default="../results", 
                      help="Directory to save results")
    parser.add_argument("--save_interval", type=int, default=0, 
                      help="Interval for saving models (0 = save only final model)")
    parser.add_argument("--early_stopping", action="store_true",
                      help="Use early stopping to prevent overfitting")
    parser.add_argument("--patience", type=int, default=5,
                      help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true", 
                      help="Disable CUDA even if available")
    parser.add_argument("--debug", action="store_true", 
                      help="Run in debug mode with smaller dataset")
    
    args = parser.parse_args()
    
    # Run the main function
    main(args)