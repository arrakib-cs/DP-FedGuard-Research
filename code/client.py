import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import time
from tqdm import tqdm
import copy
import random

def train_client(model, data_loader, device, epochs=1, lr=0.001, 
                 use_dp=False, noise_multiplier=2.0, max_grad_norm=0.5,
                 data_poisoning=False, poisoning_strength=0.1, verbose=True):
    """
    Train a client model with optional differential privacy
    
    Args:
        model: Model to train
        data_loader: DataLoader with client's data
        device: Device to train on (cpu/cuda)
        epochs: Number of local training epochs
        lr: Learning rate
        use_dp: Whether to use differential privacy
        noise_multiplier: DP noise multiplier (higher = more privacy)
        max_grad_norm: Maximum gradient norm for clipping
        data_poisoning: Whether to apply data poisoning
        poisoning_strength: Strength of data poisoning
        verbose: Whether to print progress
        
    Returns:
        model: Trained model
        metrics: Dictionary containing training metrics
    """
    # Start timing
    start_time = time.time()
    
    # Make sure model is in training mode
    model.train()
    
    # IMPROVED: Better class imbalance handling with more robust calculation
    # Calculate class weights for handling imbalance
    pos_count = 0
    total_count = 0
    
    # Sample batches to estimate class distribution
    for _, labels in data_loader:
        pos_count += torch.sum(labels).item()
        total_count += len(labels)
    
    # Calculate appropriate class weights based on distribution
    if total_count > 0:
        # Calculate positive class weight inversely proportional to frequency
        pos_ratio = pos_count / total_count
        
        if pos_ratio > 0 and pos_ratio < 1:
            # Use a smoothed inverse frequency weighting formula 
            # Smaller ratio -> higher weight, but with protection against extreme values
            # For example: 10% positive -> weight ~9, 50% positive -> weight ~1
            pos_weight = torch.tensor([(1.0 - pos_ratio) / max(0.05, pos_ratio) + 0.5])
            
            # Cap the maximum weight to prevent training instability
            pos_weight = torch.clamp(pos_weight, min=1.0, max=25.0)
            
            if verbose:
                print(f"Using weighted loss with pos_weight={pos_weight.item():.2f} (class ratio: {pos_ratio:.3f})")
        else:
            # If only one class or extreme imbalance, use a default reasonable weight
            pos_weight = torch.tensor([10.0]) if pos_ratio < 0.1 else torch.tensor([1.0])
            if verbose:
                print(f"Extreme class distribution ({pos_ratio:.3f}), using default pos_weight={pos_weight.item():.2f}")
    else:
        pos_weight = None
    
    # Set up loss function with class weights if needed
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Set up privacy engine if using DP
    privacy_engine = None
    epsilon = None
    
    if use_dp:
        try:
            # Additional validation and conversion for DP compatibility
            from opacus.validators import ModuleValidator
            
            # Check if model is already DP compatible
            if not ModuleValidator.validate(model):
                if verbose:
                    print("Model not compatible with DP. Converting BatchNorm layers...")
                
                # Use our own thorough BatchNorm to GroupNorm conversion
                def replace_all_batchnorm_modules(model):
                    """Recursively replace all BatchNorm modules with GroupNorm"""
                    for name, child in list(model.named_children()):
                        if len(list(child.named_children())) > 0:
                            # If module has children, recurse
                            replace_all_batchnorm_modules(child)
                            
                        # Check if current module is BatchNorm
                        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                            # Create GroupNorm replacement
                            num_channels = child.num_features
                            num_groups = min(32, num_channels // 2)
                            num_groups = max(1, num_groups)  # Ensure at least 1 group
                            
                            # Create GroupNorm with equivalent parameters
                            gn = nn.GroupNorm(
                                num_groups=num_groups,
                                num_channels=num_channels,
                                eps=child.eps,
                                affine=child.affine
                            )
                            
                            # Copy weights and biases if affine
                            if child.affine:
                                gn.weight.data.copy_(child.weight.data)
                                gn.bias.data.copy_(child.bias.data)
                            
                            # Replace the BatchNorm with GroupNorm
                            setattr(model, name, gn)
                
                # Apply our thorough conversion - fixed to avoid recursion issues
                replace_all_batchnorm_modules(model)
                
                # Also try Opacus fix as a backup
                try:
                    model = ModuleValidator.fix(model)
                except Exception as e:
                    if verbose:
                        print(f"ModuleValidator.fix failed: {e}")
                
                # Final check
                if not ModuleValidator.validate(model):
                    if verbose:
                        print("WARNING: Model still not fully DP-compatible")
                else:
                    if verbose:
                        print("Model successfully converted to be DP-compatible")
                
                # Recreate optimizer with fixed model
                optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Set up privacy engine
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine(secure_mode=True)
            
            # Make the model, optimizer, and data loader private
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=epochs,
                target_epsilon=3.0, 
                target_delta=1e-5,
                max_grad_norm=max_grad_norm
            )
            
            if verbose:
                print(f"Differential privacy enabled with noise_multiplier={privacy_engine.noise_multiplier:.2f}")
        except Exception as e:
            print(f"Warning: Could not enable differential privacy: {e}")
            print("Continuing without differential privacy...")
            use_dp = False
    
    # Metrics to track
    training_loss = 0.0
    correct = 0
    total = 0
    epoch_losses = []
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Use tqdm for progress bar if verbose
        iterator = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else data_loader
        
        for inputs, labels in iterator:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # Apply data poisoning if enabled
            if data_poisoning:
                # Add random noise to inputs
                noise = torch.randn_like(inputs) * poisoning_strength
                inputs = inputs + noise
                inputs = torch.clamp(inputs, 0, 1)  # Keep in valid range
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Manually clip gradients if not using DP but want gradient clipping
            if not use_dp and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / epoch_total if epoch_total > 0 else 0
        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        epoch_losses.append(epoch_loss)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    
    # Final statistics
    training_loss = epoch_losses[-1]
    training_time = time.time() - start_time
    
    # Calculate privacy used (epsilon) if using DP
    if use_dp and privacy_engine:
        try:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            try:
                epsilon = privacy_engine.get_epsilon(delta=1e-5)
            except TypeError:
                try:
                    # For newer versions of Opacus
                    epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
                except Exception as e2:
                    print(f"Could not calculate privacy budget: {e2}")
                    epsilon = None
            if verbose:
                print(f"Privacy budget used: epsilon = {epsilon:.2f}")
        except Exception as e:
            print(f"Warning: Could not calculate privacy budget: {e}")
            epsilon = None
    
    # Return metrics
    metrics = {
        'loss': training_loss,
        'time': training_time,
        'epsilon': epsilon,
        'epochs': epochs
    }
    
    return model, metrics

def create_noisy_gradients(model, noise_scale=0.1):
    """
    Create noisy gradients for a model (simplified DP)
    
    Args:
        model: Model to add noise to
        noise_scale: Scale of noise to add
        
    Returns:
        model: Model with noisy gradients
    """
    with torch.no_grad():
        for param in model.parameters():
            # Add Gaussian noise to gradients
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * noise_scale
    
    return model

def get_client_update(client_model, global_model):
    """
    Calculate the update a client has made to the global model
    
    Args:
        client_model: Client's trained model
        global_model: Global model before client training
        
    Returns:
        update_dict: Dictionary with parameter updates
    """
    update_dict = {}
    
    for (client_name, client_param), (global_name, global_param) in zip(
        client_model.named_parameters(), global_model.named_parameters()
    ):
        if client_name != global_name:
            continue
            
        # Calculate update
        update = client_param.data - global_param.data
        update_dict[client_name] = update.clone()
    
    return update_dict

def apply_client_update(global_model, update_dict, scale=1.0):
    """
    Apply a client's update to the global model
    
    Args:
        global_model: Global model to update
        update_dict: Dictionary with parameter updates
        scale: Scale factor for the update
        
    Returns:
        updated_model: Updated global model
    """
    updated_model = copy.deepcopy(global_model)
    
    for name, param in updated_model.named_parameters():
        if name in update_dict:
            # Apply scaled update
            param.data += update_dict[name] * scale
    
    return updated_model

def clip_updates(update_dict, max_norm):
    """
    Clip updates to have a maximum L2 norm
    
    Args:
        update_dict: Dictionary with parameter updates
        max_norm: Maximum L2 norm
        
    Returns:
        clipped_dict: Dictionary with clipped updates
    """
    # Calculate total L2 norm of updates
    total_norm = 0
    for update in update_dict.values():
        total_norm += torch.sum(update ** 2).item()
    total_norm = np.sqrt(total_norm)
    
    # Calculate clipping factor
    clip_factor = min(1.0, max_norm / (total_norm + 1e-10))
    
    # Apply clipping
    clipped_dict = {}
    for name, update in update_dict.items():
        clipped_dict[name] = update * clip_factor
    
    return clipped_dict

def simulate_dp_sgd(model, data_loader, criterion, optimizer, device, 
                  epochs=1, max_grad_norm=0.5, noise_multiplier=2.0, verbose=True):
    """
    Manually simulate DP-SGD without using Opacus
    
    Args:
        model: Model to train
        data_loader: DataLoader with client's data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier for DP
        verbose: Whether to print progress
        
    Returns:
        model: Trained model
        metrics: Dictionary with training metrics
    """
    # Start timing
    start_time = time.time()
    
    # Make sure model is in training mode
    model.train()
    
    # Training loop
    epoch_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar if verbose
        iterator = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else data_loader
        
        for inputs, labels in iterator:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Add noise to gradients
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        # Scale noise based on batch size
                        noise_scale = noise_multiplier * max_grad_norm / np.sqrt(len(data_loader.dataset))
                        param.grad += torch.randn_like(param.grad) * noise_scale
            
            # Optimizer step
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        epoch_losses.append(epoch_loss)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    
    # Final statistics
    training_loss = epoch_losses[-1]
    training_time = time.time() - start_time
    
    # Estimate epsilon (privacy used)
    # This is a very simplified approximation
    sampling_rate = 1.0 / len(data_loader)
    steps = epochs * len(data_loader)
    epsilon = np.sqrt(2 * np.log(1.25 / 1e-5)) * noise_multiplier * np.sqrt(steps * sampling_rate)
    
    # Return metrics
    metrics = {
        'loss': training_loss,
        'time': training_time,
        'epsilon': epsilon,
        'epochs': epochs
    }
    
    return model, metrics

def train_with_varying_dp(model, data_loader, device, base_params):
    """
    Train a model with varying DP parameters to show privacy-utility trade-off
    
    Args:
        model: Base model to train
        data_loader: DataLoader with training data
        device: Device to train on
        base_params: Dictionary with base training parameters
        
    Returns:
        results: Dictionary with training results at different privacy levels
    """
    # Define privacy levels to test
    noise_multipliers = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = []
    
    print("Training with varying DP parameters:")
    for noise_multiplier in noise_multipliers:
        print(f"\nNoise multiplier: {noise_multiplier}")
        
        # Create a copy of the model
        model_copy = copy.deepcopy(model)
        
        # Train with this noise level
        trained_model, metrics = train_client(
            model=model_copy,
            data_loader=data_loader,
            device=device,
            epochs=base_params.get('epochs', 1),
            lr=base_params.get('lr', 0.001),
            use_dp=True,
            noise_multiplier=noise_multiplier,
            max_grad_norm=base_params.get('max_grad_norm', 0.5),
            verbose=base_params.get('verbose', True)
        )
        
        # Add result
        result = {
            'noise_multiplier': noise_multiplier,
            'epsilon': metrics['epsilon'],
            'loss': metrics['loss'],
            'time': metrics['time']
        }
        results.append(result)
        
        print(f"  Privacy (epsilon): {metrics['epsilon']:.2f}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Training time: {metrics['time']:.2f}s")
    
    return results

if __name__ == "__main__":
    # Test the client functions with dummy data
    from dataset import ChestXrayDataset
    from models import create_model
    import torchvision.transforms as transforms
    
    # Create a simple dummy dataset for testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dummy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a model
    model = create_model('densenet121')
    model = model.to(device)
    
    print("Client training module loaded and ready for testing!")
    print("Use the train_client function with real data to see results.")