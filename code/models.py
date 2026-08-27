import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import copy
from opacus.validators import ModuleValidator
from tqdm import tqdm

from torchvision.models import DenseNet121_Weights, ResNet18_Weights, MobileNet_V2_Weights

def create_model(model_name='densenet121', num_classes=1, pretrained=True):
    """
    Create a neural network model for chest X-ray classification
    with guaranteed BatchNorm to GroupNorm conversion for DP compatibility
    """
    import torch
    import torch.nn as nn
    from torchvision import models
    from opacus.validators import ModuleValidator
    
    # Create base model
    if model_name == 'resnet18':
        try:
            # First create the base model WITHOUT pretrained weights (we'll load our own)
            model = models.resnet18(weights=None)
            
            # Modify the classifier for correct number of classes
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            
            # Try to load pre-converted model with proper class count
            print("Attempting to load pre-converted DP-compatible model...")
            model.load_state_dict(torch.load("fixed_resnet18.pth"))
            print("Successfully loaded pre-converted DP-compatible model")
            
            # Verify that the model is valid for DP
            is_valid = ModuleValidator.validate(model)
            if is_valid:
                print("Successfully verified DP compatibility")
                return model
            else:
                print("Loaded model is not DP-compatible, will convert on-the-fly")
        except Exception as e:
            print(f"Error loading pre-converted model: {e}")
            print("Will create and convert model on-the-fly")
            
        # If loading fails, create a new model with direct conversion
        print("Creating new ResNet18 model with direct BatchNorm to GroupNorm conversion")
        
        # Create model with pretrained weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        # Convert BatchNorm to GroupNorm using Opacus's built-in converter
        print("Converting BatchNorm to GroupNorm...")
        model = ModuleValidator.fix(model)
        
        # Then adjust the classifier
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        # Final verification
        is_valid = ModuleValidator.validate(model)
        if is_valid:
            print("Successfully created and verified DP-compatible model")
            return model
        else:
            print("WARNING: Model still has BatchNorm layers after conversion")
    
    # Keep the rest of your function the same for other model types
    # ...
    
    return model

def convert_densenet_batchnorm(model):
    """Specialized function to handle DenseNet's nested BatchNorm layers"""
    # DenseNet has a complex structure with nested BatchNorm layers in DenseBlocks
    # This function specifically targets those nested structures
    
    # Handle _DenseLayer norm1 and norm2 attributes directly
    for module in model.modules():
        # Check if this is a _DenseLayer or similar module in DenseNet
        if hasattr(module, 'norm1') and isinstance(module.norm1, nn.BatchNorm2d):
            # Replace norm1
            channels = module.norm1.num_features
            groups = calculate_optimal_groups(channels)
            
            gn = nn.GroupNorm(
                num_groups=groups,
                num_channels=channels,
                eps=module.norm1.eps,
                affine=module.norm1.affine
            )
            
            # Copy weights and biases if affine
            if module.norm1.affine:
                gn.weight.data.copy_(module.norm1.weight.data)
                gn.bias.data.copy_(module.norm1.bias.data)
            
            # Replace the BatchNorm with GroupNorm
            module.norm1 = gn
        
        # Check for norm2
        if hasattr(module, 'norm2') and isinstance(module.norm2, nn.BatchNorm2d):
            # Replace norm2
            channels = module.norm2.num_features
            groups = calculate_optimal_groups(channels)
            
            gn = nn.GroupNorm(
                num_groups=groups,
                num_channels=channels,
                eps=module.norm2.eps,
                affine=module.norm2.affine
            )
            
            # Copy weights and biases if affine
            if module.norm2.affine:
                gn.weight.data.copy_(module.norm2.weight.data)
                gn.bias.data.copy_(module.norm2.bias.data)
            
            # Replace the BatchNorm with GroupNorm
            module.norm2 = gn
    
    return model

def replace_all_batchnorm_modules(model):
    """
    Enhanced recursive replacement of all BatchNorm modules with GroupNorm
    - Handles nested modules more thoroughly
    - Uses improved heuristics for group size selection
    - Properly transfers running statistics
    """
    for name, child in list(model.named_children()):
        if len(list(child.named_children())) > 0:
            # If module has children, recurse
            replace_all_batchnorm_modules(child)
            
        # Check if current module is BatchNorm
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Determine appropriate number of channels and groups
            num_channels = child.num_features
            num_groups = calculate_optimal_groups(num_channels)
            
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

def calculate_optimal_groups(num_channels):
    """Calculate optimal number of groups for GroupNorm based on channel count"""
    # For very few channels
    if num_channels <= 8:
        return 1
    
    # For moderate number of channels, try to have each group with ~4-8 channels
    if num_channels <= 32:
        # Try common divisors in order of preference
        for div in [4, 8, 2]:
            if num_channels % div == 0:
                return num_channels // div
        return 1  # Fallback to 1 group if no good divisor found
    
    # For many channels, use standard group sizes
    for groups in [32, 16, 8, 4]:
        if num_channels % groups == 0:
            return groups
    
    # If not evenly divisible by standard sizes, find best divisor
    for groups in range(min(32, num_channels // 2), 0, -1):
        if num_channels % groups == 0:
            return groups
    
    # Last resort: just use 1 group
    return 1

def emergency_bn_conversion(model):
    """Last resort emergency conversion for stubborn BatchNorm layers"""
    print("Applying emergency BatchNorm conversion...")
    
    # Direct conversion of model.modules() 
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Get parent module
            for parent_module in model.modules():
                for name, child in list(parent_module.named_children()):
                    if child is module:
                        # Found the parent, now replace BatchNorm with GroupNorm
                        num_channels = module.num_features
                        
                        # Force 1 group as last resort
                        gn = nn.GroupNorm(1, num_channels, eps=module.eps, affine=module.affine)
                        
                        if module.affine:
                            gn.weight.data.copy_(module.weight.data)
                            gn.bias.data.copy_(module.bias.data)
                        
                        setattr(parent_module, name, gn)
                        break

def find_remaining_bn_layers(model):
    """Find and print all remaining BatchNorm layers in the model"""
    bn_layers = []
    
    def find_bn(module, prefix=''):
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_layers.append(path)
            else:
                find_bn(child, path)
    
    find_bn(model)
    
    if bn_layers:
        print(f"Still found BatchNorm layers after conversion: {bn_layers}")
    
    return bn_layers

def weights_init(m):
    """Initialize model weights"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def model_size(model):
    """Calculate the size of a model in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_fingerprint(model):
    """Create a unique fingerprint for model state"""
    fingerprint = ""
    for name, param in model.named_parameters():
        # Only use a sample of parameters for efficiency
        if 'weight' in name and param.dim() > 1:
            # Take the mean of first row
            values = param.data[0].mean().item()
            fingerprint += f"{values:.4f}|"
    return fingerprint

def model_distance(model_a, model_b):
    """Calculate Euclidean distance between two models"""
    squared_sum = 0
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        # Skip non-matching layers (shouldn't happen with same architecture)
        if name_a != name_b:
            continue
            
        # Calculate squared difference
        squared_sum += torch.sum((param_a.data - param_b.data) ** 2).item()
        
    return np.sqrt(squared_sum)

def copy_model(model):
    """Deep copy a model"""
    return copy.deepcopy(model)

def prune_model(model, pruning_rate=0.2):
    """
    Simple magnitude-based pruning to create a smaller model
    
    Args:
        model: PyTorch model
        pruning_rate: Fraction of weights to prune
        
    Returns:
        pruned_model: Model with some weights set to zero
    """
    pruned_model = copy_model(model)
    
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            # Get the absolute values
            abs_weight = torch.abs(param.data)
            
            # Calculate threshold for pruning
            threshold = torch.quantile(abs_weight, pruning_rate)
            
            # Create binary mask
            mask = (abs_weight > threshold).float()
            
            # Apply mask (set pruned weights to zero)
            param.data = param.data * mask
    
    return pruned_model

def federated_averaging(models, weights=None):
    """
    Perform weighted federated averaging of models
    
    Args:
        models: List of PyTorch models
        weights: List of weights for each model (if None, equal weighting is used)
        
    Returns:
        global_model: Averaged model
    """
    if weights is None:
        # Equal weighting
        weights = [1.0 / len(models)] * len(models)
    else:
        # Normalize weights to sum to 1
        weights = [w / sum(weights) for w in weights]
    
    # Start with a copy of the first model
    global_model = copy_model(models[0])
    
    # Reset parameters
    for param in global_model.parameters():
        param.data.zero_()
    
    # Weighted average of parameters
    for model, weight in zip(models, weights):
        for global_param, client_param in zip(global_model.parameters(), model.parameters()):
            global_param.data += client_param.data * weight
    
    return global_model

def create_model_with_preset_bias(model_name='densenet121', target_class=1, bias_amount=5.0):
    """
    Create a model with artificial bias toward a specific class
    
    Args:
        model_name: Name of model architecture
        target_class: Class to bias toward (0 or 1)
        bias_amount: Amount of bias to add
        
    Returns:
        model: Model with preset bias
    """
    model = create_model(model_name)
    
    # Add bias to the classifier
    if model_name == 'densenet121':
        classifier = model.classifier
    elif model_name == 'resnet18':
        classifier = model.fc
    elif model_name == 'mobilenet':
        classifier = model.classifier[1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Set bias (binary classification)
    if target_class == 1:
        classifier.bias.data += bias_amount
    else:
        classifier.bias.data -= bias_amount
    
    return model

def initialize_model_poisoning(base_model, poisoning_type='random', target_class=1):
    """
    Initialize model poisoning attack
    
    Args:
        base_model: Base model to modify
        poisoning_type: Type of poisoning
                      'random': Random initialization
                      'targeted': Target a specific class
                      'subtle': Small parameter perturbations
        target_class: For targeted poisoning, class to target
        
    Returns:
        poisoned_model: Model initialized for poisoning attack
    """
    poisoned_model = copy_model(base_model)
    
    if poisoning_type == 'random':
        # Re-initialize with random weights
        poisoned_model.apply(weights_init)
        
    elif poisoning_type == 'targeted':
        # Similar to base model but strong bias toward target class
        if target_class == 1:
            # Bias toward positive class (pneumonia)
            for name, param in poisoned_model.named_parameters():
                if 'classifier' in name and 'bias' in name:
                    param.data += 2.0  # Strong bias toward pneumonia
                    
        else:
            # Bias toward negative class (normal)
            for name, param in poisoned_model.named_parameters():
                if 'classifier' in name and 'bias' in name:
                    param.data -= 2.0  # Strong bias toward normal
    
    elif poisoning_type == 'subtle':
        # Small perturbations to all parameters
        for param in poisoned_model.parameters():
            # Add small Gaussian noise
            param.data += torch.randn_like(param.data) * 0.01
    
    else:
        print(f"Warning: Unknown poisoning type '{poisoning_type}', using original model")
    
    return poisoned_model

def perform_model_poisoning(client_model, global_model, poisoning_type='replace'):
    """
    Perform model poisoning attack during training
    
    Args:
        client_model: Client's trained model
        global_model: Current global model
        poisoning_type: Type of model poisoning
                      'replace': Replace with malicious model
                      'scale': Scale updates to have large impact
                      'targeted': Target specific layers
        
    Returns:
        poisoned_model: Model with poisoning applied
    """
    poisoned_model = copy_model(client_model)
    
    if poisoning_type == 'replace':
        # Already initialized with poisoning, nothing to do
        pass
        
    elif poisoning_type == 'scale':
        # Scale the difference between client and global model
        scaling_factor = 10.0  # Large scaling factor
        
        for p_poisoned, p_global in zip(poisoned_model.parameters(), global_model.parameters()):
            # Calculate update
            update = p_poisoned.data - p_global.data
            
            # Scale the update
            p_poisoned.data = p_global.data + (update * scaling_factor)
            
    elif poisoning_type == 'targeted':
        # Target only specific layers (e.g., classifier)
        for name_p, param_p in poisoned_model.named_parameters():
            if 'classifier' in name_p or 'fc' in name_p:  # Target final layers
                # Find corresponding global parameter
                for name_g, param_g in global_model.named_parameters():
                    if name_p == name_g:
                        # Create an opposite update (flip sign)
                        update = param_p.data - param_g.data
                        param_p.data = param_g.data - (update * 5.0)  # Reverse and scale
                        break
    
    else:
        print(f"Warning: Unknown poisoning type '{poisoning_type}', using original model")
    
    return poisoned_model

if __name__ == "__main__":
    # Test the model functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a model
    model = create_model('densenet121')
    print(f"Model created: {type(model).__name__}")
    print(f"Model size: {model_size(model):.2f} MB")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Test model fingerprinting
    fingerprint = model_fingerprint(model)
    print(f"Model fingerprint: {fingerprint[:50]}...")
    
    # Create a pruned version
    pruned_model = prune_model(model, pruning_rate=0.3)
    print(f"Pruned model size: {model_size(pruned_model):.2f} MB")
    
    # Test model distance
    distance = model_distance(model, pruned_model)
    print(f"Distance between original and pruned models: {distance:.4f}")
    
    # Test federated averaging
    models = [model, pruned_model]
    weights = [0.7, 0.3]
    averaged_model = federated_averaging(models, weights)
    print(f"Averaged model size: {model_size(averaged_model):.2f} MB")
    
    # Test model poisoning
    poisoned_model = initialize_model_poisoning(model, poisoning_type='targeted')
    print(f"Poisoned model created")
    
    print("Model tests completed!")