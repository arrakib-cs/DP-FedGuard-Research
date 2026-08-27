import torch
import torch.nn as nn
from torchvision import models
from opacus.validators import ModuleValidator
import os

def create_and_save_dp_compatible_model(model_name='resnet18', output_path='fixed_resnet18.pth', num_classes=1):
    """Create a DP-compatible model by converting BatchNorm to GroupNorm"""
    print(f"Creating DP-compatible {model_name}...")
    
    # Create the model with pretrained weights
    if model_name == 'resnet18':
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif model_name == 'densenet121':
        from torchvision.models import DenseNet121_Weights
        weights = DenseNet121_Weights.DEFAULT 
        model = models.densenet121(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Check if model is already valid for DP
    is_valid = ModuleValidator.validate(model)
    print(f"Initial model valid for DP: {is_valid}")
    
    if not is_valid:
        # Use Opacus to fix the model by replacing BatchNorm with GroupNorm
        print("Converting BatchNorm layers to GroupNorm...")
        model = ModuleValidator.fix(model)
        
        # Verify the model is now DP-compatible
        is_valid = ModuleValidator.validate(model)
        if is_valid:
            print("Successfully converted model!")
        else:
            # Manual replacement if automatic conversion fails
            print("Standard conversion failed, trying manual approach...")
            
            # Function to replace BatchNorm with GroupNorm
            def replace_batchnorm(module):
                for name, child in list(module.named_children()):
                    if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        num_features = child.num_features
                        # Calculate appropriate number of groups
                        num_groups = max(1, num_features // 8)  # 8 channels per group
                        # Create GroupNorm module
                        group_norm = nn.GroupNorm(num_groups=num_groups, 
                                                 num_channels=num_features,
                                                 eps=child.eps,
                                                 affine=child.affine)
                        # Copy weights if applicable
                        if child.affine:
                            group_norm.weight.data.copy_(child.weight.data)
                            group_norm.bias.data.copy_(child.bias.data)
                        # Replace the BatchNorm module
                        setattr(module, name, group_norm)
                    else:
                        # Recursively apply to child modules
                        replace_batchnorm(child)
            
            # Apply manual replacement
            replace_batchnorm(model)
            
            # Final check
            is_valid = ModuleValidator.validate(model)
            print(f"After manual conversion, model valid for DP: {is_valid}")
    
    # Modify the final layer for binary classification
    if model_name == 'resnet18':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet121':
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    # Save the model
    print(f"Saving DP-compatible model to {output_path}")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved successfully!")
    
    return model

if __name__ == "__main__":
    # Create and save models for both architectures
    create_and_save_dp_compatible_model('resnet18', 'fixed_resnet18.pth')
    create_and_save_dp_compatible_model('densenet121', 'fixed_densenet121.pth')