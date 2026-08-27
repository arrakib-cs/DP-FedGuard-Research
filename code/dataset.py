import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ChestXrayDataset(Dataset):
    """Dataset class for NIH Chest X-ray dataset"""
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a gray placeholder image instead of failing
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        label = self.dataframe.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

def get_weighted_dataloader(dataset, batch_size):
    """
    Create a DataLoader with weighted sampling to handle class imbalance
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size for the loader
        
    Returns:
        DataLoader with weighted sampling
    """
    # Extract all labels
    labels = [label for _, label in dataset]
    
    # Count occurrences of each class
    class_counts = np.bincount([int(l) for l in labels])
    
    # Calculate inverse weights (less frequent classes get higher weights)
    class_weights = 1. / class_counts
    
    # Assign weights to each sample based on its class
    sample_weights = [class_weights[int(label)] for label in labels]
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create and return DataLoader with the sampler
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, drop_last=False)

# Evaluation function with threshold tuning
def evaluate_with_threshold(model, dataloader, device):
    """
    Evaluate model with optimal threshold tuning for F1 score
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_labels = []
    all_probs = []
    losses = []
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            loss = criterion(outputs, labels.float()).item()
            losses.append(loss)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Find optimal threshold for F1 score
    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Calculate metrics with best threshold
    final_preds = (all_probs >= best_threshold).astype(int)
    final_f1 = f1_score(all_labels, final_preds)
    
    # Handle edge cases for AUC calculation
    if len(np.unique(all_labels)) < 2:
        # Only one class in the labels
        final_auc = 0.5  # Default value when only one class
    else:
        try:
            final_auc = roc_auc_score(all_labels, all_probs)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            final_auc = 0.5
    
    # Handle edge cases for precision and recall
    if np.sum(final_preds) == 0:  # No positive predictions
        final_precision = 0.0
    else:
        final_precision = precision_score(all_labels, final_preds)
        
    if np.sum(all_labels) == 0:  # No positive labels
        final_recall = 0.0
    else:
        final_recall = recall_score(all_labels, final_preds)

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"F1 Score: {final_f1:.4f}")
    print(f"AUC Score: {final_auc:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")

    return {
        'threshold': best_threshold,
        'accuracy': (final_preds == all_labels).mean(),
        'f1': final_f1,
        'auc': final_auc,
        'precision': final_precision,
        'recall': final_recall,
        'loss': np.mean(losses)
    }

def prepare_dataset(csv_path, img_dir, subset_size=20000, verbose=True, balanced=True):
    """
    Prepare the NIH Chest X-ray dataset for federated learning experiments
    
    Args:
        csv_path: Path to the Data_Entry_2017.csv file
        img_dir: Directory containing the X-ray images
        subset_size: Number of images to use (default: 20000)
        verbose: Whether to print progress information
        balanced: Whether to create a more balanced dataset
        
    Returns:
        df: Dataframe with image paths and binary labels
    """
    if verbose:
        print("Loading and preparing the NIH Chest X-ray dataset...")
    
    # Load the metadata
    df = pd.read_csv(csv_path)
    
    # Add full image path
    df['image_path'] = df['Image Index'].apply(lambda x: os.path.join(img_dir, x))
    
    # Filter to only include images that exist
    original_size = len(df)
    df = df[df['image_path'].apply(os.path.exists)]
    if verbose:
        print(f"Found {len(df)} images out of {original_size} in the CSV")
    
    # Create binary labels (1 for pneumonia, 0 for normal)
    df['label'] = df['Finding Labels'].apply(lambda x: 1.0 if 'Pneumonia' in x else 0.0)
    
    # Split into pneumonia and normal
    pneumonia_df = df[df['label'] == 1.0]
    normal_df = df[df['label'] == 0.0]
    
    if verbose:
        print(f"Total dataset: {len(df)} images")
        print(f"Pneumonia: {len(pneumonia_df)} images")
        print(f"Normal: {len(normal_df)} images")
    
    # IMPROVED: Better class balancing for the extreme imbalance case
    # Take a stratified subset to ensure balance and manage size
    if len(df) > subset_size:
        if balanced:
            # For balanced dataset, improve the handling of severe imbalance
            min_class_size = min(len(pneumonia_df), len(normal_df))
            max_class_size = max(len(pneumonia_df), len(normal_df))
            
            # Determine if we need special handling for extreme imbalance
            imbalance_ratio = max_class_size / (min_class_size if min_class_size > 0 else 1)
            is_pneumonia_minority = len(pneumonia_df) < len(normal_df)
            
            if imbalance_ratio > 10:  # Severe imbalance
                if verbose:
                    print(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}). Using special balancing.")
                
                # Include all samples from minority class
                minority_samples = pneumonia_df if is_pneumonia_minority else normal_df
                
                # Determine how many majority samples to include
                # In severe imbalance, we might want a more balanced subset than original
                remaining_slots = subset_size - len(minority_samples)
                majority_subset_size = min(remaining_slots, len(minority_samples) * 5)
                
                # Sample from majority class
                majority_samples = normal_df if is_pneumonia_minority else pneumonia_df
                majority_subset = majority_samples.sample(n=majority_subset_size, random_state=42)
                
                # Combine into final dataset
                df = pd.concat([minority_samples, majority_subset])
            else:
                # More moderate imbalance - use original balancing logic
                if min_class_size * 2 <= subset_size:
                    # Can have perfect balance
                    pneumonia_subset_size = min_class_size
                    normal_subset_size = min_class_size
                    
                    # If we still have room, add more from both classes
                    remaining = subset_size - (pneumonia_subset_size + normal_subset_size)
                    if remaining > 0:
                        pneumonia_extra = min(len(pneumonia_df) - pneumonia_subset_size, remaining // 2)
                        normal_extra = min(len(normal_df) - normal_subset_size, remaining - pneumonia_extra)
                        
                        pneumonia_subset_size += pneumonia_extra
                        normal_subset_size += normal_extra
                else:
                    # Take equal numbers up to subset_size
                    pneumonia_subset_size = subset_size // 2
                    normal_subset_size = subset_size - pneumonia_subset_size
                
                # Take random samples from each class
                pneumonia_subset = pneumonia_df.sample(n=pneumonia_subset_size, random_state=42)
                normal_subset = normal_df.sample(n=normal_subset_size, random_state=42)
                
                # Combine into final dataset
                df = pd.concat([pneumonia_subset, normal_subset])
        else:
            # Original distribution logic
            pneumonia_ratio = len(pneumonia_df) / len(df)
            pneumonia_subset_size = int(pneumonia_ratio * subset_size)
            normal_subset_size = subset_size - pneumonia_subset_size
            
            # Take random samples from each class
            pneumonia_subset = pneumonia_df.sample(n=pneumonia_subset_size, random_state=42)
            normal_subset = normal_df.sample(n=normal_subset_size, random_state=42)
            
            # Combine into final dataset
            df = pd.concat([pneumonia_subset, normal_subset])
        
        if verbose:
            print(f"Created subset with {len(df)} images")
            print(f"Pneumonia: {len(df[df['label'] == 1.0])} images ({len(df[df['label'] == 1.0])/len(df):.1%})")
            print(f"Normal: {len(df[df['label'] == 0.0])} images ({len(df[df['label'] == 0.0])/len(df):.1%})")
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def check_class_imbalance(df, title="Class Distribution"):
    """Visualize class distribution"""
    class_counts = df['label'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(['Normal', 'Pneumonia'], [class_counts[0], class_counts[1]])
    plt.title(title)
    plt.ylabel('Number of Images')
    plt.savefig(f"../results/{title.replace(' ', '_')}.png")
    plt.close()

def create_non_iid_distribution(df, num_clients=5, distribution_type='pathological'):
    """
    Create non-IID data distribution across clients
    
    Args:
        df: Dataframe with image paths and labels
        num_clients: Number of clients to create
        distribution_type: Type of non-IID distribution to create
                          'pathological': Each client gets mostly one class
                          'practical': Realistic distribution with varying degrees of imbalance
                          'balanced': Each client gets a balanced dataset
    
    Returns:
        client_dfs: List of dataframes for each client
    """
    # Split into pneumonia and normal
    pneumonia_df = df[df['label'] == 1.0]
    normal_df = df[df['label'] == 0.0]
    
    # Create client dataframes
    client_dfs = []
    
    if distribution_type == 'pathological':
        # Pathological non-IID: each client gets mostly one class
        for i in range(num_clients):
            # For most clients, skew toward one class
            if i < num_clients // 2:
                # More pneumonia than normal
                pneumonia_count = int(len(pneumonia_df) / (num_clients // 2 + num_clients % 2))
                normal_count = int(len(normal_df) / num_clients) // 2
                
                client_pneumonia = pneumonia_df.iloc[i * pneumonia_count:(i + 1) * pneumonia_count]
                client_normal = normal_df.iloc[i * normal_count:(i + 1) * normal_count]
            else:
                # More normal than pneumonia
                pneumonia_count = int(len(pneumonia_df) / num_clients) // 2
                normal_count = int(len(normal_df) / (num_clients // 2))
                
                idx = i - num_clients // 2
                client_pneumonia = pneumonia_df.iloc[
                    (num_clients // 2) * pneumonia_count + idx * pneumonia_count:
                    (num_clients // 2) * pneumonia_count + (idx + 1) * pneumonia_count
                ]
                client_normal = normal_df.iloc[idx * normal_count:(idx + 1) * normal_count]
            
            client_df = pd.concat([client_pneumonia, client_normal])
            client_dfs.append(client_df)
                
    elif distribution_type == 'practical':
        # Practical non-IID: varying degrees of class imbalance
        pneumonia_ratios = np.linspace(0.1, 0.9, num_clients)  # From 10% to 90% pneumonia
        
        # Calculate total data size
        total_size = len(df)
        
        # Aim for roughly equal total size for each client
        client_size = total_size // num_clients
        
        # Make sure we don't exceed dataset size
        total_allocated = 0
        client_sizes = []
        for i in range(num_clients - 1):  # Allocate for all but the last client
            client_sizes.append(client_size)
            total_allocated += client_size
        
        # Last client gets the remainder
        client_sizes.append(total_size - total_allocated)
        
        for i, (ratio, size) in enumerate(zip(pneumonia_ratios, client_sizes)):
            pneumonia_count = int(size * ratio)
            normal_count = size - pneumonia_count
            
            # Ensure we don't request more than available
            pneumonia_count = min(pneumonia_count, len(pneumonia_df))
            normal_count = min(normal_count, len(normal_df))
            
            # Get slices of data
            start_p = (i * pneumonia_count) % max(1, len(pneumonia_df))
            end_p = min(start_p + pneumonia_count, len(pneumonia_df))
            start_n = (i * normal_count) % max(1, len(normal_df))
            end_n = min(start_n + normal_count, len(normal_df))
            
            client_pneumonia = pneumonia_df.iloc[start_p:end_p]
            client_normal = normal_df.iloc[start_n:end_n]
            
            # If we wrapped around, take from the beginning
            if end_p - start_p < pneumonia_count and len(pneumonia_df) > 0:
                remaining = pneumonia_count - (end_p - start_p)
                client_pneumonia = pd.concat([
                    client_pneumonia, 
                    pneumonia_df.iloc[:remaining]
                ])
            
            if end_n - start_n < normal_count and len(normal_df) > 0:
                remaining = normal_count - (end_n - start_n)
                client_normal = pd.concat([
                    client_normal, 
                    normal_df.iloc[:remaining]
                ])
            
            client_df = pd.concat([client_pneumonia, client_normal])
            client_dfs.append(client_df)
                
    else:  # balanced
        # Balanced: each client gets roughly the same distribution
        # First, let's calculate the target distribution (overall dataset ratio)
        overall_ratio = len(pneumonia_df) / max(1, len(df))
        
        for i in range(num_clients):
            # Calculate per-client size
            client_size = len(df) // num_clients
            pneumonia_count = int(client_size * overall_ratio)
            normal_count = client_size - pneumonia_count
            
            # Ensure we don't exceed available data
            pneumonia_count = min(pneumonia_count, len(pneumonia_df) // num_clients)
            normal_count = min(normal_count, len(normal_df) // num_clients)
            
            # Get slices for this client
            start_p = i * pneumonia_count
            end_p = (i + 1) * pneumonia_count if i < num_clients - 1 else len(pneumonia_df)
            start_n = i * normal_count
            end_n = (i + 1) * normal_count if i < num_clients - 1 else len(normal_df)
            
            # Make sure we don't exceed dataframe bounds
            end_p = min(end_p, len(pneumonia_df))
            end_n = min(end_n, len(normal_df))
            
            client_pneumonia = pneumonia_df.iloc[start_p:end_p]
            client_normal = normal_df.iloc[start_n:end_n]
            
            client_df = pd.concat([client_pneumonia, client_normal])
            client_dfs.append(client_df)
    
    # Print distribution stats
    print("\nClient data distribution:")
    for i, client_df in enumerate(client_dfs):
        pneumonia_count = sum(client_df['label'] == 1.0)
        normal_count = sum(client_df['label'] == 0.0)
        total = len(client_df)
        if total > 0:  # Avoid division by zero
            print(f"Client {i+1}: {total} samples, {pneumonia_count} pneumonia ({pneumonia_count/total:.1%}), {normal_count} normal ({normal_count/total:.1%})")
        else:
            print(f"Client {i+1}: 0 samples")
    
    return client_dfs

def create_federated_datasets(client_dfs, test_size=0.2, batch_size=64, balanced_sampling=True):
    """
    Create DataLoaders for each client with optional balanced sampling
    
    Args:
        client_dfs: List of dataframes for each client
        test_size: Fraction of data to use for testing
        batch_size: Batch size for training
        balanced_sampling: Whether to use balanced sampling for training
        
    Returns:
        client_train_loaders: List of training DataLoaders for each client
        client_test_loaders: List of testing DataLoaders for each client
        global_test_loader: DataLoader with combined test data from all clients
    """
    # Define transforms with data augmentation for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simpler transforms for test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataloaders for each client
    client_train_loaders = []
    client_test_loaders = []
    all_test_dfs = []
    
    for i, client_df in enumerate(client_dfs):
        if len(client_df) == 0:
            print(f"Warning: Client {i+1} has no data. Creating empty loaders.")
            # Create empty datasets
            train_dataset = ChestXrayDataset(pd.DataFrame(columns=['image_path', 'label']), '', transform)
            test_dataset = ChestXrayDataset(pd.DataFrame(columns=['image_path', 'label']), '', test_transform)
        else:
            # Split into train and test
            train_df, test_df = train_test_split(
                client_df, 
                test_size=test_size, 
                stratify=client_df['label'],
                random_state=42
            )
            
            all_test_dfs.append(test_df)
            
            # Create datasets
            train_dataset = ChestXrayDataset(train_df, '', transform)
            test_dataset = ChestXrayDataset(test_df, '', test_transform)
        
        # IMPROVED: Use weighted sampling for class imbalance when requested
        if balanced_sampling:
            # Calculate class counts for this client's training data
            labels = train_df['label'].values if len(client_df) > 0 else []
            unique_labels = np.unique(labels)
            
            # Check if we have both classes
            if len(unique_labels) > 1:
                # Use weighted sampling to balance classes
                class_counts = np.bincount(labels.astype(int))
                class_weights = 1. / class_counts
                sample_weights = [class_weights[int(label)] for label in labels]
                
                # Create weighted sampler
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                
                # Create balanced loader with sampler
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=2,
                    drop_last=False
                )
                
                print(f"Client {i+1}: Using weighted sampling for class balance")
            else:
                # Only one class present, use standard loader
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    drop_last=False
                )
                print(f"Client {i+1}: Only one class present, using standard loader")
        else:
            # Standard unbalanced loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                drop_last=False
            )
        
        # Test loader (no need for balancing in test set)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)
    
    # Create a global test set with combined data from all clients
    global_test_df = pd.concat(all_test_dfs)
    global_test_dataset = ChestXrayDataset(global_test_df, '', test_transform)
    global_test_loader = DataLoader(
        global_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return client_train_loaders, client_test_loaders, global_test_loader

def simulate_attacks(client_dfs, client_train_loaders, attack_type='label_flipping', attacker_ids=None, attack_fraction=1.0):
    """
    Simulate attacks on selected clients
    
    Args:
        client_dfs: Original client dataframes
        client_train_loaders: Original client training loaders
        attack_type: Type of attack to simulate
                    'label_flipping': Flip labels (0->1, 1->0)
                    'data_poisoning': Add noise to images
                    'model_poisoning': Return original loaders (actual attack happens in training)
        attacker_ids: List of client IDs to attack (0-indexed)
        attack_fraction: Fraction of data to attack for partial attacks
    
    Returns:
        attacked_train_loaders: List of training loaders with simulated attacks
    """
    # Default: attack one client
    if attacker_ids is None:
        attacker_ids = [len(client_dfs) - 1]  # Last client is the attacker
    
    print(f"\nSimulating {attack_type} attack on clients {[i+1 for i in attacker_ids]}")
    
    attacked_train_loaders = []
    
    for i, (client_df, train_loader) in enumerate(zip(client_dfs, client_train_loaders)):
        if i in attacker_ids:
            # This client is an attacker
            if attack_type == 'label_flipping':
                # Make a copy of the client dataframe
                attacked_df = client_df.copy()
                
                # If partial attack, only flip a fraction of labels
                if attack_fraction < 1.0:
                    attack_indices = attacked_df.sample(frac=attack_fraction, random_state=42).index
                    attacked_df.loc[attack_indices, 'label'] = 1 - attacked_df.loc[attack_indices, 'label']
                else:
                    # Flip all labels
                    attacked_df['label'] = 1 - attacked_df['label']
                
                print(f"  Client {i+1}: Flipped {attack_fraction:.1%} of labels")
                
                # Split into train and test again
                attacked_train_df, _ = train_test_split(
                    attacked_df, 
                    test_size=0.2, 
                    stratify=attacked_df['label'],
                    random_state=42
                )
                
                # Create dataset and loader with flipped labels
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                attacked_dataset = ChestXrayDataset(attacked_train_df, '', transform)
                attacked_loader = DataLoader(
                    attacked_dataset, 
                    batch_size=train_loader.batch_size, 
                    shuffle=True, 
                    num_workers=train_loader.num_workers,
                    drop_last=False
                )
                
                attacked_train_loaders.append(attacked_loader)
                
            elif attack_type == 'data_poisoning':
                # For data poisoning, we'll manipulate the images during loading
                # This is implemented in the model training loop
                print(f"  Client {i+1}: Data poisoning attack will be applied during training")
                attacked_train_loaders.append(train_loader)
                
            elif attack_type == 'model_poisoning':
                # For model poisoning, we'll manipulate the model updates
                # This is implemented in the federated learning algorithm
                print(f"  Client {i+1}: Model poisoning attack will be applied during aggregation")
                attacked_train_loaders.append(train_loader)
                
            else:
                print(f"  Warning: Unknown attack type '{attack_type}', using original loader")
                attacked_train_loaders.append(train_loader)
        else:
            # Not an attacker, use original loader
            attacked_train_loaders.append(train_loader)
    
    return attacked_train_loaders

def visualize_client_distributions(client_dfs):
    """Create a visualization of the data distribution across clients"""
    num_clients = len(client_dfs)
    
    # Calculate class distributions
    pneumonia_counts = []
    normal_counts = []
    total_counts = []
    
    for client_df in client_dfs:
        pneumonia_count = sum(client_df['label'] == 1.0)
        normal_count = sum(client_df['label'] == 0.0)
        total = len(client_df)
        
        pneumonia_counts.append(pneumonia_count)
        normal_counts.append(normal_count)
        total_counts.append(total)
    
    # Create a stacked bar chart
    plt.figure(figsize=(12, 6))
    
    client_ids = [f"Client {i+1}" for i in range(num_clients)]
    
    # Plot bars
    plt.bar(client_ids, normal_counts, label='Normal')
    plt.bar(client_ids, pneumonia_counts, bottom=normal_counts, label='Pneumonia')
    
    # Add percentages on bars
    for i in range(num_clients):
        if total_counts[i] > 0:  # Avoid division by zero
            pneumonia_pct = pneumonia_counts[i] / total_counts[i] * 100
            normal_pct = normal_counts[i] / total_counts[i] * 100
            
            # Add percentage labels
            plt.text(i, normal_counts[i] / 2, f"{normal_pct:.1f}%", ha='center')
            plt.text(i, normal_counts[i] + pneumonia_counts[i] / 2, f"{pneumonia_pct:.1f}%", ha='center')
    
    plt.xlabel('Client')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.title('Data Distribution Across Clients')
    plt.savefig("../results/client_data_distribution.png")
    plt.close()

if __name__ == "__main__":
    # Test the dataset functions
    csv_path = "../data/Data_Entry_2017.csv"
    img_dir = "../data/NIH_Chest_Xray/images"
    
    # Prepare the dataset
    df = prepare_dataset(csv_path, img_dir, subset_size=5000, balanced=True)
    
    # Check class balance
    check_class_imbalance(df)
    
    # Create non-IID distribution
    client_dfs = create_non_iid_distribution(df, num_clients=5, distribution_type='practical')
    
    # Visualize client distributions
    visualize_client_distributions(client_dfs)
    
    # Create dataloaders with balanced sampling
    client_train_loaders, client_test_loaders, global_test_loader = create_federated_datasets(
        client_dfs, balanced_sampling=True
    )
    
    # Simulate attacks
    attacked_train_loaders = simulate_attacks(
        client_dfs, 
        client_train_loaders, 
        attack_type='label_flipping', 
        attacker_ids=[1, 3]
    )
    
    print("Dataset preparation complete!")