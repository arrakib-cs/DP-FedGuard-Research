import torch
import numpy as np
import copy
from scipy import stats
import time
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fedavg(client_models, client_sample_sizes=None):
    """
    Standard Federated Averaging (FedAvg) algorithm
    
    Args:
        client_models: List of client models
        client_sample_sizes: Number of samples for each client (for weighted averaging)
        
    Returns:
        global_model: Updated global model
    """
    # Start with a copy of the first client model
    global_model = copy.deepcopy(client_models[0])
    
    # If no sample sizes are provided, use equal weighting
    if client_sample_sizes is None:
        client_sample_sizes = [1] * len(client_models)
    
    total_samples = sum(client_sample_sizes)
    
    # Reset parameters to zero
    for param in global_model.parameters():
        param.data.zero_()
    
    # Weighted average based on sample sizes
    for client_idx, client_model in enumerate(client_models):
        weight = client_sample_sizes[client_idx] / total_samples
        
        for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            global_param.data += client_param.data * weight
    
    return global_model

def calculate_update_norm(client_model, global_model):
    """
    Calculate the L2 norm of a client's update
    
    Args:
        client_model: Client model after training
        global_model: Global model before client training
        
    Returns:
        update_norm: L2 norm of the update
    """
    squared_sum = 0
    for (client_name, client_param), (global_name, global_param) in zip(
        client_model.named_parameters(), global_model.named_parameters()
    ):
        if client_name != global_name:
            continue
            
        # Calculate parameter update
        update = client_param.data - global_param.data
        squared_sum += torch.sum(update ** 2).item()
    
    return np.sqrt(squared_sum)

def get_update_tensor(client_model, global_model):
    """
    Get a flattened tensor of all updates from a client
    
    Args:
        client_model: Client model after training
        global_model: Global model before client training
        
    Returns:
        update_tensor: Flattened tensor with all updates
    """
    updates = []
    for (client_name, client_param), (global_name, global_param) in zip(
        client_model.named_parameters(), global_model.named_parameters()
    ):
        if client_name != global_name:
            continue
            
        # Flatten and append the update
        update = (client_param.data - global_param.data).flatten()
        updates.append(update)
    
    # Concatenate all updates into a single tensor
    # Check if updates list is empty
    if not updates:
        print(f"Warning: No matching parameters found between client and global model")
        # Return a dummy tensor to avoid errors
        return torch.zeros(1, device=next(client_model.parameters()).device)
        
    # Concatenate all updates into a single tensor
    update_tensor = torch.cat(updates)
    return update_tensor

def fed_median(client_models, global_model):
    """
    Coordinate-wise median aggregation for Byzantine robustness
    
    Args:
        client_models: List of client models
        global_model: Current global model
        
    Returns:
        new_global_model: Updated global model
    """
    new_global_model = copy.deepcopy(global_model)
    
    # For each parameter in the model
    for param_idx, (param_name, param) in enumerate(new_global_model.named_parameters()):
        # Skip non-trainable parameters
        if not param.requires_grad:
            continue
            
        # Collect parameters from all clients
        parameter_list = []
        for client_model in client_models:
            # Extract the parameter with the same name from the client model
            for client_param_name, client_param in client_model.named_parameters():
                if client_param_name == param_name:
                    parameter_list.append(client_param.data)
                    break
        
        # Stack parameters and calculate coordinate-wise median
        stacked_parameters = torch.stack(parameter_list, dim=0)
        median_parameter, _ = torch.median(stacked_parameters, dim=0)
        
        # Update global model parameter
        param.data.copy_(median_parameter)
    
    return new_global_model

def trimmed_mean(client_models, global_model, trim_ratio=0.2):
    """
    Trimmed mean aggregation for Byzantine robustness
    
    Args:
        client_models: List of client models
        global_model: Current global model
        trim_ratio: Percentage of largest and smallest values to trim
        
    Returns:
        new_global_model: Updated global model
    """
    # Need at least 3 clients for this to make sense
    if len(client_models) < 3:
        logger.warning("Using trimmed mean with fewer than 3 clients. Falling back to FedAvg.")
        return fedavg(client_models)
    
    new_global_model = copy.deepcopy(global_model)
    
    # Calculate how many clients to trim from each end
    num_clients = len(client_models)
    num_to_trim = int(num_clients * trim_ratio)
    
    # For each parameter in the model
    for param_idx, (param_name, param) in enumerate(new_global_model.named_parameters()):
        # Skip non-trainable parameters
        if not param.requires_grad:
            continue
            
        # Collect parameters from all clients
        parameter_list = []
        for client_model in client_models:
            # Extract the parameter with the same name from the client model
            for client_param_name, client_param in client_model.named_parameters():
                if client_param_name == param_name:
                    parameter_list.append(client_param.data)
                    break
        
        # Stack parameters
        stacked_parameters = torch.stack(parameter_list, dim=0)
        
        # Use more efficient implementation based on sorting
        sorted_parameters, _ = torch.sort(stacked_parameters, dim=0)
        if num_to_trim > 0:
            trimmed_parameters = sorted_parameters[num_to_trim:-num_to_trim]
        else:
            trimmed_parameters = sorted_parameters
            
        trimmed_param = torch.mean(trimmed_parameters, dim=0)
        
        # Update global model parameter
        param.data.copy_(trimmed_param)
    
    return new_global_model

def krum(client_models, global_model, num_attackers=1, multi_krum=False, num_to_select=1):
    """
    Krum aggregation for Byzantine robustness
    
    Args:
        client_models: List of client models
        global_model: Current global model
        num_attackers: Expected number of malicious clients
        multi_krum: Whether to use Multi-Krum (select multiple clients)
        num_to_select: Number of clients to select if using Multi-Krum
        
    Returns:
        new_global_model: Updated global model
    """
    num_clients = len(client_models)
    
    # Need more honest clients than Byzantine
    if num_clients <= 2 * num_attackers + 2:
        logger.warning(f"Krum needs more than {2 * num_attackers + 2} clients. Falling back to FedAvg.")
        return fedavg(client_models)
    
    # Extract flattened updates from each client
    updates = []
    for client_model in client_models:
        update = get_update_tensor(client_model, global_model)
        updates.append(update)
    
    # Calculate pairwise squared distances between updates
    distances = torch.zeros(num_clients, num_clients)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = torch.sum((updates[i] - updates[j]) ** 2).item()
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each client, sum the n-f-2 smallest distances
    scores = []
    for i in range(num_clients):
        # Get distances to other clients
        client_distances = distances[i].clone()
        client_distances[i] = float('inf')  # Exclude self
        
        # Get the n-f-2 smallest distances
        num_to_sum = num_clients - num_attackers - 2
        smallest_distances, _ = torch.topk(client_distances, num_to_sum, largest=False)
        
        # Score is the sum of these distances (lower is better)
        score = torch.sum(smallest_distances).item()
        scores.append(score)
    
    # Select the client(s) with the lowest score(s)
    if multi_krum:
        # Select multiple clients
        num_to_select = min(num_to_select, num_clients - num_attackers)
        selected_indices = torch.topk(torch.tensor(scores), num_to_select, largest=False).indices.tolist()
        
        # Average the selected models
        selected_models = [client_models[i] for i in selected_indices]
        return fedavg(selected_models)
    else:
        # Select the single model with lowest score
        selected_index = np.argmin(scores)
        return copy.deepcopy(client_models[selected_index])

def calculate_cosine_similarity(update_a, update_b):
    """
    Calculate cosine similarity between two update vectors
    
    Args:
        update_a: First update vector
        update_b: Second update vector
        
    Returns:
        similarity: Cosine similarity (-1 to 1)
    """
    dot_product = torch.sum(update_a * update_b).item()
    norm_a = torch.norm(update_a).item()
    norm_b = torch.norm(update_b).item()
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def enhanced_outlier_detection(updates, contamination=0.2):
    """
    Enhanced outlier detection using multiple techniques
    
    Args:
        updates: List of update tensors
        contamination: Expected fraction of outliers
        
    Returns:
        outlier_scores: Anomaly scores for each client (0-1, higher is more trusted)
    """
    # Convert updates to numpy arrays
    updates_np = [update.cpu().numpy() for update in updates]
    
    # Dimensionality reduction for efficiency
    reduced_updates = []
    
    # Apply PCA if we have enough updates
    if len(updates_np) >= 3:
        try:
            # Stack updates
            stacked_updates = np.vstack([update.reshape(1, -1) for update in updates_np])
            
            # Standardize the data
            scaler = StandardScaler()
            standardized_updates = scaler.fit_transform(stacked_updates)
            
            # Apply PCA 
            n_components = min(len(updates_np) - 1, 10)
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(standardized_updates)
            
            # Use reduced representation
            reduced_updates = reduced_data
        except Exception as e:
            logger.warning(f"PCA failed: {e}. Using alternative reduction.")
            # Fallback to simple chunking
            reduced_updates = []
            for update in updates_np:
                # Simple dimensionality reduction
                chunk_size = 1000
                if len(update) > chunk_size:
                    chunks = len(update) // chunk_size
                    chunked = update[:chunks * chunk_size].reshape(chunks, chunk_size)
                    reduced = np.mean(chunked, axis=1)
                else:
                    reduced = update
                reduced_updates.append(reduced)
    else:
        # With too few clients, use simple reduction
        for update in updates_np:
            # Simple chunking
            chunk_size = 1000
            if len(update) > chunk_size:
                chunks = len(update) // chunk_size
                chunked = update[:chunks * chunk_size].reshape(chunks, chunk_size)
                reduced = np.mean(chunked, axis=1)
            else:
                reduced = update
            reduced_updates.append(reduced)
    
    # Convert to 2D array for analysis
    if isinstance(reduced_updates, list):
        X = np.vstack(reduced_updates)
    else:
        X = reduced_updates
    
    # Apply multiple detection methods
    
    # 1. Isolation Forest
    iso_scores = np.ones(len(updates))
    try:
        clf = IsolationForest(contamination=contamination, random_state=42)
        iso_scores = clf.fit_predict(X)
        # Convert from {-1, 1} to [0, 1]
        iso_scores = np.where(iso_scores == 1, 1.0, 0.3)
    except Exception as e:
        logger.warning(f"Isolation Forest failed: {e}")
    
    # 2. Distance-based outlier detection
    distance_scores = np.ones(len(updates))
    try:
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(X, 'euclidean'))
        
        # Average distance to other points
        avg_distances = np.mean(distances, axis=1)
        
        # Normalize to [0, 1] (invert so higher = more normal)
        max_dist = np.max(avg_distances)
        if max_dist > 0:
            distance_scores = 1 - (avg_distances / max_dist)
    except Exception as e:
        logger.warning(f"Distance-based detection failed: {e}")
    
    # 3. Clustering-based detection
    cluster_scores = np.ones(len(updates))
    try:
        if len(updates) >= 3:
            # Apply KMeans clustering
            k = min(3, len(updates))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Count elements in each cluster
            from collections import Counter
            cluster_counts = Counter(labels)
            
            # Clients in smaller clusters are more suspicious
            for i, label in enumerate(labels):
                if cluster_counts[label] < len(updates) / k / 2:
                    cluster_scores[i] = 0.5
    except Exception as e:
        logger.warning(f"Clustering-based detection failed: {e}")
    
    # Combine scores (weighted average)
    weights = [0.4, 0.4, 0.2]  # Weights for each method
    combined_scores = weights[0] * iso_scores + weights[1] * distance_scores + weights[2] * cluster_scores
    
    # Normalize to [0, 1]
    normalized_scores = np.clip(combined_scores, 0, 1)
    
    return normalized_scores

def get_layer_update_statistics(client_models, global_model):
    """
    Get statistics about updates for each layer
    
    Args:
        client_models: List of client models
        global_model: Current global model
        
    Returns:
        layer_stats: Dictionary with layer-wise statistics
    """
    num_clients = len(client_models)
    layer_stats = {}
    
    # Extract updates for each layer
    for param_name, global_param in global_model.named_parameters():
        if global_param.requires_grad:
            layer_updates = []
            
            for client_model in client_models:
                # Get the parameter with the same name
                for client_param_name, client_param in client_model.named_parameters():
                    if client_param_name == param_name:
                        # Calculate update
                        update = client_param.data - global_param.data
                        
                        # Calculate statistics
                        mean = torch.mean(update).item()
                        std = torch.std(update).item() 
                        norm = torch.norm(update).item()
                        
                        layer_updates.append({
                            'mean': mean,
                            'std': std,
                            'norm': norm,
                            'quantiles': {
                                '25': torch.quantile(update.flatten(), 0.25).item(),
                                '50': torch.quantile(update.flatten(), 0.50).item(),
                                '75': torch.quantile(update.flatten(), 0.75).item()
                            }
                        })
                        break
            
            layer_stats[param_name] = layer_updates
    
    return layer_stats

def calculate_client_trust_scores(client_models, global_model, label_distributions=None, 
                                 prev_trust_scores=None, round_num=1, client_history=None):
    """
    Enhanced trust score calculation with improved statistical detection
    
    Args:
        client_models: List of trained client models
        global_model: Current global model
        label_distributions: List of dictionaries with client label distribution info
        prev_trust_scores: Previous round's trust scores (if available)
        round_num: Current round number (for adaptive weighting)
        client_history: Dictionary tracking client behavior over time
        
    Returns:
        trust_scores: List of trust scores for each client (0 to 1, higher is more trusted)
        updated_history: Updated client history
    """
    start_time = time.time()
    num_clients = len(client_models)
    
    # Initialize client history if not provided
    if client_history is None:
        client_history = {
            'update_norms': [[] for _ in range(num_clients)],
            'cosine_similarities': [[] for _ in range(num_clients)],
            'suspected_attacks': [0] * num_clients,
            'trust_scores': [[] for _ in range(num_clients)],
            'layer_norms': [{} for _ in range(num_clients)]
        }
    
    # Get update tensors and norms
    update_tensors = []
    update_norms = []
    
    for client_idx, client_model in enumerate(client_models):
        update = get_update_tensor(client_model, global_model)
        update_tensors.append(update)
        
        # Calculate update norm
        norm = torch.norm(update).item()
        update_norms.append(norm)
        
        # Update history
        client_history['update_norms'][client_idx].append(norm)
    
    # ======== Multi-faceted Trust Score Calculation ========
    
    # 1. Norm-based outlier detection with robust statistics
    # Use Median Absolute Deviation (MAD) for robust outlier detection
    median_norm = np.median(update_norms)
    mad = np.median(np.abs(np.array(update_norms) - median_norm))
    
    # Prevent division by zero
    if mad < 1e-8:
        mad = np.mean(np.abs(np.array(update_norms) - median_norm))
        if mad < 1e-8:
            mad = 1.0
    
    norm_scores = []
    for norm in update_norms:
        # Calculate robust z-score
        z_score = 0.6745 * abs(norm - median_norm) / mad
        
        # Use sigmoid to convert to trust score (centered at z=3)
        trust = 1.0 / (1.0 + np.exp(z_score - 3))
        norm_scores.append(trust)
    
    # 2. Advanced statistical outlier detection
    isolation_scores = enhanced_outlier_detection(update_tensors, contamination=0.2)
    
    # 3. Cosine similarity analysis
    # Calculate pairwise cosine similarities between updates
    similarities = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            sim = calculate_cosine_similarity(update_tensors[i], update_tensors[j])
            similarities[i, j] = sim
            similarities[j, i] = sim
    
    # Average similarity to other clients
    avg_similarities = np.mean(similarities, axis=1)
    
    # Update history
    for i, sim in enumerate(avg_similarities):
        client_history['cosine_similarities'][i].append(sim)
    
    # Convert to trust scores
    similarity_scores = []
    for sim in avg_similarities:
        # Normalize from [-1, 1] to [0, 1]
        norm_sim = (sim + 1) / 2
        
        # Score based on normalized similarity
        # We want to penalize both very low and very high similarities
        # Very high similarity could indicate colluding attackers
        if norm_sim < 0.3:
            # Very dissimilar - suspicious
            score = norm_sim * 2  # Linear increase
        elif norm_sim > 0.9:
            # Very similar - potential collusion
            score = 2.0 - norm_sim * 1.5  # Linear decrease
        else:
            # Normal range - highest trust
            score = 0.8 + (norm_sim - 0.5) * 0.4  # Peak around 0.6-0.7
        
        similarity_scores.append(min(1.0, max(0.0, score)))
    
    # 4. Layer-wise analysis for targeted attacks
    layer_stats = get_layer_update_statistics(client_models, global_model)
    
    # Analyze layer-specific behaviors
    layer_trust_scores = [1.0] * num_clients  # Default full trust
    
    for layer_name, client_stats in layer_stats.items():
        # Check if this is a classifier layer (usually more targeted in attacks)
        is_classifier = any(keyword in layer_name for keyword in 
                          ['classifier', 'fc', 'linear', 'out'])
        
        # Get norms for this layer
        layer_norms = [stats['norm'] for stats in client_stats]
        
        # Use robust statistics
        layer_median = np.median(layer_norms)
        layer_mad = np.median(np.abs(np.array(layer_norms) - layer_median))
        if layer_mad < 1e-8:
            layer_mad = np.mean(np.abs(np.array(layer_norms) - layer_median)) 
            if layer_mad < 1e-8:
                layer_mad = 1.0
        
        # Calculate robust z-scores
        z_scores = [0.6745 * abs(norm - layer_median) / layer_mad for norm in layer_norms]
        
        # Update client layer norm history
        for i, norm in enumerate(layer_norms):
            if layer_name not in client_history['layer_norms'][i]:
                client_history['layer_norms'][i][layer_name] = []
            client_history['layer_norms'][i][layer_name].append(norm)
        
        # Analyze temporal patterns
        if round_num > 2:
            for i, client_idx in enumerate(range(num_clients)):
                if layer_name in client_history['layer_norms'][client_idx]:
                    history = client_history['layer_norms'][client_idx][layer_name]
                    
                    if len(history) >= 3:
                        # Check for oscillating pattern (could be sign of strategic attack)
                        if (max(history[-3:]) / min(history[-3:]) > 3 and 
                            abs(history[-3] - history[-1]) / history[-3] < 0.2):
                            # Oscillating behavior - highly suspicious
                            layer_trust_scores[client_idx] *= 0.7
                            logger.debug(f"Oscillating behavior detected in client {client_idx+1}, layer {layer_name}")
        
        # Apply layer-specific trust penalties
        for i, z_score in enumerate(z_scores):
            # Stronger penalties for classifier layers
            if is_classifier:
                if z_score > 5.0:  # Extreme outlier
                    layer_trust_scores[i] *= 0.5
                elif z_score > 3.0:  # Strong outlier
                    layer_trust_scores[i] *= 0.7
                elif z_score > 2.0:  # Moderate outlier
                    layer_trust_scores[i] *= 0.9
            else:
                # Less severe penalties for non-classifier layers
                if z_score > 5.0:
                    layer_trust_scores[i] *= 0.7
                elif z_score > 3.0:
                    layer_trust_scores[i] *= 0.9
    
    # 5. Label distribution analysis (if available)
    if label_distributions is not None:
        # Calculate average positive ratio across clients
        avg_positive_ratio = np.mean([d.get('positive_ratio', 0.5) for d in label_distributions])
        
        distribution_scores = []
        for dist in label_distributions:
            positive_ratio = dist.get('positive_ratio', 0.5)
            
            # Calculate deviation from average
            deviation = abs(positive_ratio - avg_positive_ratio)
            
            # Maximum possible deviation 
            max_deviation = max(avg_positive_ratio, 1.0 - avg_positive_ratio)
            
            if max_deviation > 0:
                # More lenient scoring for data distribution
                # We don't want to heavily penalize legitimate non-IID data
                distribution_score = 1.0 - (deviation / max_deviation) * 0.4  # Max penalty of 0.4
            else:
                distribution_score = 1.0
                
            distribution_scores.append(distribution_score)
    else:
        distribution_scores = [1.0] * num_clients
    
    # 6. Combine scores with adaptive weighting
    # Adjust weights based on round number and observed patterns
    weights = {}
    
    if round_num < 3:
        # Early rounds: focus on statistical outlier detection and data distribution
        weights = {
            'norm': 0.25,
            'isolation': 0.25,
            'similarity': 0.2,
            'layer': 0.1,
            'distribution': 0.2
        }
    elif round_num < 8:
        # Middle rounds: more weight on similarity and layer analysis
        weights = {
            'norm': 0.2,
            'isolation': 0.2,
            'similarity': 0.25,
            'layer': 0.25,
            'distribution': 0.1
        }
    else:
        # Later rounds: emphasize behavior patterns and layer analysis
        weights = {
            'norm': 0.2,
            'isolation': 0.15,
            'similarity': 0.3,
            'layer': 0.3,
            'distribution': 0.05
        }
    
    # Calculate combined trust scores
    raw_trust_scores = []
    for i in range(num_clients):
        score = (
            weights['norm'] * norm_scores[i] +
            weights['isolation'] * isolation_scores[i] +
            weights['similarity'] * similarity_scores[i] +
            weights['layer'] * layer_trust_scores[i] +
            weights['distribution'] * distribution_scores[i]
        )
        raw_trust_scores.append(score)
    
    # 7. Apply temporal analysis with adaptive momentum
    if prev_trust_scores is not None:
        # Base momentum - how much to consider previous scores
        # Decreases over time to allow adaptation to evolving attacks
        momentum_base = max(0.3, min(0.8, 0.8 - 0.05 * round_num))
        
        # Suspicion threshold - clients below this are suspicious
        suspicion_threshold = 0.5
        
        trust_scores = []
        for i, (curr, prev) in enumerate(zip(raw_trust_scores, prev_trust_scores)):
            # Track if this client is suspected of being malicious
            if curr < suspicion_threshold:
                client_history['suspected_attacks'][i] += 1
            
            # Calculate adaptive momentum based on behavior patterns
            consecutive_suspicions = client_history['suspected_attacks'][i]
            
            if consecutive_suspicions > 3:
                # Strong evidence of attack - higher momentum to maintain low trust
                momentum = min(0.9, momentum_base + 0.2)
            elif consecutive_suspicions > 1:
                # Some evidence of attack - slightly higher momentum
                momentum = min(0.85, momentum_base + 0.1)
            elif curr < prev - 0.3:
                # Sharp drop in trust - respond quickly
                momentum = max(0.4, momentum_base - 0.2)
            elif curr > prev + 0.3:
                # Sharp improvement - be cautious
                momentum = min(0.9, momentum_base + 0.1)
            else:
                # Normal change - use default momentum
                momentum = momentum_base
            
            # Apply momentum
            trust = momentum * prev + (1 - momentum) * curr
            
            # Add adaptive forgiveness mechanism
            if curr > 0.8 and prev < 0.7 and round_num > 3:
                # Check recent history
                if i < len(client_history['trust_scores']) and len(client_history['trust_scores'][i]) >= 3:
                    recent_scores = client_history['trust_scores'][i][-3:]
                    
                    # If consistently good recently, allow faster rehabilitation
                    if min(recent_scores) > 0.6:
                        trust += 0.05
            
            trust_scores.append(trust)
            
            # Update trust score history
            if i < len(client_history['trust_scores']):
                client_history['trust_scores'][i].append(trust)
    else:
        # First round - use raw scores
        trust_scores = raw_trust_scores
        
        # Initialize trust score history
        for i, score in enumerate(trust_scores):
            client_history['trust_scores'][i].append(score)
    
    # 8. Apply final bounds and adjustments
    trust_scores = [max(0.0, min(1.0, score)) for score in trust_scores]
    
    # Log calculation time (for performance monitoring)
    calc_time = time.time() - start_time
    logger.debug(f"Trust calculation completed in {calc_time:.4f}s")
    
    # Log trust scores and factors
    logger.debug(f"Trust Scores: {[f'{score:.3f}' for score in trust_scores]}")
    logger.debug(f"Norm Scores: {[f'{score:.3f}' for score in norm_scores]}")
    logger.debug(f"Isolation Scores: {[f'{score:.3f}' for score in isolation_scores]}")
    logger.debug(f"Similarity Scores: {[f'{score:.3f}' for score in similarity_scores]}")
    logger.debug(f"Layer Trust Scores: {[f'{score:.3f}' for score in layer_trust_scores]}")
    logger.debug(f"Distribution Scores: {[f'{score:.3f}' for score in distribution_scores]}")
    
    return trust_scores, client_history

def calculate_adaptive_weights(trust_scores, min_weight=0.0, max_weight=1.0, sharpness=4.0):
    """
    Calculate adaptive weights based on trust scores using a sigmoid function
    
    Args:
        trust_scores: Trust scores for each client (0 to 1)
        min_weight: Minimum weight to assign
        max_weight: Maximum weight to assign
        sharpness: Sharpness of the sigmoid function (higher = sharper transition)
        
    Returns:
        weights: List of weights for each client
    """
    weights = []
    
    for trust in trust_scores:
        # Centered sigmoid function
        x = sharpness * (trust - 0.5)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        
        # Scale to desired range
        weight = min_weight + sigmoid * (max_weight - min_weight)
        weights.append(weight)
    
    # Ensure weights sum to 1
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        # If all weights are 0, use equal weighting
        weights = [1.0 / len(trust_scores)] * len(trust_scores)
    
    return weights

def dynamic_dp_noise(trust_scores, base_noise=1.0, max_noise=5.0):
    """
    Calculate dynamic DP noise based on trust scores
    
    Args:
        trust_scores: List of trust scores for each client
        base_noise: Base noise multiplier for fully trusted clients
        max_noise: Maximum noise multiplier for untrusted clients
        
    Returns:
        noise_multipliers: List of noise multipliers for each client
    """
    noise_multipliers = []
    
    for trust in trust_scores:
        # Transfer function from trust to noise:
        # 1. Quadratic scaling for more aggressive noise increase at lower trust
        trust_factor = trust ** 2  
        
        # 2. Apply inverse relationship (lower trust → higher noise)
        noise = base_noise + (max_noise - base_noise) * (1.0 - trust_factor)
        
        # 3. Add a minimum noise floor for security
        noise = max(base_noise * 0.7, noise)
        
        noise_multipliers.append(noise)
    
    return noise_multipliers

def dp_fedguard_aggregate(client_models, global_model, trust_scores, 
                         aggregation_type='weighted_averaging', round_num=1):
    """
    Enhanced DP-FedGuard aggregation with multi-tier defense
    
    Args:
        client_models: List of client models
        global_model: Current global model
        trust_scores: Trust scores for each client
        aggregation_type: Type of aggregation to use
            - 'weighted_averaging': Weight clients by trust scores
            - 'krum_filtered': Use Krum to filter out low-trust clients
            - 'trimmed_mean': Use trimmed mean with trust-based trimming
            - 'adaptive': Automatically select best approach based on trust distribution
            - 'ensemble': Combine multiple approaches
        round_num: Current round number (for adaptive strategies)
        
    Returns:
        new_global_model: Updated global model
    """
    # Start timing
    start_time = time.time()
    
    # Create a copy of the global model to update
    new_global_model = copy.deepcopy(global_model)
    
    # Get number of clients
    num_clients = len(client_models)
    
    # Determine best aggregation approach if 'adaptive' is selected
    if aggregation_type == 'adaptive':
        # Calculate trust distribution metrics
        avg_trust = sum(trust_scores) / num_clients if num_clients > 0 else 0
        min_trust = min(trust_scores) if trust_scores else 0
        num_suspicious = sum(1 for score in trust_scores if score < 0.3)
        suspicious_ratio = num_suspicious / num_clients if num_clients > 0 else 0
        
        # Select strategy based on trust distribution
        if suspicious_ratio > 0.4:
            # High attack ratio - use stronger defense
            if num_clients >= 5:
                aggregation_type = 'krum_filtered'
            else:
                aggregation_type = 'trimmed_mean'
        elif suspicious_ratio > 0.1 or min_trust < 0.2:
            # Moderate attack ratio or at least one highly suspicious client
            if round_num > 5:
                # In later rounds, use ensemble for better stability
                aggregation_type = 'ensemble'
            else:
                aggregation_type = 'trimmed_mean'
        else:
            # Low/no attack ratio
            aggregation_type = 'weighted_averaging'
        
        logger.info(f"Adaptive aggregation selected: {aggregation_type} " + 
                   f"(suspicious ratio: {suspicious_ratio:.2f}, avg trust: {avg_trust:.2f})")
    
    # Apply the selected aggregation method
    if aggregation_type == 'weighted_averaging':
        # Calculate adaptive weights using sigmoid function
        weights = calculate_adaptive_weights(
            trust_scores, 
            min_weight=0.0,  # Completely untrusted clients get zero weight
            max_weight=1.0,
            sharpness=4.0 + round_num / 5.0  # Increase sharpness over time
        )
        
        # For clients with very low trust, set weight to zero
        for i, trust in enumerate(trust_scores):
            if trust < 0.1:  # Extremely low trust
                weights[i] = 0.0
        
        # Re-normalize weights if needed
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            
            # Apply weighted aggregation by layer type
            for param_name, param in new_global_model.named_parameters():
                # Skip non-trainable parameters
                if not param.requires_grad:
                    continue
                
                # Reset parameter
                param.data.zero_()
                
                # Determine layer type for specialized handling
                is_classifier = any(keyword in param_name for keyword in 
                                   ['classifier', 'fc', 'linear', 'out'])
                
                if is_classifier:
                    # For classifier layers (often targeted):
                    # Apply stricter filtering for low-trust clients
                    for i, client_model in enumerate(client_models):
                        if weights[i] < 1e-6:  # Skip zero-weight clients
                            continue
                            
                        for client_param_name, client_param in client_model.named_parameters():
                            if client_param_name == param_name:
                                # For low trust clients, limit contribution to critical layers
                                if trust_scores[i] < 0.5:
                                    # Get update
                                    update = client_param.data - global_model.state_dict()[param_name]
                                    
                                    # Scale update based on trust
                                    trust_factor = trust_scores[i] / 0.5  # Normalize to [0,1]
                                    scaled_update = update * trust_factor
                                    
                                    # Apply scaled update
                                    param.data += weights[i] * (global_model.state_dict()[param_name] + scaled_update)
                                else:
                                    # Normal contribution for trusted clients
                                    param.data += weights[i] * client_param.data
                                break
                else:
                    # For non-classifier layers:
                    # Standard weighted averaging
                    for i, client_model in enumerate(client_models):
                        if weights[i] < 1e-6:  # Skip zero-weight clients
                            continue
                            
                        for client_param_name, client_param in client_model.named_parameters():
                            if client_param_name == param_name:
                                param.data += weights[i] * client_param.data
                                break
        else:
            # If all clients have zero weight, fall back to median
            logger.warning("All clients have zero weight. Using median aggregation.")
            return fed_median(client_models, global_model)
    
    elif aggregation_type == 'krum_filtered':
        # Use trust scores to determine suspected number of attackers
        trust_threshold = 0.3  # Clients below this are considered potential attackers
        suspected_attackers = sum(trust < trust_threshold for trust in trust_scores)
        
        # Ensure we have at least 1 suspected attacker for Krum
        num_attackers = max(1, suspected_attackers)
        
        # Use Multi-Krum to select the most trusted clients
        return krum(client_models, global_model, num_attackers=num_attackers, 
                  multi_krum=True, num_to_select=max(1, num_clients - num_attackers))
    
    elif aggregation_type == 'trimmed_mean':
        # Use trust scores to determine trim ratio
        trust_threshold = 0.3  # Clients below this are considered potential attackers
        suspected_attackers = sum(trust < trust_threshold for trust in trust_scores)
        
        # Calculate trim ratio based on suspected attackers
        trim_ratio = max(0.1, min(0.4, suspected_attackers / num_clients))
        
        # Use trimmed mean with calculated ratio
        return trimmed_mean(client_models, global_model, trim_ratio=trim_ratio)
    
    elif aggregation_type == 'ensemble':
        # Combine multiple aggregation methods for robust defense
        # 1. Get model from each approach
        weighted_model = dp_fedguard_aggregate(client_models, global_model, trust_scores, 'weighted_averaging')
        median_model = fed_median(client_models, global_model)
        
        # For trimmed mean, use adaptive trim ratio
        trust_threshold = 0.3
        suspected_attackers = sum(trust < trust_threshold for trust in trust_scores)
        trim_ratio = max(0.1, min(0.4, suspected_attackers / num_clients))
        trimmed_model = trimmed_mean(client_models, global_model, trim_ratio=trim_ratio)
        
        # 2. Combine models based on trust distribution
        # If we have high trust overall, prefer weighted averaging
        # If we have low trust, prefer robust aggregation
        avg_trust = sum(trust_scores) / len(trust_scores)
        
        if avg_trust > 0.7:  # High overall trust
            weights = [0.7, 0.15, 0.15]  # Favor weighted averaging
        elif avg_trust > 0.4:  # Moderate trust
            weights = [0.4, 0.3, 0.3]    # Balance approaches
        else:  # Low trust
            weights = [0.2, 0.4, 0.4]    # Favor robust methods
        
        # 3. Perform weighted ensemble of models
        models = [weighted_model, median_model, trimmed_model]
        ensemble_model = copy.deepcopy(global_model)
        
        # For each parameter, take weighted average from the three models
        for param_name, param in ensemble_model.named_parameters():
            # Skip non-trainable parameters
            if not param.requires_grad:
                continue
                
            param.data.zero_()
            
            for i, model in enumerate(models):
                for m_param_name, m_param in model.named_parameters():
                    if m_param_name == param_name and m_param.requires_grad:
                        param.data += m_param.data * weights[i]
                        break
        
        # Apply small momentum from global model for stability
        momentum = 0.1
        for new_param, old_param in zip(ensemble_model.parameters(), global_model.parameters()):
            if new_param.requires_grad:
                new_param.data = (1 - momentum) * new_param.data + momentum * old_param.data
        
        return ensemble_model
    
    else:
        logger.warning(f"Unknown aggregation type '{aggregation_type}'. Using weighted averaging.")
        return dp_fedguard_aggregate(client_models, global_model, trust_scores, 'weighted_averaging')
    
    # Apply momentum from previous global model for stability
    # Particularly important when we have high variance in client updates
    momentum_factor = 0.1
    if round_num > 5:
        # Increase momentum in later rounds for stability
        trust_variance = np.var(trust_scores)
        momentum_factor = min(0.3, 0.1 + trust_variance * 2)
    
    for new_param, old_param in zip(new_global_model.parameters(), global_model.parameters()):
        if new_param.requires_grad:
            new_param.data = (1 - momentum_factor) * new_param.data + momentum_factor * old_param.data
    
    # Log aggregation time
    logger.debug(f"Aggregation completed in {time.time() - start_time:.4f}s")
    
    return new_global_model

def fedprox(client_models, global_model, client_sample_sizes=None, mu=0.01):
    """
    FedProx aggregation - standard FedAvg with proximal term
    
    Args:
        client_models: List of client models
        global_model: Current global model
        client_sample_sizes: Number of samples for each client (for weighted averaging)
        mu: Proximal term weight
        
    Returns:
        new_global_model: Updated global model
    """
    # Start with standard FedAvg
    new_global_model = fedavg(client_models, client_sample_sizes)
    
    # Apply proximal term - pull slightly back toward original global model
    with torch.no_grad():
        for new_param, global_param in zip(new_global_model.parameters(), global_model.parameters()):
            if new_param.requires_grad:
                # Move slightly back toward the original global model
                new_param.data = (1 - mu) * new_param.data + mu * global_param.data
    
    return new_global_model

def adaptive_trust_aggregation(client_models, global_model, trust_scores, round_num=1):
    """
    Adaptive trust-based aggregation that selects the best defense
    based on the current trust distribution
    
    Args:
        client_models: List of client models
        global_model: Current global model
        trust_scores: Trust scores for each client
        round_num: Current round number
        
    Returns:
        new_global_model: Updated global model
    """
    num_clients = len(client_models)
    
    # Analyze trust distribution
    avg_trust = sum(trust_scores) / num_clients if num_clients > 0 else 0
    min_trust = min(trust_scores) if trust_scores else 0
    trusted_ratio = sum(1 for t in trust_scores if t > 0.7) / num_clients if num_clients > 0 else 0
    untrusted_ratio = sum(1 for t in trust_scores if t < 0.3) / num_clients if num_clients > 0 else 0
    
    # Log trust distribution
    logger.info(f"Trust analysis: avg={avg_trust:.3f}, min={min_trust:.3f}, " +
                f"trusted={trusted_ratio:.2f}, untrusted={untrusted_ratio:.2f}")
    
    # Apply layer-wise aggregation approach
    new_global_model = copy.deepcopy(global_model)
    
    # For each parameter, apply different aggregation strategies
    for param_name, param in new_global_model.named_parameters():
        # Skip non-trainable parameters
        if not param.requires_grad:
            continue
            
        # Identify layer type
        is_classifier = any(keyword in param_name for keyword in 
                           ['classifier', 'fc', 'linear', 'out'])
        is_early_layer = any(keyword in param_name for keyword in 
                            ['conv1', 'layer1', 'block1'])
        
        # Collect parameters from all clients
        params = []
        for i, client_model in enumerate(client_models):
            for client_param_name, client_param in client_model.named_parameters():
                if client_param_name == param_name:
                    params.append({
                        'client_idx': i,
                        'param': client_param.data,
                        'trust': trust_scores[i]
                    })
                    break
        
        # Apply different strategies based on layer type and trust distribution
        if is_classifier:
            # Classifier layers - more susceptible to attacks
            if untrusted_ratio > 0.3:
                # High proportion of untrusted clients - use robust method
                layer_params = torch.stack([p['param'] for p in params])
                if round_num > 5:
                    # In later rounds, use trimmed mean
                    trim_ratio = untrusted_ratio
                    # Sort and trim
                    sorted_params, _ = torch.sort(layer_params, dim=0)
                    trim_count = max(1, int(len(params) * trim_ratio))
                    trimmed_params = sorted_params[trim_count:-trim_count]
                    param.data = torch.mean(trimmed_params, dim=0)
                else:
                    # In early rounds, use median
                    param.data, _ = torch.median(layer_params, dim=0)
            else:
                # Use trust-weighted average with strict filtering
                param.data.zero_()
                total_weight = 0
                
                for p in params:
                    if p['trust'] < 0.3:
                        # Skip low-trust clients for classifier layers
                        continue
                    
                    # Apply quadratic weighting (emphasize high trust)
                    weight = p['trust'] ** 2
                    param.data += weight * p['param']
                    total_weight += weight
                
                if total_weight > 0:
                    param.data /= total_weight
                else:
                    # Fallback to median if all clients filtered
                    layer_params = torch.stack([p['param'] for p in params])
                    param.data, _ = torch.median(layer_params, dim=0)
        
        elif is_early_layer:
            # Early layers - less susceptible but still need protection
            if untrusted_ratio > 0.5:
                # Majority untrusted - use median
                layer_params = torch.stack([p['param'] for p in params])
                param.data, _ = torch.median(layer_params, dim=0)
            else:
                # Use trust-weighted average with moderate filtering
                param.data.zero_()
                total_weight = 0
                
                for p in params:
                    # Apply minimum trust threshold
                    if p['trust'] < 0.2:
                        continue
                    
                    # Linear weighting
                    weight = p['trust']
                    param.data += weight * p['param']
                    total_weight += weight
                
                if total_weight > 0:
                    param.data /= total_weight
                else:
                    # Fallback to average if all clients filtered
                    layer_params = torch.stack([p['param'] for p in params])
                    param.data = torch.mean(layer_params, dim=0)
        
        else:
            # Middle layers - use balanced approach
            if untrusted_ratio > 0.4:
                # Use trimmed mean
                layer_params = torch.stack([p['param'] for p in params])
                trim_ratio = untrusted_ratio / 2  # Less aggressive trimming
                
                # Sort and trim
                sorted_params, _ = torch.sort(layer_params, dim=0)
                trim_count = max(1, int(len(params) * trim_ratio))
                if trim_count < len(params) / 2:
                    trimmed_params = sorted_params[trim_count:-trim_count]
                    param.data = torch.mean(trimmed_params, dim=0)
                else:
                    # Standard weighted average
                    param.data.zero_()
                    total_weight = 0
                    
                    for p in params:
                        # Apply minimum trust threshold
                        if p['trust'] < 0.1:
                            continue
                        
                        weight = p['trust']
                        param.data += weight * p['param']
                        total_weight += weight
                    
                    if total_weight > 0:
                        param.data /= total_weight
                    else:
                        # Fallback to average
                        layer_params = torch.stack([p['param'] for p in params])
                        param.data = torch.mean(layer_params, dim=0)
    
    # Apply momentum from previous global model for stability
    momentum = min(0.3, 0.1 + 0.02 * round_num)  # Increases with rounds
    for new_param, old_param in zip(new_global_model.parameters(), global_model.parameters()):
        if new_param.requires_grad:
            new_param.data = (1 - momentum) * new_param.data + momentum * old_param.data
    
    return new_global_model

def create_defense_ensemble(client_models, global_model, trust_scores, beta=0.5):
    """
    Create an ensemble of the current global model and aggregated client models
    for improved stability and defense
    
    Args:
        client_models: List of client models
        global_model: Current global model
        trust_scores: Trust scores for each client
        beta: Weight for the current global model (0-1)
        
    Returns:
        ensemble_model: Ensemble model
    """
    # Aggregate client models with trust-based weighting
    aggregated_model = dp_fedguard_aggregate(
        client_models, 
        global_model, 
        trust_scores, 
        aggregation_type='weighted_averaging'
    )
    
    # Create a second model using robust aggregation
    robust_model = adaptive_trust_aggregation(
        client_models,
        global_model,
        trust_scores
    )
    
    # Create ensemble model
    ensemble_model = copy.deepcopy(global_model)
    
    # Adaptive ensemble weights based on trust distribution
    avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.5
    min_trust = min(trust_scores) if trust_scores else 0.0
    
    # Adjust weights based on trust
    if min_trust < 0.2:
        # At least one highly suspicious client - rely more on robust model
        weight_global = 0.2
        weight_aggregated = 0.3
        weight_robust = 0.5
    elif avg_trust < 0.6:
        # Moderate trust - balanced weights
        weight_global = 0.3
        weight_aggregated = 0.3
        weight_robust = 0.4
    else:
        # High trust - prefer weighted aggregation
        weight_global = 0.2
        weight_aggregated = 0.6
        weight_robust = 0.2
    
    # Apply weighted aggregation by layer
    for param_name, param in ensemble_model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Get params from each model
        global_param = global_model.state_dict()[param_name]
        
        # Find corresponding parameters in other models
        for agg_param_name, agg_param in aggregated_model.named_parameters():
            if agg_param_name == param_name:
                for rob_param_name, rob_param in robust_model.named_parameters():
                    if rob_param_name == param_name:
                        # Layer-specific weighting
                        is_classifier = any(keyword in param_name for keyword in 
                                          ['classifier', 'fc', 'linear', 'out'])
                        
                        if is_classifier:
                            # For classifier layers, emphasize robust model
                            w_global = weight_global - 0.1
                            w_aggregated = weight_aggregated - 0.1
                            w_robust = weight_robust + 0.2
                        else:
                            # Use default weights for other layers
                            w_global = weight_global
                            w_aggregated = weight_aggregated 
                            w_robust = weight_robust
                        
                        # Normalize weights
                        total = w_global + w_aggregated + w_robust
                        w_global /= total
                        w_aggregated /= total
                        w_robust /= total
                        
                        # Weighted combination
                        param.data = (
                            w_global * global_param + 
                            w_aggregated * agg_param.data + 
                            w_robust * rob_param.data
                        )
                        
                        break
                break
    
    return ensemble_model

def adaptive_aggregation_framework(client_models, global_model, trust_scores, 
                                  round_num=1, metrics=None):
    """
    Advanced adaptive aggregation framework that selects the best defense strategy
    based on current trust scores and historical performance
    
    Args:
        client_models: List of client models
        global_model: Current global model
        trust_scores: Trust scores for each client
        round_num: Current round number
        metrics: Historical performance metrics
        
    Returns:
        new_global_model: Updated global model
    """
    # Analyze trust distribution
    num_clients = len(client_models)
    avg_trust = sum(trust_scores) / num_clients if num_clients > 0 else 0
    min_trust = min(trust_scores) if trust_scores else 0
    num_suspicious = sum(1 for t in trust_scores if t < 0.3)
    suspicious_ratio = num_suspicious / num_clients if num_clients > 0 else 0
    
    # Log trust analysis
    logger.info(f"Trust analysis: avg={avg_trust:.3f}, min={min_trust:.3f}, " + 
               f"suspicious_ratio={suspicious_ratio:.2f}, round={round_num}")
    
    # Determine base strategy
    if suspicious_ratio > 0.4:
        # High proportion of suspicious clients - use robust aggregation
        logger.info("High attack ratio detected - using robust aggregation strategy")
        strategy = "robust"
    elif suspicious_ratio > 0.1 or min_trust < 0.2:
        # Moderate attack - use balanced approach
        logger.info("Moderate attack detected - using balanced aggregation strategy")
        strategy = "balanced"
    else:
        # Low/no attack - use standard approach
        logger.info("Low/no attack detected - using standard aggregation strategy")
        strategy = "standard"
    
    # Adaptive strategy refinement based on round number
    if round_num < 3:
        # Early rounds - be more conservative
        if strategy == "standard":
            strategy = "balanced"  # More cautious
    elif round_num > 10:
        # Later rounds - use more sophisticated ensemble
        if strategy == "balanced":
            strategy = "ensemble"
    
    # Apply selected strategy
    if strategy == "robust":
        # For high attack scenarios
        # Combine Krum and trimmed mean approaches
        if num_clients >= 5:
            # Use multi-Krum first to filter extreme outliers
            krum_model = krum(
                client_models, 
                global_model, 
                num_attackers=max(1, num_suspicious),
                multi_krum=True,
                num_to_select=max(1, num_clients - num_suspicious)
            )
            
            # Then apply adaptive trust aggregation
            return adaptive_trust_aggregation(client_models, krum_model, trust_scores, round_num)
        else:
            # With few clients, use robust method directly
            return adaptive_trust_aggregation(client_models, global_model, trust_scores, round_num)
    
    elif strategy == "balanced":
        # For moderate attack scenarios
        # Use trust-weighted averaging with enhanced filtering
        return dp_fedguard_aggregate(
            client_models,
            global_model,
            trust_scores,
            aggregation_type='trimmed_mean',
            round_num=round_num
        )
    
    elif strategy == "ensemble":
        # For sophisticated scenarios
        # Create an ensemble with multiple aggregation methods
        return create_defense_ensemble(client_models, global_model, trust_scores)
    
    else:  # "standard"
        # For low/no attack scenarios
        # Use basic trust-weighted averaging
        return dp_fedguard_aggregate(
            client_models,
            global_model,
            trust_scores,
            aggregation_type='weighted_averaging',
            round_num=round_num
        )

if __name__ == "__main__":
    # Simple test for the defense functions
    import torch.nn as nn
    
    # Create dummy models for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.fc(x)
    
    # Create a global model and some client models
    global_model = SimpleModel()
    client_models = [SimpleModel() for _ in range(5)]
    
    # Simulate some parameter updates
    for i, model in enumerate(client_models):
        with torch.no_grad():
            # Normal clients
            if i < 4:
                model.fc.weight.data = global_model.fc.weight.data + torch.randn_like(global_model.fc.weight.data) * 0.1
                model.fc.bias.data = global_model.fc.bias.data + torch.randn_like(global_model.fc.bias.data) * 0.1
            # Attacker
            else:
                model.fc.weight.data = global_model.fc.weight.data + torch.randn_like(global_model.fc.weight.data) * 2.0
                model.fc.bias.data = global_model.fc.bias.data + torch.randn_like(global_model.fc.bias.data) * 2.0
    
    # Test enhanced trust score calculation
    trust_scores, client_history = calculate_client_trust_scores(client_models, global_model)
    print("Enhanced trust scores:", trust_scores)
    
    # Test adaptive aggregation framework
    new_global_model = adaptive_aggregation_framework(client_models, global_model, trust_scores)
    
    # Calculate distances to original global model
    distances = []
    for i, model in enumerate(client_models):
        dist = calculate_update_norm(model, global_model)
        distances.append(dist)
    
    global_distance = calculate_update_norm(new_global_model, global_model)
    
    print("Client distances:", distances)
    print("New global model distance:", global_distance)
    
    print("Enhanced defense testing completed!")