import torch
import numpy as np
import copy
from collections import defaultdict
import logging
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dp_fedguard_enhanced")

class EnhancedDPFedGuard:
    """Enhanced DP-FedGuard with advanced trust calculation and aggregation"""
    
    def __init__(self, num_clients, device, use_dp=True, noise_multiplier=1.0, max_grad_norm=1.0):
        """Initialize the enhanced DP-FedGuard"""
        self.num_clients = num_clients
        self.device = device
        self.use_dp = use_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        # Initialize trust scores
        self.trust_scores = [1.0] * num_clients
        self.current_round = 0
        
        # Initialize client history for temporal analysis
        self.client_history = {
            'update_norms': [[] for _ in range(num_clients)],
            'cosine_similarities': [[] for _ in range(num_clients)],
            'suspected_attacks': [0] * num_clients,
            'trust_scores': [[] for _ in range(num_clients)],
            'contribution_quality': [[] for _ in range(num_clients)]
        }
        
        # Thresholds for trust management
        self.trust_thresholds = {
            'high_trust': 0.8,  # Clients above this are highly trusted
            'medium_trust': 0.5,  # Clients above this have moderate trust
            'low_trust': 0.3,   # Clients below this are suspicious
            'minimum_trust': 0.1 # Even suspicious clients get this minimum weight
        }
        
        # Adaptive parameters that will change during training
        self.adaptive_params = {
            'memory_factor': 0.7,  # How much to remember from previous rounds
            'forgiveness_rate': 0.05,  # Rate at which to forgive previous bad behavior
            'penalty_rate': 0.1,  # Rate at which to penalize bad behavior
            'noise_scaling': 1.0,  # Scaling factor for noise based on trust
        }
    
    def calculate_trust_scores(self, client_models, global_model, round_num):
        """
        Calculate trust scores for each client using multiple sophisticated techniques
        
        Args:
            client_models: List of client models after training
            global_model: Global model before client training
            round_num: Current round number
            
        Returns:
            trust_scores: List of trust scores (0-1) for each client
        """
        start_time = time.time()
        self.current_round = round_num
        num_clients = len(client_models)
        
        # 1. Extract updates from each client
        updates = []
        update_norms = []
        flattened_updates = []
        
        for client_idx, client_model in enumerate(client_models):
            # Get all parameter updates
            client_update = []
            for (client_name, client_param), (global_name, global_param) in zip(
                client_model.named_parameters(), global_model.named_parameters()
            ):
                if client_name == global_name:
                    # Calculate update
                    param_update = client_param.data - global_param.data
                    client_update.append(param_update.flatten())
            
            # Concatenate all parameter updates
            update_tensor = torch.cat(client_update)
            updates.append(update_tensor)
            
            # Calculate update norm
            update_norm = torch.norm(update_tensor).item()
            update_norms.append(update_norm)
            
            # Store flattened update as numpy array for statistical analysis
            flattened_updates.append(update_tensor.cpu().numpy())
            
            # Update client history
            self.client_history['update_norms'][client_idx].append(update_norm)
        
        # 2. Multiple trust evaluation strategies
        
        # 2.1 Norm-based analysis with robust statistics
        median_norm = np.median(update_norms)
        mad = np.median([abs(norm - median_norm) for norm in update_norms])
        
        # Prevent division by zero
        if mad < 1e-8:
            mad = np.mean([abs(norm - median_norm) for norm in update_norms])
            if mad < 1e-8:
                mad = 1.0
        
        # Calculate Median Absolute Deviation (MAD) z-scores
        # More robust than standard z-scores for outlier detection
        norm_z_scores = [0.6745 * abs(norm - median_norm) / mad for norm in update_norms]
        
        # Convert to trust scores (higher z-score → lower trust)
        # Using sigmoid function centered at z=3 for smooth transition
        norm_trust_scores = [1.0 / (1.0 + np.exp(z - 3.0)) for z in norm_z_scores]
        
        # 2.2 Direction-based analysis with cosine similarity
        # Compute pairwise cosine similarities between client updates
        cosine_similarities = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                # Get flattened updates
                update_i = flattened_updates[i]
                update_j = flattened_updates[j]
                
                # Calculate cosine similarity
                dot_product = np.dot(update_i, update_j)
                norm_i = np.linalg.norm(update_i)
                norm_j = np.linalg.norm(update_j)
                
                # Prevent division by zero
                denominator = norm_i * norm_j
                if denominator < 1e-8:
                    similarity = 0.0
                else:
                    similarity = dot_product / denominator
                
                cosine_similarities[i, j] = similarity
                cosine_similarities[j, i] = similarity
        
        # Average similarity to other clients
        avg_similarities = np.sum(cosine_similarities, axis=1) / (num_clients - 1)
        
        # Update client history
        for i in range(num_clients):
            self.client_history['cosine_similarities'][i].append(avg_similarities[i])
        
        # Convert to trust scores
        # Higher similarity generally means more trustworthy
        # But very high similarity could indicate collusion
        # We use a bell curve with peak at moderate similarity
        similarity_trust_scores = []
        for sim in avg_similarities:
            # Normalize similarity from [-1, 1] to [0, 1]
            norm_sim = (sim + 1) / 2
            
            # Calculate trust based on normalized similarity
            # Peak trust at 0.5-0.7 similarity, lower for extreme values
            if norm_sim < 0.3:
                # Very low similarity - suspicious
                trust = norm_sim * 2  # Linear increase
            elif norm_sim > 0.9:
                # Very high similarity - potential collusion
                trust = 2.0 - norm_sim  # Linear decrease
            else:
                # Normal range - highest trust
                trust = 0.8 + (norm_sim - 0.6) * 0.4  # Peak around 0.6
            
            similarity_trust_scores.append(min(1.0, max(0.0, trust)))
        
        # 2.3 Layer-wise analysis (focusing on classifier layers)
        layer_trust_scores = []
        
        for client_idx, client_model in enumerate(client_models):
            layer_suspicion = 0.0
            
            for (client_name, client_param), (global_name, global_param) in zip(
                client_model.named_parameters(), global_model.named_parameters()
            ):
                if client_name != global_name:
                    continue
                
                # Focus on classifier layers which are often targeted
                is_classifier = any(keyword in client_name for keyword in 
                                    ['classifier', 'fc', 'linear', 'output'])
                
                if is_classifier:
                    # Calculate update for this layer
                    layer_update = client_param.data - global_param.data
                    layer_norm = torch.norm(layer_update).item()
                    
                    # Check if this layer's update is suspiciously large
                    if layer_norm > 5 * self.max_grad_norm:
                        layer_suspicion += 0.3
                    elif layer_norm > 3 * self.max_grad_norm:
                        layer_suspicion += 0.1
            
            # Convert suspicion to trust score
            layer_trust = max(0.0, 1.0 - layer_suspicion)
            layer_trust_scores.append(layer_trust)
        
        # 2.4 Advanced statistical analysis with dimensionality reduction
        if num_clients >= 3:  # Only meaningful with enough clients
            try:
                # Convert updates to format suitable for PCA
                stacked_updates = np.vstack([update.reshape(1, -1) for update in flattened_updates])
                
                # Standardize updates
                scaler = StandardScaler()
                standardized_updates = scaler.fit_transform(stacked_updates)
                
                # Apply PCA to reduce dimensionality while preserving variance
                n_components = min(num_clients - 1, 5)
                pca = PCA(n_components=n_components)
                reduced_updates = pca.fit_transform(standardized_updates)
                
                # Apply Isolation Forest for outlier detection
                # This uses the reduced representation for more efficient computation
                iso_forest = IsolationForest(contamination=0.25, random_state=42)
                outlier_scores = iso_forest.fit_predict(reduced_updates)
                
                # Convert to trust scores (1 for inliers, lower for outliers)
                isolation_trust_scores = [1.0 if score == 1 else 0.3 for score in outlier_scores]
                
                # Cluster analysis to find groups of clients
                kmeans = KMeans(n_clusters=min(3, num_clients), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(reduced_updates)
                
                # Count members in each cluster
                cluster_counts = np.bincount(cluster_labels)
                min_cluster_idx = np.argmin(cluster_counts)
                
                # Clients in smaller clusters are more suspicious
                cluster_trust_scores = []
                for label in cluster_labels:
                    if label == min_cluster_idx and cluster_counts[label] < num_clients / 3:
                        trust = 0.6  # Reduce trust for minority clusters
                    else:
                        trust = 1.0
                    cluster_trust_scores.append(trust)
            except Exception as e:
                # Fallback if statistical analysis fails
                logger.warning(f"Advanced statistical analysis failed: {e}")
                isolation_trust_scores = [1.0] * num_clients
                cluster_trust_scores = [1.0] * num_clients
        else:
            # With too few clients, don't use these methods
            isolation_trust_scores = [1.0] * num_clients
            cluster_trust_scores = [1.0] * num_clients
            
        # 3. Combine multiple signals with adaptive weighting
        weights = {
            'norm': 0.3,
            'similarity': 0.3,
            'layer': 0.2,
            'isolation': 0.1,
            'cluster': 0.1
        }
        
        # Adjust weights based on round number
        # Later rounds focus more on consistency and less on statistical outliers
        if round_num > 10:
            weights['norm'] = 0.25
            weights['similarity'] = 0.35
            weights['layer'] = 0.25
            weights['isolation'] = 0.1
            weights['cluster'] = 0.05
        
        # Calculate combined trust scores
        combined_trust_scores = []
        for i in range(num_clients):
            trust = (
                weights['norm'] * norm_trust_scores[i] +
                weights['similarity'] * similarity_trust_scores[i] +
                weights['layer'] * layer_trust_scores[i] +
                weights['isolation'] * isolation_trust_scores[i] +
                weights['cluster'] * cluster_trust_scores[i]
            )
            combined_trust_scores.append(min(1.0, max(0.0, trust)))
        
        # 4. Apply temporal analysis and pattern detection
        if round_num > 1:
            # Apply adaptive memory factor
            # Starts high and decreases over time to adapt to evolving attacks
            memory_factor = max(0.1, min(0.8, self.adaptive_params['memory_factor'] - 0.02 * round_num))
            self.adaptive_params['memory_factor'] = memory_factor
            
            final_trust_scores = []
            for i in range(num_clients):
                # Get current combined score
                current_score = combined_trust_scores[i]
                
                # Get previous score
                prev_score = self.trust_scores[i]
                
                # Check for suspicious patterns
                if len(self.client_history['trust_scores'][i]) >= 3:
                    recent_scores = self.client_history['trust_scores'][i][-3:]
                    
                    # Detect oscillating behavior (could be a strategic attacker)
                    if (max(recent_scores) - min(recent_scores) > 0.4 and
                        abs(recent_scores[0] - recent_scores[2]) < 0.1):
                        # Penalize oscillating behavior
                        current_score *= 0.8
                        logger.debug(f"Detected oscillating behavior for client {i+1}")
                
                # Apply temporally sensitive trust update
                if current_score < prev_score - 0.3:
                    # Sudden large drop - likely attack, respond quickly
                    trust = 0.3 * prev_score + 0.7 * current_score
                    
                    # Record suspicious activity
                    self.client_history['suspected_attacks'][i] += 1
                elif current_score > prev_score + 0.3:
                    # Sudden improvement - be cautious
                    trust = 0.7 * prev_score + 0.3 * current_score
                else:
                    # Normal change - use regular memory factor
                    trust = memory_factor * prev_score + (1 - memory_factor) * current_score
                
                final_trust_scores.append(trust)
                
                # Update client history
                self.client_history['trust_scores'][i].append(trust)
        else:
            # First round - use combined scores directly
            final_trust_scores = combined_trust_scores
            
            # Initialize history
            for i in range(num_clients):
                self.client_history['trust_scores'][i].append(final_trust_scores[i])
        
        # 5. Apply forgiveness and reputation mechanisms
        if round_num >= 3:
            for i in range(num_clients):
                # Get current score
                trust = final_trust_scores[i]
                
                # Apply forgiveness for consistently good behavior
                recent_history = self.client_history['trust_scores'][i][-3:]
                if min(recent_history) > 0.7 and trust < 0.9:
                    # Apply small boost for consistent good behavior
                    trust += self.adaptive_params['forgiveness_rate']
                    
                # Apply penalty for consistently bad behavior
                if max(recent_history) < 0.4:
                    # Reduce trust to minimum threshold
                    trust = max(self.trust_thresholds['minimum_trust'], 
                                trust - self.adaptive_params['penalty_rate'])
                
                # Ensure trust is in [0, 1]
                final_trust_scores[i] = min(1.0, max(0.0, trust))
        
        # Log calculation time for performance monitoring
        calc_time = time.time() - start_time
        logger.debug(f"Trust calculation completed in {calc_time:.4f}s")
        
        # Update instance trust scores
        self.trust_scores = final_trust_scores
        
        return final_trust_scores
    
    def filter_suspicious_gradients(self, client_model, global_model, trust_score):
        """
        Filter suspicious gradients with adaptive mechanisms
        
        Args:
            client_model: Client model after training
            global_model: Global model before client training
            trust_score: Trust score for this client
            
        Returns:
            filtered_model: Model with filtered gradients
        """
        filtered_model = copy.deepcopy(client_model)
        
        with torch.no_grad():
            for (client_name, client_param), (global_name, global_param) in zip(
                filtered_model.named_parameters(), global_model.named_parameters()
            ):
                if client_name != global_name:
                    continue
                
                # Calculate update
                update = client_param.data - global_param.data
                
                # Adaptive clipping based on trust score
                # Lower trust = more aggressive clipping
                trust_factor = max(0.1, trust_score)
                
                # Identify layer type for layer-specific handling
                is_classifier = any(x in client_name for x in ['classifier', 'fc', 'linear', 'out'])
                is_early_layer = any(x in client_name for x in ['conv1', 'layer1', 'block1'])
                
                # Layer-specific handling
                if is_classifier:
                    # Stricter filtering for classifier layers (often targeted)
                    if trust_score < self.trust_thresholds['medium_trust']:
                        # Low trust - apply aggressive gradient clipping
                        update_norm = torch.norm(update)
                        clip_threshold = self.max_grad_norm * trust_factor
                        
                        if update_norm > clip_threshold:
                            update = update * (clip_threshold / update_norm)
                            
                        # Additional scaling for very low trust
                        if trust_score < self.trust_thresholds['low_trust']:
                            update = update * trust_score * 2  # Further reduce impact
                
                elif is_early_layer:
                    # Early layers are less vulnerable but still need protection
                    if trust_score < self.trust_thresholds['low_trust']:
                        # For very suspicious clients, limit early layer updates
                        update_norm = torch.norm(update)
                        clip_threshold = self.max_grad_norm * 2  # More permissive
                        
                        if update_norm > clip_threshold:
                            update = update * (clip_threshold / update_norm)
                
                else:
                    # Middle layers - moderate approach
                    if trust_score < self.trust_thresholds['medium_trust']:
                        # Apply moderate clipping
                        update_norm = torch.norm(update)
                        clip_threshold = self.max_grad_norm * 1.5
                        
                        if update_norm > clip_threshold:
                            update = update * (clip_threshold / update_norm)
                
                # Apply filtered update
                client_param.data = global_param.data + update
        
        return filtered_model
    
    def aggregate(self, client_models, global_model, client_data_sizes=None):
        """
        Aggregate client models using enhanced DP-FedGuard with multi-tiered defense
        
        Args:
            client_models: List of client models after training
            global_model: Global model before client training
            client_data_sizes: Optional list of client data sizes for weighted avg
            
        Returns:
            new_global_model: Updated global model
        """
        # Calculate trust scores
        trust_scores = self.calculate_trust_scores(
            client_models, global_model, self.current_round)
        
        # Use more robust aggregation in early rounds
        if self.current_round < 3:
            logger.info(f"Using robust aggregation in early round {self.current_round}")
            return self.robust_aggregation(client_models, global_model, trust_scores)
        
        # Log trust scores
        logger.info("Client trust scores:")
        for i, trust in enumerate(trust_scores):
            logger.info(f"  Client {i+1}: {trust:.4f}")
        
        # Multi-tier defense based on trust distribution
        avg_trust = sum(trust_scores) / len(trust_scores)
        min_trust = min(trust_scores)
        num_trusted = sum(1 for t in trust_scores if t > self.trust_thresholds['high_trust'])
        num_suspicious = sum(1 for t in trust_scores if t < self.trust_thresholds['low_trust'])
        
        # Select appropriate aggregation strategy based on trust landscape
        if num_suspicious > len(client_models) / 3:
            # High number of suspicious clients - use robust aggregation
            logger.info("High number of suspicious clients detected. Using robust aggregation.")
            return self.robust_aggregation(client_models, global_model, trust_scores)
        elif min_trust < 0.2:
            # At least one highly suspicious client - use filtered aggregation
            logger.info("Highly suspicious client detected. Using filtered weighted aggregation.")
            return self.filtered_weighted_aggregation(client_models, global_model, trust_scores)
        else:
            # Normal scenario - use adaptive weighted aggregation
            logger.info("Using standard trust-weighted aggregation.")
            return self.adaptive_weighted_aggregation(client_models, global_model, trust_scores, client_data_sizes)
    
    def adaptive_weighted_aggregation(self, client_models, global_model, trust_scores, client_data_sizes=None):
        """
        Trust-based weighted aggregation with adaptive mechanisms
        
        Args:
            client_models: List of client models
            global_model: Global model before training
            trust_scores: Trust scores for each client
            client_data_sizes: Optional client data sizes
            
        Returns:
            new_global_model: Updated global model
        """
        new_global_model = copy.deepcopy(global_model)
        
        # Calculate adaptive weights
        # Transform trust scores using sigmoid to sharpen differences
        weights = []
        for trust in trust_scores:
            # Apply sigmoid transformation centered at 0.5
            # Higher values increase the gap between high and low trust
            sharpness = 4.0 + min(4.0, 0.5 * self.current_round)  # Increases over rounds
            sigmoid_input = sharpness * (trust - 0.5)
            sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
            
            # Ensure minimum weight for low trust
            weight = max(self.trust_thresholds['minimum_trust'], sigmoid_output)
            weights.append(weight)
        
        # Incorporate client data sizes if provided
        if client_data_sizes is not None:
            # Combine trust and data size
            combined_weights = []
            total_size = sum(client_data_sizes)
            
            for i, (weight, size) in enumerate(zip(weights, client_data_sizes)):
                # Data size factor (normalized)
                size_factor = size / total_size
                
                # Combined weight (70% trust, 30% data size)
                combined_weight = 0.7 * weight + 0.3 * size_factor
                combined_weights.append(combined_weight)
            
            weights = combined_weights
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Fallback to equal weighting
            weights = [1.0 / len(client_models)] * len(client_models)
        
        # Apply layer-wise aggregation with filtering
        for param_name, param in new_global_model.named_parameters():
            # Skip non-trainable parameters
            if not param.requires_grad:
                continue
            
            # Reset parameter
            param.data.zero_()
            
            # Layer-specific logic
            is_classifier = any(keyword in param_name for keyword in 
                               ['classifier', 'fc', 'linear', 'out'])
            
            # Apply weighted average with potential filtering
            for i, client_model in enumerate(client_models):
                for client_param_name, client_param in client_model.named_parameters():
                    if client_param_name == param_name:
                        # Apply different strategy for classifier layers
                        if is_classifier and trust_scores[i] < self.trust_thresholds['medium_trust']:
                            # Get update
                            update = client_param.data - global_model.state_dict()[param_name]
                            
                            # Scale down updates from less trusted clients for critical layers
                            scaling = max(0.2, trust_scores[i] / self.trust_thresholds['high_trust'])
                            param.data += weights[i] * (global_model.state_dict()[param_name] + update * scaling)
                        else:
                            # Normal weighted aggregation for other layers
                            param.data += weights[i] * client_param.data
                        break
        
        # Apply momentum from previous global model for stability
        # Start with a small factor that increases with client variance
        momentum_factor = 0.1
        if self.current_round > 5:
            # Increase momentum for stability in later rounds
            trust_variance = np.var(trust_scores)
            momentum_factor = min(0.3, 0.1 + trust_variance)
        
        for new_param, old_param in zip(new_global_model.parameters(), global_model.parameters()):
            new_param.data = (1 - momentum_factor) * new_param.data + momentum_factor * old_param.data
        
        return new_global_model
    
    def filtered_weighted_aggregation(self, client_models, global_model, trust_scores):
        """
        Weighted aggregation with explicit filtering of suspicious updates
        
        Args:
            client_models: List of client models
            global_model: Global model before training
            trust_scores: Trust scores for each client
            
        Returns:
            new_global_model: Updated global model
        """
        # First apply gradient filtering to each client model
        filtered_models = []
        for i, model in enumerate(client_models):
            filtered_model = self.filter_suspicious_gradients(
                model, global_model, trust_scores[i])
            filtered_models.append(filtered_model)
        
        # Then apply weighted aggregation
        return self.adaptive_weighted_aggregation(filtered_models, global_model, trust_scores)
    
    def robust_aggregation(self, client_models, global_model, trust_scores):
        """
        Robust aggregation for high-attack scenarios
        Uses multi-krum and trimmed mean with trust-based filtering
        
        Args:
            client_models: List of client models
            global_model: Global model before training
            trust_scores: Trust scores for each client
            
        Returns:
            new_global_model: Updated global model
        """
        new_global_model = copy.deepcopy(global_model)
        num_clients = len(client_models)
        
        # Identify suspected number of attackers
        suspected_attackers = sum(1 for t in trust_scores if t < self.trust_thresholds['low_trust'])
        num_attackers = max(1, suspected_attackers)
        
        # Apply different strategies for different parameter types
        for param_idx, (param_name, param) in enumerate(new_global_model.named_parameters()):
            # Skip non-trainable parameters
            if not param.requires_grad:
                continue
            
            # Get parameter updates from all clients
            updates = []
            for i, client_model in enumerate(client_models):
                for client_param_name, client_param in client_model.named_parameters():
                    if client_param_name == param_name:
                        updates.append({
                            'client_idx': i,
                            'update': client_param.data,
                            'trust': trust_scores[i]
                        })
                        break
            
            # Identify layer type
            is_classifier = any(keyword in param_name for keyword in 
                               ['classifier', 'fc', 'linear', 'out'])
            is_early_layer = any(keyword in param_name for keyword in 
                                ['conv1', 'layer1', 'block1'])
            
            # Use different aggregation strategies for different layer types
            if is_classifier:
                # Classifier layers - use trimmed mean with trust-based trimming
                # Sort updates by trust score
                sorted_updates = sorted(updates, key=lambda x: x['trust'])
                
                # Determine trim ratio based on trust distribution
                trim_ratio = min(0.4, max(0.1, suspected_attackers / num_clients))
                trim_count = max(1, int(len(updates) * trim_ratio))
                
                # Remove lowest trust updates
                trimmed_updates = sorted_updates[trim_count:]
                
                if trimmed_updates:
                    # Stack remaining updates
                    stacked_updates = torch.stack([u['update'] for u in trimmed_updates])
                    
                    # Calculate trimmed mean
                    mean_update = torch.mean(stacked_updates, dim=0)
                    param.data.copy_(mean_update)
                else:
                    # If all updates were trimmed, keep original parameter
                    param.data.copy_(global_model.state_dict()[param_name])
            
            elif is_early_layer:
                # Early layers - use median (more robust than mean)
                stacked_updates = torch.stack([u['update'] for u in updates])
                median_update, _ = torch.median(stacked_updates, dim=0)
                param.data.copy_(median_update)
            
            else:
                # Middle layers - use trust-weighted average with clipping
                param.data.zero_()
                
                # Apply trust-weighted average with gradient clipping
                total_weight = 0
                
                for update_info in updates:
                    # Skip very low trust clients
                    if update_info['trust'] < self.trust_thresholds['minimum_trust']:
                        continue
                    
                    # Get update
                    update = update_info['update']
                    
                    # Add weighted update
                    weight = update_info['trust']
                    param.data += weight * update
                    total_weight += weight
                
                # Normalize
                if total_weight > 0:
                    param.data /= total_weight
                else:
                    # If all clients were filtered, use global model
                    param.data.copy_(global_model.state_dict()[param_name])
        
        # Apply defensive stabilization
        # Add small momentum to prevent large jumps
        momentum_factor = 0.2
        for new_param, old_param in zip(new_global_model.parameters(), global_model.parameters()):
            new_param.data = (1 - momentum_factor) * new_param.data + momentum_factor * old_param.data
        
        return new_global_model
    
    def get_noise_multipliers(self, trust_scores):
        """
        Get dynamic noise multipliers based on trust scores
        
        Args:
            trust_scores: Trust scores for each client
            
        Returns:
            noise_multipliers: List of noise multipliers for each client
        """
        noise_multipliers = []
        
        for trust in trust_scores:
            # Dynamic scaling based on trust
            # Low trust = more noise, high trust = less noise
            if trust > self.trust_thresholds['high_trust']:
                # High trust - reduce noise
                noise = self.noise_multiplier * 0.7
            elif trust < self.trust_thresholds['low_trust']:
                # Low trust - increase noise
                noise = self.noise_multiplier * 2.0
            else:
                # Medium trust - scale linearly
                trust_factor = (trust - self.trust_thresholds['low_trust']) / (
                    self.trust_thresholds['high_trust'] - self.trust_thresholds['low_trust'])
                noise = self.noise_multiplier * (2.0 - 1.3 * trust_factor)
            
            # Scale by adaptive factor
            noise *= self.adaptive_params['noise_scaling']
            
            noise_multipliers.append(noise)
        
        return noise_multipliers
    
    def update_adaptive_parameters(self):
        """Update adaptive parameters based on current state"""
        # Adjust memory factor based on round number
        self.adaptive_params['memory_factor'] = max(0.3, 0.7 - 0.02 * self.current_round)
        
        # Adjust forgiveness rate based on trust distribution
        avg_trust = sum(self.trust_scores) / len(self.trust_scores)
        if avg_trust < 0.5:
            # Lower forgiveness when overall trust is low
            self.adaptive_params['forgiveness_rate'] = 0.02
        else:
            # Higher forgiveness when overall trust is high
            self.adaptive_params['forgiveness_rate'] = 0.05
        
        # Adjust noise scaling based on attack detection
        num_suspicious = sum(1 for t in self.trust_scores if t < self.trust_thresholds['low_trust'])
        if num_suspicious > len(self.trust_scores) / 3:
            # Increase noise when many attackers detected
            self.adaptive_params['noise_scaling'] = 1.5
        else:
            # Normal noise scaling
            self.adaptive_params['noise_scaling'] = 1.0
    
    def fed_median(self, client_models, global_model):
        """
        Coordinate-wise median aggregation (fallback method)
        
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
    
    def adaptive_layer_aggregation(self, client_models, global_model, trust_scores):
        """
        Layer-specific aggregation with different strategies per layer
        
        Args:
            client_models: List of client models
            global_model: Current global model
            trust_scores: Trust scores for each client
            
        Returns:
            new_global_model: Updated global model
        """
        new_global_model = copy.deepcopy(global_model)
        
        # For each parameter in the model, apply different aggregation strategies
        for param_idx, (param_name, param) in enumerate(new_global_model.named_parameters()):
            # Skip non-trainable parameters
            if not param.requires_grad:
                continue
            
            # Determine layer type and apply appropriate strategy
            if any(keyword in param_name for keyword in ['classifier', 'fc', 'linear', 'out']):
                # Classifier layers - use robust methods (trimmed mean)
                self.aggregate_classifier_layer(param_name, param, client_models, global_model, trust_scores)
                
            elif any(keyword in param_name for keyword in ['conv1', 'layer1', 'block1']):
                # Early layers - use weighted averaging with less strict filtering
                self.aggregate_early_layer(param_name, param, client_models, global_model, trust_scores)
                
            else:
                # Middle layers - use balanced approach
                self.aggregate_middle_layer(param_name, param, client_models, global_model, trust_scores)
        
        return new_global_model
    
    def aggregate_classifier_layer(self, param_name, param, client_models, global_model, trust_scores):
        """Aggregate classifier layer with robust methods"""
        # Get updates for this parameter
        updates = []
        for i, client_model in enumerate(client_models):
            for client_param_name, client_param in client_model.named_parameters():
                if client_param_name == param_name:
                    updates.append({
                        'client_idx': i,
                        'param': client_param.data,
                        'trust': trust_scores[i]
                    })
                    break
        
        # Sort by trust score
        updates.sort(key=lambda x: x['trust'])
        
        # Determine trimming threshold based on trust distribution
        low_trust_count = sum(1 for u in updates if u['trust'] < self.trust_thresholds['low_trust'])
        trim_count = max(1, low_trust_count)
        
        # Trim lowest trust updates
        if trim_count < len(updates):
            trimmed_updates = updates[trim_count:]
            
            # Apply trust-weighted average on remaining updates
            param.data.zero_()
            total_weight = sum(u['trust'] for u in trimmed_updates)
            
            for update in trimmed_updates:
                weight = update['trust'] / total_weight
                param.data += weight * update['param']
        else:
            # If all would be trimmed, use median as fallback
            stacked_params = torch.stack([u['param'] for u in updates])
            median_param, _ = torch.median(stacked_params, dim=0)
            param.data.copy_(median_param)
    
    def aggregate_early_layer(self, param_name, param, client_models, global_model, trust_scores):
        """Aggregate early layer with less strict filtering"""
        # Early layers are less vulnerable to attacks
        # Use weighted average with gradient clipping
        
        # Get updates for this parameter
        updates = []
        for i, client_model in enumerate(client_models):
            for client_param_name, client_param in client_model.named_parameters():
                if client_param_name == param_name:
                    # Calculate update
                    update = client_param.data - global_model.state_dict()[param_name]
                    
                    # Apply clipping for low trust clients
                    if trust_scores[i] < self.trust_thresholds['medium_trust']:
                        update_norm = torch.norm(update)
                        clip_threshold = self.max_grad_norm * 2.0  # More permissive
                        
                        if update_norm > clip_threshold:
                            update = update * (clip_threshold / update_norm)
                    
                    updates.append({
                        'client_idx': i,
                        'update': update,
                        'trust': trust_scores[i]
                    })
                    break
        
        # Apply trust-weighted average
        param.data.zero_()
        param.data.copy_(global_model.state_dict()[param_name])  # Start with global param
        
        total_weight = sum(max(0.2, u['trust']) for u in updates)  # Minimum weight of 0.2
        
        for update in updates:
            weight = max(0.2, update['trust']) / total_weight
            param.data += weight * update['update']
    
    def aggregate_middle_layer(self, param_name, param, client_models, global_model, trust_scores):
        """Aggregate middle layer with balanced approach"""
        # Middle layers - use balanced approach between robust and weighted
        
        # Get parameters for this layer
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
        
        # Separate into high and low trust clients
        high_trust = [p for p in params if p['trust'] >= self.trust_thresholds['medium_trust']]
        low_trust = [p for p in params if p['trust'] < self.trust_thresholds['medium_trust']]
        
        if high_trust:
            # If we have high trust clients, use their weighted average
            param.data.zero_()
            total_weight = sum(p['trust'] for p in high_trust)
            
            for p in high_trust:
                weight = p['trust'] / total_weight
                param.data += weight * p['param']
                
            # Incorporate low trust clients with limited impact
            if low_trust:
                # Calculate average from low trust clients
                low_trust_params = torch.stack([p['param'] for p in low_trust])
                low_trust_avg = torch.mean(low_trust_params, dim=0)
                
                # Add with small weight
                low_trust_weight = 0.2  # Fixed small weight
                param.data = (1 - low_trust_weight) * param.data + low_trust_weight * low_trust_avg
        
        else:
            # If no high trust clients, use median as robust estimator
            all_params = torch.stack([p['param'] for p in params])
            median_param, _ = torch.median(all_params, dim=0)
            param.data.copy_(median_param)
        