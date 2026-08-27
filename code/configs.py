# Create a configs.py file to define experiment configurations
experiment_configs = {
    # Baseline comparisons
    "baselines": [
        {"defense_type": "none", "use_dp": False, "attack_fraction": 0.0},  # No defense, no attack
        {"defense_type": "none", "use_dp": False, "attack_fraction": 0.2},  # No defense, with attack
        {"defense_type": "median", "use_dp": False, "attack_fraction": 0.2},  # Median defense
        {"defense_type": "krum", "use_dp": False, "attack_fraction": 0.2},   # Krum defense
        {"defense_type": "trimmed_mean", "use_dp": False, "attack_fraction": 0.2}  # Trimmed mean
    ],
    
    # Your method with different attack scenarios
    "attack_scenarios": [
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.0},   # No attack
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.1},   # 10% attackers
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2},   # 20% attackers
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.4}    # 40% attackers
    ],
    
    # Attack type comparisons
    "attack_types": [
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "attack_type": "label_flipping"},
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "attack_type": "data_poisoning"},
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "attack_type": "model_poisoning"}
    ],
    
    # Client distribution comparisons
    "client_distributions": [
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "distribution_type": "pathological"},
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "distribution_type": "practical"},
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "distribution_type": "balanced"}
    ],
    
    # Privacy-utility trade-off
    "privacy_utility": [
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "noise_multiplier": 0.1},
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "noise_multiplier": 0.5},
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "noise_multiplier": 1.0},
        {"defense_type": "dp_fedguard", "use_dp": True, "attack_fraction": 0.2, "noise_multiplier": 2.0}
    ]
}