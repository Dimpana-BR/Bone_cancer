import numpy as np

# Set a consistent random seed for reproducibility in subset selection
RANDOM_SEED = 42

def prepare_for_quantum(X_test_scaled, y_test, num_samples=10, num_features=2):
    """
    Prepares a smaller subset of the scaled test data for Qiskit simulation.
    
    Quantum simulations are resource-intensive, so we typically reduce 
    the number of features and samples significantly.
    
    Args:
        X_test_scaled (np.ndarray): Full scaled features.
        y_test (np.ndarray): Full true labels.
        num_samples (int): Number of samples to use for QSVM testing.
        num_features (int): Number of features to keep (must be <= X_test_scaled.shape[1]).
        
    Returns:
        tuple: (X_quantum: np.ndarray, y_quantum: np.ndarray)
    """
    
    if num_features > X_test_scaled.shape[1]:
        print("Warning: Requested features exceed available features. Using all available features.")
        num_features = X_test_scaled.shape[1]
    
    np.random.seed(RANDOM_SEED)
    
    # 1. Randomly select a subset of indices for samples
    sample_indices = np.random.choice(
        len(X_test_scaled), 
        min(num_samples, len(X_test_scaled)), 
        replace=False
    )
    
    X_subset = X_test_scaled[sample_indices]
    y_subset = y_test[sample_indices]

    # 2. Select features with highest variance for better representation
    variances = np.var(X_subset, axis=0)
    top_feature_indices = np.argsort(variances)[-num_features:]  # Select top N features by variance
    X_quantum = X_subset[:, top_feature_indices]
    
    print(f"Quantum data prepared. Samples: {X_quantum.shape[0]}, Features: {X_quantum.shape[1]}")
    
    return X_quantum, y_subset
