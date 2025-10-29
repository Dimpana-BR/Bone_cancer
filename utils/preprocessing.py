import numpy as np
import os
import joblib
import shutil
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def setup_project_environment(base_dir="."):
    """
    Creates the necessary folder structure for the project and ensures 
    the results directory is clean.
    """
    print("Setting up project environment and folders...")
    
    # Define required directories
    dirs = [
        "images", "train", "test", "valid", 
        "quantum_model", "comparison", "utils", "results"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        
    # Clean up results folder
    results_dir = os.path.join(base_dir, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print("Setup complete. Directories created.")

def simulate_classical_artifacts(n_features=10, n_samples=200):
    """
    Simulates the classical training process to generate dummy 
    svm_model.pkl, scaler.pkl, and saves the X_test and y_test data.
    
    Args:
        n_features (int): Number of features to simulate (e.g., flattened image pixels).
        n_samples (int): Total number of samples for simulation.
    """
    print("Simulating classical artifacts (SVM model, scaler, and test data)...")
    
    # 1. Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_redundant=0, 
        n_informative=n_features,
        n_classes=2, 
        random_state=42
    )

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 3. Scale data (and save scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # 4. Train model (and save model)
    # Using a simple SVC for simulation
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, "svm_model.pkl")

    # 5. Save the scaled test data for all subsequent scripts to use
    np.save(os.path.join("test", "X_test_scaled.npy"), X_test_scaled)
    np.save(os.path.join("test", "y_test.npy"), y_test)
    
    print(f"Artifacts simulated. Scaled test data shape: {X_test_scaled.shape}")
    print("Dummy svm_model.pkl and scaler.pkl created.")


def load_test_data():
    """
    Loads the scaled test features (X_test_scaled) and true labels (y_test) 
    from the 'test' directory.
    
    Returns:
        tuple: (X_test_scaled: np.ndarray, y_test: np.ndarray)
    """
    try:
        X_test_scaled = np.load(os.path.join("test", "X_test_scaled.npy"))
        y_test = np.load(os.path.join("test", "y_test.npy"))
        return X_test_scaled, y_test
    except FileNotFoundError:
        print("Error: Test data files not found. Run 'simulate_classical_artifacts()' first.")
        return None, None
