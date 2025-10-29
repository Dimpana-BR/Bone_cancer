import os
import numpy as np

# Relative Imports for Modular Components
from utils.preprocessing import setup_project_environment, simulate_classical_artifacts, load_test_data
from predict_xray import predict_classical
from quantum_model.quantum_preprocess import prepare_for_quantum
from quantum_model.quantum_svm_qiskit import train_and_predict_qsvm, split_quantum_data
from comparison.compare_results import compare_and_plot

# --- Configuration ---
# Set the number of features and samples for the *classical* simulation
CLASSICAL_FEATURES = 10
CLASSICAL_SAMPLES = 500 # 30% split = 150 test samples for more data

# Set the reduced features and samples for the *quantum* simulation
QUANTUM_FEATURES = 10 # Use all features for better representation
QUANTUM_SAMPLES = 500 # Use full test size for quantum

def main():
    """
    Main function to run the Bone Cancer Detection comparison project.
    1. Setup environment and simulate required artifacts.
    2. Run Classical SVM prediction.
    3. Prepare and run Quantum SVM prediction.
    4. Compare and save results.
    """
    print("--- Starting Quantum + Classical Bone Cancer Detection Project ---")
    
    # 1. Setup and Simulation
    setup_project_environment()
    simulate_classical_artifacts(
        n_features=CLASSICAL_FEATURES, 
        n_samples=CLASSICAL_SAMPLES
    )

    # 2. Load the common, scaled test data
    X_test_scaled, y_test = load_test_data()
    if X_test_scaled is None:
        return # Exit if data loading failed

    # --- CLASSICAL MODEL EXECUTION ---
    print("\n--- Running Classical SVM ---")
    classical_acc, classical_y_pred, classical_cm = predict_classical(
        X_test_scaled, y_test
    )
    
    if classical_acc is None:
        print("Classical model execution failed. Exiting.")
        return

    # --- QUANTUM MODEL EXECUTION ---
    
    # Prepare the resource-constrained quantum subset
    X_q, y_q = prepare_for_quantum(
        X_test_scaled, 
        y_test, 
        num_samples=QUANTUM_SAMPLES, 
        num_features=QUANTUM_FEATURES
    )
    
    # Split the small quantum subset into train/test for QSVC
    X_train_q, y_train_q, X_test_q, y_test_q = split_quantum_data(X_q, y_q)
    
    quantum_acc, quantum_y_pred, quantum_cm = train_and_predict_qsvm(
        X_train_q, y_train_q, X_test_q, y_test_q
    )
    
    if quantum_acc is None:
        print("Quantum model execution failed. Exiting.")
        return

    # --- COMPARISON AND RESULTS ---
    print("\n--- Generating Comparison Reports ---")
    compare_and_plot(classical_acc, classical_cm, quantum_acc, quantum_cm)
    
    # --- FINAL SUMMARY ---
    print("\n\n################ FINAL SUMMARY ################")
    print(f"Classical SVM Accuracy (on {len(y_test)} samples): {classical_acc:.4f}")
    print(f"Quantum SVM Accuracy (on {len(y_test_q)} samples): {quantum_acc:.4f}")
    print("Classical Confusion Matrix:")
    print(classical_cm)
    print("Quantum Confusion Matrix:")
    print(quantum_cm)
    print("\nComparison complete. Check the 'results/' folder for plots.")

if __name__ == '__main__':
    # Ensure correct working directory if run from outside the main directory
    try:
        # Move up to the project root if currently in a submodule
        if os.path.basename(os.getcwd()) in ["quantum_model", "comparison", "utils"]:
            os.chdir("..")
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
