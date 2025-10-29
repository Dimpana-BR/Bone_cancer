import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix

# Try to import Qiskit components. If unavailable or incompatible, fall back
# to a classical SVC implementation so the pipeline can still run.
QISKIT_AVAILABLE = True
try:
    # from qiskit_algorithms.utils import algorithm_globals  # Deprecated in newer Qiskit
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit.primitives import Sampler
    from qiskit.circuit.library import ZFeatureMap
except Exception as _ex:
    print("Qiskit imports failed or are incompatible in this environment. Falling back to classical SVC for the quantum stage.")
    QISKIT_AVAILABLE = False
    from sklearn.svm import SVC

# Set seed for reproducibility using standard libraries, as 'algorithm_globals' is deprecated/removed.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def train_and_predict_qsvm(X_train_q, y_train_q, X_test_q, y_test_q):
    """
    Builds, trains, and tests a Quantum Kernel Support Vector Classifier (QSVC) 
    using Qiskit's ZFeatureMap.
    
    Args:
        X_train_q (np.ndarray): Quantum training features (simulated, often same as test for kernel methods).
        y_train_q (np.ndarray): Quantum training labels.
        X_test_q (np.ndarray): Quantum testing features.
        y_test_q (np.ndarray): Quantum testing labels (true labels).
        
    Returns:
        tuple: (accuracy: float, y_pred: np.ndarray, cm: np.ndarray)
    """
    print("\n--- Running Quantum SVM (QSVC) ---")

    try:
        if QISKIT_AVAILABLE:
            num_features = X_train_q.shape[1]
            # 1. Define Feature Map (Quantum Kernel)
            feature_map = ZFeatureMap(feature_dimension=num_features, reps=3, entanglement='linear')

            # 2. Define Quantum Kernel
            quantum_kernel = QuantumKernel(feature_map=feature_map, sampler=Sampler())

            # 3. Define QSVC Classifier
            qsvc = QSVC(quantum_kernel=quantum_kernel)

            # 4. Train the QSVC
            print(f"Training QSVC with {X_train_q.shape[0]} samples and {num_features} features...")
            qsvc.fit(X_train_q, y_train_q)

            # 5. Predict on the test data
            y_pred_q = qsvc.predict(X_test_q)

            # 6. Calculate metrics
            accuracy_q = accuracy_score(y_test_q, y_pred_q)
            cm_q = confusion_matrix(y_test_q, y_pred_q)

            print("QSVC training and prediction complete.")
            return accuracy_q, y_pred_q, cm_q
        else:
            # Fallback: use a classical SVC on the reduced feature space to simulate
            # the 'quantum' stage when Qiskit isn't available.
            print("Running fallback classical SVC for the quantum stage...")
            clf = SVC(kernel='rbf', C=10.0, probability=False, random_state=RANDOM_SEED)
            clf.fit(X_train_q, y_train_q)
            y_pred_q = clf.predict(X_test_q)
            accuracy_q = accuracy_score(y_test_q, y_pred_q)
            cm_q = confusion_matrix(y_test_q, y_pred_q)
            print("Fallback SVC training and prediction complete.")
            return accuracy_q, y_pred_q, cm_q

    except Exception as e:
        print(f"An error occurred during Quantum SVM execution: {e}")
        return None, None, None

# Helper function to split the small quantum dataset (for QSVC demonstration)
def split_quantum_data(X_q, y_q):
    """Splits the small quantum subset into train and test parts."""
    # Since the quantum data is already small (e.g., 10 samples), we'll split it 50/50
    split_idx = len(X_q) // 2
    X_train = X_q[:split_idx]
    y_train = y_q[:split_idx]
    X_test = X_q[split_idx:]
    y_test = y_q[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        # Fallback if too few samples
        return X_q, y_q, X_q, y_q 
        
    return X_train, y_train, X_test, y_test