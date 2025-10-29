🦴 Quantum + Classical Bone Cancer Detection Project

This project extends a classical Support Vector Machine (SVM) bone cancer detection pipeline with a Quantum Kernel SVM (QSVM) using Qiskit for comparative analysis.

Goal: Compare the performance (Accuracy and Confusion Matrix) of a pre-trained classical SVM model against a resource-constrained Quantum SVM (QSVM) on a subset of the test data.

🛠️ Setup and Installation

Clone/Create Project Structure: Ensure your local environment matches the directory structure:

MINI PROJECT/
├── ... (folders)
├── main.py
└── ... (files)


Install Dependencies: All necessary libraries can be installed via the provided requirements.txt.

pip install -r requirements.txt


Note: The script generates dummy data and models (svm_model.pkl, scaler.pkl) on first run to ensure immediate execution.

▶️ How to Run

Execute the main script from the root of the project directory:

python main.py


📂 Output Files Generated

The main.py script automatically generates the following artifacts:

File Path

Description

svm_model.pkl

Dummy pre-trained classical SVM model (simulated).

scaler.pkl

Dummy pre-fitted StandardScaler object (simulated).

test/X_test_scaled.npy

The shared, scaled test features used by both models.

results/confusion_matrix_classical.png

Visualization of the Classical SVM's performance.

results/confusion_matrix_quantum.png

Visualization of the Quantum SVM's performance.

results/accuracy_comparison.png

Bar chart comparing the accuracy scores of both models.

🧠 Implementation Details

Classical Model (predict_xray.py)

The script loads the existing svm_model.pkl and scaler.pkl.

It performs predictions on the full test set (X_test_scaled.npy).

No retraining of the classical model is performed.

Quantum Model (quantum_model/quantum_svm_qiskit.py)

Quantum Preprocessing: Due to the high computational cost of simulating quantum circuits, quantum_model/quantum_preprocess.py reduces the test data to a small subset (e.g., 10 samples, 2 features).

Architecture: It uses a ZFeatureMap with a QuantumKernel and the QSVC algorithm from qiskit_machine_learning.

Backend: All quantum computations are run on a classical simulator (Sampler), requiring no external cloud account.

📊 Interpreting the Results

Accuracy Comparison (accuracy_comparison.png): This chart provides a direct visual comparison of the two models' predictive power. The Quantum SVM is trained on significantly less and highly reduced data, so a direct, feature-for-feature comparison is not fair. The goal is to see if QML can achieve competitive results with highly simplified input/resource constraints.

Confusion Matrices: These show how often each model correctly identifies Non-Cancerous (True Negatives) and Cancerous (True Positives) cases, which is crucial for medical applications. Look for the differences in False Positives (Type I error) and False Negatives (Type II error).