import numpy as np
import pandas as pd
import os, cv2, joblib, json
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Load Images from CSV
# =======================
def load_images_from_csv(folder_path, csv_path, size=(64, 64)):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    data, labels = [], []
    file_col = [col for col in df.columns if 'file' in col.lower() or 'image' in col.lower()][0]
    label_cols = [col for col in df.columns if col != file_col]

    for _, row in df.iterrows():
        img_path = os.path.join(folder_path, str(row[file_col]))
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, size)
            data.append(img.flatten())
            labels.append('cancer' if row[label_cols[0]] == 1 else 'normal')

    return np.array(data), np.array(labels)


# =======================
# Paths
# =======================
train_path = r'D:\new_MINI\Team-MINI-project\dataset\train'
valid_path = r'D:\new_MINI\Team-MINI-project\dataset\valid'
test_path  = r'D:\new_MINI\Team-MINI-project\dataset\test'

train_csv = r'D:\new_MINI\Team-MINI-project\dataset\train\_classes.csv'
valid_csv = r'D:\new_MINI\Team-MINI-project\dataset\valid\_classes.csv'
test_csv  = r'D:\new_MINI\Team-MINI-project\dataset\test\_classes.csv'

# =======================
# Load Data
# =======================
X_train, y_train = load_images_from_csv(train_path, train_csv)
X_val, y_val = load_images_from_csv(valid_path, valid_csv)
X_test, y_test = load_images_from_csv(test_path, test_csv)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# =======================
# Scaling (with memory optimization)
# =======================
print("Applying StandardScaler...")
scaler = StandardScaler()
# Process in batches to avoid memory issues
batch_size = 1000
X_train_scaled = []
for i in range(0, len(X_train), batch_size):
    batch = X_train[i:i+batch_size].astype(np.float32)
    if i == 0:
        X_train_scaled.extend(scaler.fit_transform(batch))
    else:
        X_train_scaled.extend(scaler.transform(batch))
X_train = np.array(X_train_scaled, dtype=np.float32)

X_val_scaled = []
for i in range(0, len(X_val), batch_size):
    batch = X_val[i:i+batch_size].astype(np.float32)
    X_val_scaled.extend(scaler.transform(batch))
X_val = np.array(X_val_scaled, dtype=np.float32)

X_test_scaled = []
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size].astype(np.float32)
    X_test_scaled.extend(scaler.transform(batch))
X_test = np.array(X_test_scaled, dtype=np.float32)

# =======================
# Classical SVM
# =======================
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm.fit(X_train, y_train)

val_pred = svm.predict(X_val)
test_pred = svm.predict(X_test)

print("\nValidation Accuracy:", accuracy_score(y_val, val_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred))

# =======================
# Confusion Matrix (optional - comment out to save memory)
# =======================
# cm = confusion_matrix(y_test, test_pred)
# plt.figure(figsize=(6,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix - SVM')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# =======================
# Save Classical Model
# =======================
os.makedirs("models", exist_ok=True)
joblib.dump(svm, "D:\\new_MINI\\Team-MINI-project\\svm_model.pkl")
joblib.dump(scaler, "D:\\new_MINI\\Team-MINI-project\\scaler.pkl")
print("\n✅ SVM model & scaler saved successfully!")


# =======================
# Quantum Model (with PCA)
# =======================
from sklearn.decomposition import PCA
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Randomly select small subset (quantum is resource heavy)
idx = np.random.choice(len(X_test), 10, replace=False)
X_small, y_small = X_test[idx], y_test[idx]

# Reduce to 2D for 2-qubit feature map
pca = PCA(n_components=2)
X_small_pca = pca.fit_transform(X_small)

# Quantum setup
feature_map = ZFeatureMap(feature_dimension=2, reps=2)
sampler = StatevectorSampler()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Train quantum SVM
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_small_pca, y_small)

# Evaluate quantum SVM on the same small subset
q_pred = qsvc.predict(X_small_pca)
print("\nQuantum SVM Accuracy on subset:", accuracy_score(y_small, q_pred))
print("Quantum Classification Report:\n", classification_report(y_small, q_pred))

# Confusion Matrix for Quantum
q_cm = confusion_matrix(y_small, q_pred)
plt.figure(figsize=(6,6))
sns.heatmap(q_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Quantum SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save metrics for comparison page
metrics = {
    'classical': {
        'validation_accuracy': float(accuracy_score(y_val, val_pred)),
        'test_accuracy': float(accuracy_score(y_test, test_pred)),
        'classification_report': classification_report(y_test, test_pred, output_dict=True)
    },
    'quantum': {
        'subset_accuracy': float(accuracy_score(y_small, q_pred)),
        'classification_report': classification_report(y_small, q_pred, output_dict=True)
    }
}

with open(r"D:\new_MINI\Team-MINI-project\model_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=4)

# Save model
joblib.dump(qsvc,r"D:\new_MINI\Team-MINI-project\quantum_model.pkl")
joblib.dump(pca,r"D:\new_MINI\Team-MINI-project\pca.pkl")

print("✅ Quantum SVM and PCA models trained and saved successfully!")
print("✅ Model metrics saved to model_metrics.json")
