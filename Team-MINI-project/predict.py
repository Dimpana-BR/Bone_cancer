import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Load Trained Models
# ===========================
qsvc = joblib.load(r"quantum_model.pkl")
pca = joblib.load(r"pca.pkl")  # Load saved PCA model

print("✅ Quantum model and PCA loaded successfully!")

# ===========================
# Image Prediction Function
# ===========================
def predict_xray_quantum(image_path, img_size=(64, 64)):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("❌ Invalid image path or unreadable image!")

    # Resize and flatten
    img_resized = cv2.resize(img, img_size)
    img_flatten = img_resized.flatten().reshape(1, -1)

    # Apply saved PCA transformation
    X_pca = pca.transform(img_flatten)

    # Predict using Quantum SVM
    prediction = qsvc.predict(X_pca)[0]
    probs = qsvc.decision_function(X_pca)

    # Convert decision score → probability-like value
    prob = 1 / (1 + np.exp(-abs(probs[0]))) * 100

    # Display results
    if prediction == 'cancer' or prediction == 1:
        cancer_stage = estimate_stage(prob)
        print(f"\n🧬 Prediction: Cancer Detected")
        print(f"🔹 Probability: {prob:.2f}%")
        print(f"🔹 Estimated Stage: {cancer_stage}")
        plt.title(f"Predicted: Cancer\nProbability: {prob:.2f}%\nStage: {cancer_stage}")
    else:
        print("\n✅ Prediction: Normal Bone (No Cancer Detected)")
        plt.title("Predicted: Normal Bone\n(No Cancer Detected)")

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# ===========================
# Helper: Estimate Stage
# ===========================
def estimate_stage(prob):
    if prob < 25:
        return "Stage 1"
    elif prob < 50:
        return "Stage 2"
    elif prob < 75:
        return "Stage 3"
    else:
        return "Stage 4"

# ===========================
# Example Usage
# ===========================
if __name__ == "__main__":
    image_path = r"images\xray-image-knee-joint-600nw-2488771867.webp"  # Replace with your image
    predict_xray_quantum(image_path)

