from flask import Flask, render_template, request, jsonify
import os, cv2, joblib, json
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load trained models
qsvc = joblib.load(r"quantum_model.pkl")
pca = joblib.load(r"pca.pkl")
scaler = joblib.load(r"scaler.pkl")

def estimate_stage(prob):
    if prob < 25:
        return "Stage 1"
    elif prob < 50:
        return "Stage 2"
    elif prob < 75:
        return "Stage 3"
    else:
        return "Stage 4"

def predict_bone_cancer(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"status": "Error: Unable to read image file", "prob": "0.00%", "stage": "N/A"}
        img = cv2.resize(img, (64, 64))
        img_flat = img.flatten().reshape(1, -1)
        # Apply scaling first
        img_scaled = scaler.transform(img_flat)
        # Then apply PCA
        X_pca = pca.transform(img_scaled)
        pred = qsvc.predict(X_pca)[0]
        prob = 1 / (1 + np.exp(-abs(qsvc.decision_function(X_pca)[0]))) * 100

        if pred == 'cancer' or pred == 1:
            stage = estimate_stage(prob)
            return {"status": "Cancer Detected", "prob": f"{prob:.2f}%", "stage": stage}
        else:
            return {"status": "Normal Bone (No Cancer Detected)", "prob": f"{prob:.2f}%", "stage": "Normal"}
    except Exception as e:
        return {"status": f"Error during prediction: {str(e)}", "prob": "0.00%", "stage": "N/A"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    result = predict_bone_cancer(filepath)
    result["image"] = f"/static/uploads/{file.filename}"
    return jsonify(result)

@app.route("/comparison")
def comparison():
    try:
        with open(r"model_metrics.json", 'r') as f:
            metrics = json.load(f)

        classical_val_acc = f"{metrics['classical']['validation_accuracy']:.4f}"
        classical_test_acc = f"{metrics['classical']['test_accuracy']:.4f}"

        # Format classical classification report
        classical_report = metrics['classical']['classification_report']
        classical_report_str = f"""              precision    recall  f1-score   support

      cancer       {classical_report['cancer']['precision']:.2f}      {classical_report['cancer']['recall']:.2f}      {classical_report['cancer']['f1-score']:.2f}       {classical_report['cancer']['support']}
      normal       {classical_report['normal']['precision']:.2f}      {classical_report['normal']['recall']:.2f}      {classical_report['normal']['f1-score']:.2f}       {classical_report['normal']['support']}

    accuracy                           {classical_report['accuracy']:.2f}       {classical_report['macro avg']['support']}
   macro avg       {classical_report['macro avg']['precision']:.2f}      {classical_report['macro avg']['recall']:.2f}      {classical_report['macro avg']['f1-score']:.2f}       {classical_report['macro avg']['support']}
weighted avg       {classical_report['weighted avg']['precision']:.2f}      {classical_report['weighted avg']['recall']:.2f}      {classical_report['weighted avg']['f1-score']:.2f}       {classical_report['weighted avg']['support']}"""

        quantum_accuracy = f"{metrics['quantum']['subset_accuracy']:.4f}"

        # Format quantum classification report
        quantum_report = metrics['quantum']['classification_report']
        quantum_report_str = f"""              precision    recall  f1-score   support

      cancer       {quantum_report['cancer']['precision']:.2f}      {quantum_report['cancer']['recall']:.2f}      {quantum_report['cancer']['f1-score']:.2f}       {quantum_report['cancer']['support']}
      normal       {quantum_report['normal']['precision']:.2f}      {quantum_report['normal']['recall']:.2f}      {quantum_report['normal']['f1-score']:.2f}       {quantum_report['normal']['support']}

    accuracy                           {quantum_report['accuracy']:.2f}       {quantum_report['macro avg']['support']}
   macro avg       {quantum_report['macro avg']['precision']:.2f}      {quantum_report['macro avg']['recall']:.2f}      {quantum_report['macro avg']['f1-score']:.2f}       {quantum_report['macro avg']['support']}
weighted avg       {quantum_report['weighted avg']['precision']:.2f}      {quantum_report['weighted avg']['recall']:.2f}      {quantum_report['weighted avg']['f1-score']:.2f}       {quantum_report['weighted avg']['support']}"""

        return render_template("comparison.html",
                             classical_val_acc=classical_val_acc,
                             classical_test_acc=classical_test_acc,
                             classical_report=classical_report_str,
                             quantum_accuracy=quantum_accuracy,
                             quantum_report=quantum_report_str)

    except FileNotFoundError:
        return "Model metrics not found. Please run BNN.py first to generate the metrics."
    except Exception as e:
        return f"Error loading metrics: {str(e)}"

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
