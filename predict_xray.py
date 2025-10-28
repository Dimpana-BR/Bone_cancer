import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

svm = joblib.load('D:\MINI PROJECT\svm_model.pkl')        # trained SVM
scaler = joblib.load('D:\MINI PROJECT\scaler.pkl')        # fitted scaler

def predict_xray(image_path, img_size=(64,64)):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image. Check the file path.")

    # Resize and flatten
    img_resized = cv2.resize(img, img_size)
    img_flatten = img_resized.flatten().reshape(1, -1)

    # Scale features
    img_scaled = scaler.transform(img_flatten)

    # Predict
    prediction = svm.predict(img_scaled)

    # image with prediction
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {prediction[0]}")
    plt.axis('off')
    plt.show()

    return prediction[0]

# to predict with new image
image_path = r'D:\MINI PROJECT\images\stock-photo-bone-cancer-in-shoulder-444387289.jpg'  # replace with your X-ray image
result = predict_xray(image_path)
print(f"Prediction: {result}")
