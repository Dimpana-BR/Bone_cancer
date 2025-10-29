import numpy as np
import os
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_images_from_csv(folder_path, csv_path, size=(64, 64)):
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {csv_path}")
    print("Columns found:", df.columns.tolist())

    df.columns = [col.strip() for col in df.columns]

    data, labels = [], []

    file_col = None
    for col in df.columns:
        if 'file' in col.lower() or 'image' in col.lower() or 'name' in col.lower():
            file_col = col
            break
    if file_col is None:
        raise ValueError("Could not detect filename column. Please check your CSV.")

    label_cols = [col for col in df.columns if col != file_col]

    for i, row in df.iterrows():
        img_path = os.path.join(folder_path, str(row[file_col]))
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, size)
                data.append(img.flatten())
                for col in label_cols:
                    if row[col] == 1:
                        labels.append(col)  
                        break
            else:
                print(f"Warning: Cannot read image {img_path}")
        else:
            print(f"Warning: File does not exist {img_path}")

    return np.array(data), np.array(labels)


train_path = r'D:\MINI PROJECT\train'
valid_path = r'D:\MINI PROJECT\valid'
test_path  = r'D:\MINI PROJECT\test'

train_csv = r'D:\MINI PROJECT\train\_classes.csv'
valid_csv = r'D:\MINI PROJECT\valid\_classes.csv'
test_csv  = r'D:\MINI PROJECT\test\_classes.csv'


X_train, y_train = load_images_from_csv(train_path, train_csv)
X_val, y_val     = load_images_from_csv(valid_path, valid_csv)
X_test, y_test   = load_images_from_csv(test_path, test_csv)

print("\nData Loaded Successfully!")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)
print("\nSVM Training Complete!")


val_pred = svm.predict(X_val)
test_pred = svm.predict(X_test)

print("\nValidation Accuracy:", accuracy_score(y_val, val_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred))


cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


import joblib

# Save trained SVM and scaler
joblib.dump(svm, 'D:\MINI PROJECT\svm_model.pkl')
joblib.dump(scaler, 'D:\MINI PROJECT\scaler.pkl')
print("Model and scaler saved!")
