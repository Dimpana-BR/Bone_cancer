import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, model_name, filepath):
    """
    Generates and saves a confusion matrix plot.
    
    Args:
        cm (np.ndarray): The confusion matrix array.
        model_name (str): Name of the model ('Classical' or 'Quantum').
        filepath (str): Full path to save the .png file.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Cancerous', 'Cancerous'], 
                yticklabels=['Non-Cancerous', 'Cancerous'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved {model_name.lower()}_confusion_matrix.png to /results/")


def plot_accuracy_comparison(classical_acc, quantum_acc, filepath):
    """
    Generates and saves a bar chart comparing classical and quantum accuracy.
    
    Args:
        classical_acc (float): Accuracy of the classical model.
        quantum_acc (float): Accuracy of the quantum model.
        filepath (str): Full path to save the .png file.
    """
    models = ['Classical SVM', 'Quantum SVM']
    accuracies = [classical_acc, quantum_acc]

    plt.figure(figsize=(7, 6))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    
    # Add accuracy values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                 f'{yval:.4f}', ha='center', va='bottom', fontsize=10)
                 
    plt.ylim(0, 1.0)
    plt.title('Accuracy Comparison: Classical vs Quantum Bone Cancer Detection')
    plt.ylabel('Accuracy Score')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved accuracy_comparison.png to /results/")


def compare_and_plot(classical_acc, classical_cm, quantum_acc, quantum_cm):
    """
    Orchestrates the plotting of confusion matrices and accuracy comparison.
    """
    
    # 1. Plot Classical Confusion Matrix
    plot_confusion_matrix(
        classical_cm, 
        "Classical", 
        os.path.join("results", "confusion_matrix_classical.png")
    )
    
    # 2. Plot Quantum Confusion Matrix
    plot_confusion_matrix(
        quantum_cm, 
        "Quantum", 
        os.path.join("results", "confusion_matrix_quantum.png")
    )
    
    # 3. Plot Accuracy Comparison
    plot_accuracy_comparison(
        classical_acc, 
        quantum_acc, 
        os.path.join("results", "accuracy_comparison.png")
    )
