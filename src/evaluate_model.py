import os
import torch
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from train_cnn_model import CNN1DClassifier

DATA_PATH = "data/processed_data"
MODEL_PATH = "models"

def evaluate_cnn_model(X_train, y_train, X_test, y_test):
    print("\nEvaluating CNN Model...")
    
    model_path = os.path.join(MODEL_PATH, "cnn_model.pth")
    if not os.path.exists(model_path):
        print("CNN model file not found.")
        return

    model = CNN1DClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        train_output = model(X_train)
        test_output = model(X_test)
        _, train_predicted = torch.max(train_output, 1)
        _, test_predicted = torch.max(test_output, 1)

        train_accuracy = (train_predicted == y_train).float().mean().item()
        test_accuracy = (test_predicted == y_test).float().mean().item()

        print(f"CNN Training Accuracy: {train_accuracy:.4f}")
        print(f"CNN Test Accuracy: {test_accuracy:.4f}")

        print("\nCNN Classification Report:")
        print(classification_report(y_test.cpu().numpy(), test_predicted.cpu().numpy()))

        cm = confusion_matrix(y_test.cpu().numpy(), test_predicted.cpu().numpy())
        print("\nCNN Confusion Matrix:\n", cm)

def evaluate_ml_models(X_train_flat, y_train, X_test_flat, y_test):
    print("\nEvaluating Classic ML Models...")

    for model_name in ["LogisticRegression", "SVM", "DecisionTree", "RandomForest"]:
        model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: Model not found.")
            continue

        model = joblib.load(model_path)

        train_pred = model.predict(X_train_flat)
        test_pred = model.predict(X_test_flat)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"\n{model_name} Training Accuracy: {train_acc:.4f}")
        print(f"{model_name} Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    X_train = torch.FloatTensor(np.load(os.path.join(DATA_PATH, "X_train.npy")))
    y_train = torch.LongTensor(np.load(os.path.join(DATA_PATH, "y_train.npy")))
    X_test = torch.FloatTensor(np.load(os.path.join(DATA_PATH, "X_test.npy")))
    y_test = torch.LongTensor(np.load(os.path.join(DATA_PATH, "y_test.npy")))

    evaluate_cnn_model(X_train, y_train, X_test, y_test)

    X_train_flat = np.load(os.path.join(DATA_PATH, "X_train_flat.npy"))
    X_test_flat = np.load(os.path.join(DATA_PATH, "X_test_flat.npy"))

    evaluate_ml_models(X_train_flat, y_train, X_test_flat, y_test)
