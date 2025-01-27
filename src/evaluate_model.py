import os
import torch
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from train_cnn_model import CNN1DClassifier

# Paths to the data and models
DATA_PATH = "data/processed_data"
MODEL_PATH = "models"

def evaluate_cnn_model(X_test, y_test):
    """
    Evaluate the trained CNN model using the correct test data.
    """
    try:
        model_path = os.path.join(MODEL_PATH, "cnn_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CNN model file '{model_path}' not found.")

        model = CNN1DClassifier()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        with torch.no_grad():
            output = model(X_test)
            _, predicted = torch.max(output, 1)

            accuracy = (predicted == y_test).float().mean()
            print(f"\nCNN Model Accuracy: {accuracy.item():.4f}")

            y_true = y_test.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            print("\nCNN Classification Report:")
            print(classification_report(y_true, y_pred, target_names=["Normal", "Broken"]))

            cm = confusion_matrix(y_true, y_pred)
            print("\nCNN Confusion Matrix:")
            print(cm)
            plot_confusion_matrix(cm, classes=["Normal", "Broken"], title="CNN Confusion Matrix")

    except Exception as e:
        print(f"Error evaluating CNN model: {e}")

def evaluate_ml_models(X_test_flat, y_test):
    """
    Evaluate classic ML models using the flattened test data.
    """
    model_names = ["LogisticRegression", "SVM", "DecisionTree", "RandomForest", "GradientBoosting"]

    print(f"Evaluating ML models with input shape: {X_test_flat.shape}")

    for model_name in model_names:
        model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: Model file not found ({model_path})")
            continue

        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_test_flat)

            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n{model_name} Accuracy: {accuracy:.4f}")

            print(f"\n{model_name} Classification Report:")
            print(classification_report(y_test, y_pred, target_names=["Normal", "Broken"]))

            cm = confusion_matrix(y_test, y_pred)
            print(f"\n{model_name} Confusion Matrix:")
            print(cm)
            plot_confusion_matrix(cm, classes=["Normal", "Broken"], title=f"{model_name} Confusion Matrix")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        print("Loading test data...")

        # Load CNN test data
        X_test_cnn = np.load(os.path.join(DATA_PATH, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

        # Convert CNN test data to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test_cnn)  # Shape: [batch_size, 40, sequence_length]
        y_test_tensor = torch.LongTensor(y_test)

        # Load flattened test data for ML models
        X_test_flat = np.load(os.path.join(DATA_PATH, "X_test_flat.npy"))

        # Evaluate the models
        print("\nEvaluating CNN Model...")
        evaluate_cnn_model(X_test_tensor, y_test_tensor)

        print("\nEvaluating Classic ML Models...")
        evaluate_ml_models(X_test_flat, y_test)

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
