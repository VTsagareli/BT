import os
import torch
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from train_cnn_model import CNN1DClassifier
import librosa

# Paths
DATA_PATH = "data/processed_data"
MODEL_PATH = "models"

# Unseen data paths
UNSEEN_NORMAL_PATH = "data/unseen_normal_audio_samples"
UNSEEN_BROKEN_PATH = "data/unseen_broken_audio_samples"
UNSEEN_AUGMENTED_BROKEN_PATH = "data/unseen_augmented_broken_audio_samples"

# Parameters
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_TIME_STEPS = 100

# Feature Extraction
def extract_mfcc(audio_file, n_mfcc=N_MFCC, max_time_steps=MAX_TIME_STEPS):
    """Extracts MFCC features from an audio file."""
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to ensure consistent shape
    if mfcc.shape[1] < max_time_steps:
        pad_width = max_time_steps - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_time_steps]

    return mfcc

def load_unseen_data():
    """Loads and extracts MFCC features from unseen data."""
    X_unseen, y_unseen = [], []

    # Process unseen normal samples (Label: 0)
    print("Loading unseen normal audio samples...")
    normal_files = [os.path.join(UNSEEN_NORMAL_PATH, f) for f in os.listdir(UNSEEN_NORMAL_PATH) if f.endswith(".wav")]
    for file in normal_files:
        X_unseen.append(extract_mfcc(file))
        y_unseen.append(0)

    # Process unseen broken samples (Label: 1)
    print("Loading unseen broken audio samples...")
    broken_files = [os.path.join(UNSEEN_BROKEN_PATH, f) for f in os.listdir(UNSEEN_BROKEN_PATH) if f.endswith(".wav")]
    for file in broken_files:
        X_unseen.append(extract_mfcc(file))
        y_unseen.append(1)

    # Process unseen augmented broken samples (Label: 1)
    print("Loading unseen augmented broken audio samples...")
    augmented_files = [os.path.join(UNSEEN_AUGMENTED_BROKEN_PATH, f) for f in os.listdir(UNSEEN_AUGMENTED_BROKEN_PATH) if f.endswith(".wav")]
    for file in augmented_files:
        X_unseen.append(extract_mfcc(file))
        y_unseen.append(1)

    # Convert to NumPy arrays
    X_unseen = np.array(X_unseen)
    y_unseen = np.array(y_unseen)

    return X_unseen, y_unseen

def evaluate_cnn_model(X_test, y_test):
    """Evaluates the CNN model on unseen data."""
    try:
        model_path = os.path.join(MODEL_PATH, "cnn_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CNN model file '{model_path}' not found.")

        model = CNN1DClassifier()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        with torch.no_grad():
            output = model(X_test_tensor)
            _, predicted = torch.max(output, 1)

            accuracy = (predicted == y_test_tensor).float().mean()
            print(f"\nCNN Model Accuracy on Unseen Data: {accuracy.item():.4f}")

            y_true = y_test_tensor.cpu().numpy()
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
    """Evaluates ML models on unseen data."""
    model_names = ["LogisticRegression", "SVM", "DecisionTree", "RandomForest"]

    print(f"\nEvaluating ML models with input shape: {X_test_flat.shape}")

    for model_name in model_names:
        model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: Model file not found ({model_path})")
            continue

        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_test_flat)

            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n{model_name} Accuracy on Unseen Data: {accuracy:.4f}")

            print(f"\n{model_name} Classification Report:")
            print(classification_report(y_test, y_pred, target_names=["Normal", "Broken"]))

            cm = confusion_matrix(y_test, y_pred)
            print(f"\n{model_name} Confusion Matrix:")
            print(cm)
            plot_confusion_matrix(cm, classes=["Normal", "Broken"], title=f"{model_name} Confusion Matrix")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    """Plots the confusion matrix."""
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
        print("\nLoading unseen test data...")
        X_unseen, y_unseen = load_unseen_data()

        # Convert CNN test data to PyTorch tensors
        print("\nEvaluating CNN Model on Unseen Data...")
        evaluate_cnn_model(X_unseen, y_unseen)

        # Flatten unseen test data for ML models
        X_unseen_flat = X_unseen.reshape(X_unseen.shape[0], -1)

        print("\nEvaluating Classic ML Models on Unseen Data...")
        evaluate_ml_models(X_unseen_flat, y_unseen)

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
