import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from train_model import CNN1DClassifier

def evaluate_model(X_test, y_test):
    # Load the trained model
    model = CNN1DClassifier()
    model.load_state_dict(torch.load("models/cnn_model.pth"))
    model.eval()

    # Evaluate the model
    with torch.no_grad():
        output = model(X_test)
        _, predicted = torch.max(output, 1)

        # Calculate accuracy
        accuracy = (predicted == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item()}")

        # Generate detailed metrics
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Broken"]))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot the confusion matrix
        plot_confusion_matrix(cm, classes=["Normal", "Broken"])

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
    # Load test data
    X_test = np.load("data/processed_data/X_test.npy")
    y_test = np.load("data/processed_data/y_test.npy")

    # Convert data to PyTorch tensors
    X_test = torch.FloatTensor(X_test)  # Shape: [batch_size, 40, sequence_length]
    y_test = torch.LongTensor(y_test)

    # Evaluate the model
    evaluate_model(X_test, y_test)
