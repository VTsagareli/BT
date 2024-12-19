import torch
import numpy as np
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
        accuracy = (predicted == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item()}")

if __name__ == "__main__":
    # Load test data
    X_test = np.load("data/processed_data/X_test.npy")
    y_test = np.load("data/processed_data/y_test.npy")

    # Convert data to PyTorch tensors
    X_test = torch.FloatTensor(X_test)  # Shape: [batch_size, 40, sequence_length]
    y_test = torch.LongTensor(y_test)

    # Evaluate the model
    evaluate_model(X_test, y_test)
