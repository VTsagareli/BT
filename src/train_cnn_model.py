import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# CNN Model Definition with Stronger Regularization
class CNN1DClassifier(nn.Module):
    def __init__(self):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)  # Increased Dropout Regularization
        self.fc1 = nn.Linear(32 * 25, 64)  
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.dropout(x)  # Dropout before fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to train the model with Early Stopping and Regularization
def train_model(X_train, y_train, num_epochs=100, learning_rate=0.001, weight_decay=1e-2, patience=10):
    model = CNN1DClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Increased L2 Regularization

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Validation Loss Calculation
        model.eval()
        with torch.no_grad():
            val_output = model(X_train)  # Using train set as validation for now (should use separate val set)
            val_loss = criterion(val_output, y_train)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Early Stopping Condition
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), "models/cnn_model.pth")  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    # Plot Training vs Validation Loss
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig("models/loss_curve.png")
    plt.show()

if __name__ == "__main__":
    # Load the training data
    X_train = np.load("data/processed_data/X_train.npy")
    y_train = np.load("data/processed_data/y_train.npy")

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)

    print(f"Shape of X_train: {X_train.shape}")

    train_model(X_train, y_train)

    print("Model trained and saved!")
