import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# CNN Model Definition
class CNN1DClassifier(nn.Module):
    def __init__(self):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Updated in_features based on calculated flattened size
        self.fc1 = nn.Linear(32 * 25, 64)  # 32 channels * 25 time steps after pooling
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(X_train, y_train, num_epochs=50, learning_rate=0.001):
    model = CNN1DClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model

if __name__ == "__main__":
    # Load the training data
    X_train = np.load("data/processed_data/X_train.npy")
    y_train = np.load("data/processed_data/y_train.npy")

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)  # Shape: [batch_size, 40, sequence_length]
    y_train = torch.LongTensor(y_train)

    # Debugging shape of input to confirm correctness
    print(f"Shape of X_train: {X_train.shape}")  # Should be [batch_size, 40, 100] or similar

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model's state_dict
    torch.save(model.state_dict(), "models/cnn_model.pth")
    print("Model trained and saved!")
