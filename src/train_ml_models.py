import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

# Define paths for data and model storage
DATA_PATH = "data/processed_data"
MODEL_PATH = "models"

# Load training data
print("Loading training data...")
X_train = np.load(os.path.join(DATA_PATH, "X_train_flat.npy"))  # Use pre-flattened data
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))

print(f"Training data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Ensure model directory exists
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Define classic ML models with stronger regularization
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs', penalty="l2", C=0.1),  # Increased L2 regularization
    "SVM": SVC(kernel="rbf", probability=True, C=0.5),  # Increased C for better regularization
    "DecisionTree": DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2),  # Pruned Decision Tree
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2)
}

# Train and save each model with 10-fold cross-validation
for model_name, model in models.items():
    print(f"\nTraining {model_name} with 10-fold cross-validation...")

    scores = cross_val_score(model, X_train, y_train, cv=10)  # 10-Fold CV
    print(f"{model_name} Cross-Validation Accuracy: {scores.mean():.4f}")

    model.fit(X_train, y_train)  # Train final model

    # Save the trained model
    model_filepath = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    joblib.dump(model, model_filepath)

    print(f"{model_name} trained and saved at {model_filepath}")

print("\nAll ML models successfully trained with stronger regularization and cross-validation!")
