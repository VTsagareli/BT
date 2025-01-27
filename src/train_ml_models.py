import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Define classic ML models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "SVM": SVC(kernel="rbf", probability=True),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    #"GradientBoosting": GradientBoostingClassifier(n_estimators=50)
}

# Train and save each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)  # Fit model to training data

    # Save the trained model
    model_filepath = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    joblib.dump(model, model_filepath)

    print(f"{model_name} trained and saved at {model_filepath}")

print("\nAll ML models have been successfully trained and saved!")
