import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def split_data(features_path="data/processed_data/features.npy", 
               labels_path="data/processed_data/labels.npy", 
               output_path="data/processed_data/"):
    """
    Splits the data into training and test sets, ensures no data leakage, and prepares data for both CNN and ML models.
    """
    # Load features and labels
    print("Loading data...")
    X = np.load(features_path)
    y = np.load(labels_path)
    print(f"Original data loaded. X shape: {X.shape}, y shape: {y.shape}")

    # Remove duplicate samples to prevent data redundancy
    print("Removing duplicates...")
    X, unique_indices = np.unique(X, axis=0, return_index=True)
    y = y[unique_indices]
    print(f"After removing duplicates: X shape: {X.shape}, y shape: {y.shape}")

    # Shuffle the data to prevent order-based biases
    print("Shuffling data...")
    X, y = shuffle(X, y, random_state=42)
    print("Data shuffled.")

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Check for data leakage between train and test sets
    print("Checking for data leakage...")
    train_set = {tuple(sample) for sample in X_train.reshape(X_train.shape[0], -1)}
    test_set = {tuple(sample) for sample in X_test.reshape(X_test.shape[0], -1)}
    overlap_count = len(train_set.intersection(test_set))
    overlap_ratio = overlap_count / len(test_set)

    if overlap_ratio > 0:
        print(f"Warning: {overlap_ratio:.2%} overlap detected between training and test sets.")
    else:
        print("No data leakage detected.")

    # Save splits for CNN models
    print("Saving CNN-compatible data...")
    np.save(f"{output_path}/X_train.npy", X_train)
    np.save(f"{output_path}/X_test.npy", X_test)
    np.save(f"{output_path}/y_train.npy", y_train)
    np.save(f"{output_path}/y_test.npy", y_test)
    print("CNN-compatible data saved.")

    # Prepare and save flattened data for ML models
    print("Preparing ML-compatible data...")
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)  
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    np.save(f"{output_path}/X_train_flat.npy", X_train_flattened)
    np.save(f"{output_path}/X_test_flat.npy", X_test_flattened)
    print("ML-compatible flattened data saved.")

    print(f"Data processing completed! Files saved to {output_path}")

if __name__ == "__main__":
    split_data()
