import numpy as np
from sklearn.model_selection import train_test_split

def split_data(features_path="data/processed_data/features.npy", 
               labels_path="data/processed_data/labels.npy", 
               output_path="data/processed_data/"):
    # Load features and labels
    X = np.load(features_path)
    y = np.load(labels_path)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the splits
    np.save(f"{output_path}/X_train.npy", X_train)
    np.save(f"{output_path}/X_test.npy", X_test)
    np.save(f"{output_path}/y_train.npy", y_train)
    np.save(f"{output_path}/y_test.npy", y_test)

    print(f"Data split completed! Files saved to {output_path}")

if __name__ == "__main__":
    split_data()
