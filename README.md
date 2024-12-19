# BT
Bachelor Thesis Project - AI Sound Anomaly Detection Model.

Steps To Run the model:

1. Run preprocess_audio.py:
    Splits the audio into smaller parts.
    Cleans the audio.
    Extracts numerical features (MFCCs).
    Saves the processed data for training a model.

2. Run split_data.py:
    takes processed audio features and labels, splits them into training and testing datasets, and saves these splits for later use in training and evaluating a machine learning model.

3. Run train_model.py:
    Defines a 1D CNN to classify audio features.
    Trains the model using the training dataset.
    Saves the trained model for future use.

4. Run evaluate_model.py:
    Loads a trained CNN model.
    Evaluates the model using the test dataset.
    Prints the test accuracy, giving you an idea of how well the model generalizes to new data.
