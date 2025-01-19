# BT
Bachelor Thesis Project - AI Sound Anomaly Detection Model
Steps To Run the model:

1. Run cleaning_and_sampling.py:
    
    This script processes raw audio data by:
    
    Cleaning the audio:
    Loads audio files from data/broken_audio and data/normal_audio.
    Skips hidden and corrupted files.
    Saves cleaned files to data/broken_audio_samples and data/normal_audio_samples.
    
    Splitting the audio:
    Broken audio files are split into 3-second chunks.
    Normal audio files are cleaned but not split.
    
    Clearing old files:
    Deletes existing processed files in output folders to ensure fresh data.

2. Run preprocess_audio.py:
    
    This script processes cleaned audio data by:

    Augmenting the audio:

    Applies various augmentations to broken audio samples, including:
    Adding noise
    Reversing the audio
    Dynamic range compression
    Time stretching
    Equalization (EQ)
    Clipping the audio
    Scaling the amplitude
    Inserting silence
    Frequency masking

    Extracting features:

    Extracts MFCC (Mel-Frequency Cepstral Coefficients) features from both normal and broken audio samples.
    Ensures consistency by padding or truncating features to a fixed size.
    
    Saving processed data:

    Saves extracted features and labels as .npy files in data/processed_data for model training.

3. Run split_data.py:
    
    This script prepares the processed audio data for training by:

    Loading processed features and labels:

    Reads the features.npy and labels.npy files generated in the preprocessing step.
    
    Splitting the data:

    Divides the dataset into training (80%) and testing (20%) subsets using train_test_split from sklearn.
    Ensures reproducibility with a fixed random state.
    
    Saving the split data:

    Saves the training and test splits to data/processed_data/ as:
        X_train.npy (training features)
        X_test.npy (testing features)
        y_train.npy (training labels)
        y_test.npy (testing labels)

4. Run train_model.py
    
    This script trains a 1D Convolutional Neural Network (CNN) to classify audio features.

    Steps performed in this script:

    Load processed training data:

    Reads X_train.npy (features) and y_train.npy (labels) from data/processed_data/.
    Converts them into PyTorch tensors for training.
    
    Define the CNN model architecture:

    A 1D CNN is used with the following layers:
    Conv1D layers for feature extraction.
    MaxPooling layers for downsampling.
    Fully connected (Linear) layers for classification.
    
    Train the model:

    Optimizer: Adam with a learning rate of 0.001.
    Loss function: Cross-Entropy Loss.
    Training for 50 epochs.
    Loss is printed every 10 epochs for monitoring.
    
    Save the trained model:

    The trained model weights are saved in models/cnn_model.pth using torch.save().

5. Run evaluate_model.py
    This script evaluates the trained 1D CNN model on the test dataset and provides detailed performance metrics.

    Steps performed in this script:

    Load the trained model:

    Loads the saved model weights from models/cnn_model.pth.
    Sets the model to evaluation mode (model.eval()).
    
    Load test data:

    Reads X_test.npy (features) and y_test.npy (labels) from data/processed_data/.
    Converts them into PyTorch tensors for evaluation.
    
    Evaluate the model:

    Performs inference on the test dataset without updating model weights.
    Computes the test accuracy.
    
    Generate performance metrics:

    Classification report:
    
    Includes precision, recall, and F1-score for "Normal" and "Broken" audio samples.
    
    Confusion matrix:
    
    Displays the number of correct and incorrect predictions in a matrix format.
    
    Visualization:
    
    Plots the confusion matrix for better insights into model performance.

Optional Step:
    
    clean_sample_folders.py:

    cleans folders augmetned_broken_audio_samples, broken_audio_samples and normal_audio_samples for convenience.