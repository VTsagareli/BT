# BT
Bachelor Thesis Project - AI Sound Anomaly Detection Model.

Before running the model, one should download broken and normal audio files and place them in the folders, broken_audio and normal_audio respectively.

For normal audio, go on to this link and doenload the normal audio dataset. After that, one should copy/paste about 1500 normal audio samples to the normal_audio folder. 

Link to the normal audio dataset: https://www.kaggle.com/datasets/omkarmb/dataset-open-access-vehicle-interior-sound-dataset

The broken audio files are stored on the github repo, under releases.

Then one can proceed with running the model.

I wrote a script "run_pipeline.py" that does all of the steps below:


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

4. Run train_cnn_model.py
    
    This script trains a 1D Convolutional Neural Network (CNN) to classify audio features extracted from the processed data.

    Steps performed in this script:

    Load processed training data:

    Loads X_train.npy and y_train.npy from data/processed_data/.
    Converts the data into PyTorch tensors for training.
    The feature tensors are shaped as [batch_size, 40, sequence_length].
    Define the CNN model architecture:

    The model consists of:
    Two 1D convolutional layers for feature extraction.
    MaxPooling layers for downsampling.
    Fully connected layers for classification.
    Train the model:

    Uses Adam optimizer with a learning rate of 0.001.
    Cross-entropy loss function is applied.
    Trains for 50 epochs with loss displayed every 10 epochs.
    Save the trained model:

    Saves the trained model weights to models/cnn_model.pth.


5. Run train_ml_models.py:

    This script trains various classic machine learning models to classify audio features.

    Steps performed in this script:

    Load processed training data:

    Reads X_train_flat.npy and y_train.npy from data/processed_data/.
    The features are reshaped to [batch_size, feature_vector_size] for ML models.
    Train classic ML models:

    Trains the following models:
    Logistic Regression
    Support Vector Machine (SVM)
    Decision Tree
    Random Forest
    Save the trained models:

    Each trained model is saved in the models/ directory as .pkl files for later evaluation.

6. Run evaluate_model.py
    
    This script evaluates the performance of both the trained CNN model and the classic ML models on the test dataset.

    Steps performed in this script:

    Load test data:

    Reads X_test.npy and y_test.npy from data/processed_data/ for CNN evaluation.
    Reads X_test_flat.npy for ML model evaluation.
    Converts the CNN test features to PyTorch tensors.
    Evaluate the CNN model:

    Loads the trained CNN model from models/cnn_model.pth.
    Performs inference on the test dataset.
    Computes the test accuracy.
    Generates a classification report with precision, recall, and F1-score.
    Displays a confusion matrix for visual analysis.
    Evaluate classic ML models:

    Loads trained models from the models/ directory.
    Evaluates Logistic Regression, SVM, Decision Tree, and Random Forest classifiers.
    Generates accuracy scores, classification reports, and confusion matrices for each model.
    Handle potential errors:

    Checks for missing model files and handles exceptions gracefully.
    Provides detailed output in case of any issues during evaluation.

Optional Steps:
    
    count_sample.py:

    Counts the amount of samples at hand. Useful for knowing how many normal and broken audio samples there are after running the augmentaitons.
    
    clean_sample_folders.py:

    cleans folders augmetned_broken_audio_samples, broken_audio_samples and normal_audio_samples for convenience. It will also clean normal_audio and broken_audio if confirmed; This is useful when I have to make commits, since its not convenient to have the audio data on git.