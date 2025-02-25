# AI Sound Anomaly Detection Model - Bachelor Thesis Project

This project is part of my Bachelor Thesis at **XU Exponential University of Applied Sciences**. The goal is to develop an **AI-powered anomaly detection system** that classifies vehicle sounds as **normal** or **faulty**, specifically identifying fuel pump issues based on audio analysis.

## Dataset Preparation - Required Downloads

Before running the pipeline, the necessary datasets must be placed in the appropriate folders.

### 1. Download the Normal Audio Dataset (Vehicle Interior Sounds)
- **Source**: [Kaggle: Open-Access Vehicle Interior Sound Dataset](https://www.kaggle.com/datasets/omkarmb/dataset-open-access-vehicle-interior-sound-dataset)
- **Steps**:
  - Download the dataset from Kaggle.
  - Extract the files.
  - **Copy approximately 1500** normal audio samples from the dataset.
  - Place them in the following directory:
    ```
    data/normal_audio_samples/
    ```

### 2. Download the Broken Audio Dataset
- The dataset of **broken vehicle sounds (faulty fuel pumps)** is available under the **Releases** section of this GitHub repository.
- **Steps**:
  - Navigate to the **Releases** section of this repository.
  - Download the provided broken audio dataset.
  - Extract and place the files in:
    ```
    data/broken_audio_samples/
    ```

### 3. Download the Unseen Test Data
This project also includes an **"unseen" dataset** to evaluate how well the model generalizes to new data.

#### Unseen Broken Audio
- Available under **Releases** on GitHub.
- Extract and place in: 
data/unseen_broken_audio_samples/

#### Unseen Normal Audio
- The unseen normal audio files must be **manually copied** from the Kaggle dataset used for normal sounds.
- **Steps**:
- Select **about 300** samples from the Kaggle dataset.
- Copy them into:
  ```
  data/unseen_normal_audio_samples/
  ```

Once these files are placed correctly, you can proceed with running the model.

---

## Running the Pipeline

The entire pipeline is automated using the **`run_pipeline.py`** script, which executes all the steps below.

### 1. Run `cleaning_and_sampling.py`
Cleans and prepares raw audio data:
- Removes **corrupt or hidden** files.
- Converts files to a standard **WAV format (22,050 Hz)**.
- Splits broken audio files into **3-second chunks** for consistency.
- Organizes files into:
data/broken_audio_samples/ 
data/normal_audio_samples/ 
data/unseen_broken_audio_samples/ 
data/unseen_normal_audio_samples/

- Deletes previously processed files to avoid conflicts.

### 2. Run `preprocess_audio.py`
Extracts features and applies augmentations:
- **Augmentation (applied to broken audio samples only)**:
- Background noise addition
- Reversing audio
- Dynamic range compression
- Time stretching
- Equalization (EQ)
- Clipping
- Scaling the amplitude
- Inserting silence
- Frequency masking
- **Feature Extraction**:
- Extracts **MFCC (Mel-Frequency Cepstral Coefficients)** for ML models.
- Saves extracted features as `.npy` files for model training.

### 3. Run `split_data.py`
Prepares data for training:
- Reads **processed features and labels**.
- **Splits data** into training (80%) and testing (20%).
- Saves the split data in:
data/processed_data/


### 4. Run `train_cnn_model.py`
Trains the Convolutional Neural Network (CNN):
- Loads training data.
- Defines a **1D CNN model** with convolutional, pooling, and fully connected layers.
- Applies **dropout and L2 regularization** to prevent overfitting.
- Uses the **Adam optimizer** and trains for **50 epochs**.
- Saves the trained model as:
models/cnn_model.pth


### 5. Run `train_ml_models.py`
Trains traditional machine learning models:
- Loads **processed training data**.
- Trains the following ML models:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Uses **10-fold cross-validation** to improve generalization.
- Saves trained models in:
models/


### 6. Run `evaluate_model.py`
Evaluates the trained models on the test dataset:
- **CNN Model Evaluation**:
- Loads the trained CNN model.
- Computes **accuracy, precision, recall, and F1-score**.
- Generates a **confusion matrix**.
- **ML Model Evaluation**:
- Loads trained ML models.
- Computes evaluation metrics (same as CNN).
- Displays performance for comparison.

### 7. Run `evaluate_unseen_data.py` (Optional)
Tests models on the unseen dataset:
- Loads previously unseen **normal and broken** audio samples.
- Extracts features in the same way as training data.
- **Evaluates** each trained model on unseen data.
- Generates classification reports and confusion matrices.

---

## Additional Utility Scripts

### `run_pipeline.py`
- Runs **all the scripts** in the correct order, automating the entire workflow.

### `count_sample.py`
- Counts the number of **normal and broken audio samples** after augmentation.

### `clean_sample_folders.py`
- **Empties** the following folders:
data/augmented_broken_audio_samples/ 
data/broken_audio_samples/ 
data/normal_audio_samples/ 
data/unseen_broken_audio_samples/ 
data/unseen_normal_audio_samples/ 
data/unseen_augmented_broken_audio_samples/

- Helps maintain a clean dataset, especially when preparing for a fresh run.

---

## Technology Stack

The project was implemented using the following technologies:

| **Category** | **Technology** |
|-------------|--------------|
| **Programming Language** | Python 3.12 |
| **Deep Learning Framework** | PyTorch |
| **Machine Learning Libraries** | Scikit-learn, NumPy |
| **Audio Processing** | Librosa, SoundFile |
| **Data Handling** | Pandas, NumPy |
| **Model Storage** | Joblib, PyTorch `.pth` |
| **Visualization** | Matplotlib, Seaborn |

This technology stack was chosen for its **efficiency** in handling audio data, **scalability**, and **robust model training capabilities**.
