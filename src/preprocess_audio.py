import os
import librosa
import numpy as np
import soundfile as sf
import random

# Paths
NORMAL_AUDIO_PATH = "data/normal_audio_samples"
BROKEN_AUDIO_PATH = "data/broken_audio_samples"
AUGMENTED_AUDIO_PATH = "data/augmented_broken_audio_samples"

UNSEEN_NORMAL_AUDIO_PATH = "data/unseen_normal_audio_samples"
UNSEEN_BROKEN_AUDIO_PATH = "data/unseen_broken_audio_samples"
UNSEEN_AUGMENTED_BROKEN_AUDIO_PATH = "data/unseen_augmented_broken_audio_samples"

OUTPUT_PATH = "data/processed_data"

# Parameters
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_TIME_STEPS = 100

# Augmentation Functions
def add_background_noise(audio, noise_factor=0.02):
    """Adds natural background noise from a noise sample or random noise."""
    noise = np.random.normal(0, 1, len(audio))
    return audio + noise_factor * noise

def apply_reverb(audio, reverb_factor=0.4):
    """Applies artificial reverberation (simulates room echo)."""
    return np.convolve(audio, np.hanning(50) * reverb_factor, mode="same")

def time_stretch(audio, rate_range=(0.9, 1.1)):
    """Alters speed slightly without changing pitch."""
    rate = random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, pitch_range=(-2, 2)):
    """Shifts pitch up/down slightly to simulate different environments."""
    n_steps = random.uniform(pitch_range[0], pitch_range[1]) 
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def augment_audio(file_path, output_path):
    """
    Applies **natural** augmentations to a given audio file and saves the results.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # New Augmentations
        augmentations = {
            "background_noise": add_background_noise(audio),
            "reverb": apply_reverb(audio),
            "time_stretched": time_stretch(audio),
            "pitch_shifted": pitch_shift(audio, sr),
        }

        # Save all augmentations
        for aug_name, aug_audio in augmentations.items():
            sf.write(os.path.join(output_path, f"{aug_name}_{os.path.basename(file_path)}"), aug_audio, sr)

    except Exception as e:
        print(f"Error augmenting {file_path}: {e}")

# Feature Extraction
def extract_mfcc(audio_file, n_mfcc=N_MFCC, max_time_steps=MAX_TIME_STEPS):
    """
    Extracts MFCC features from an audio file.
    """
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to ensure consistent shape
    if mfcc.shape[1] < max_time_steps:
        pad_width = max_time_steps - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_time_steps]

    return mfcc

# Main Preprocessing Function
def save_features_and_labels(output_path=OUTPUT_PATH):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    X, y = [], []

    # Process normal audio samples
    print("Processing normal audio samples...")
    normal_files = [os.path.join(NORMAL_AUDIO_PATH, f) for f in os.listdir(NORMAL_AUDIO_PATH) if f.endswith(".wav")]
    for file in normal_files:
        X.append(extract_mfcc(file))
        y.append(0)  

    # Process broken audio samples
    print("Processing broken audio samples...")
    broken_files = [os.path.join(BROKEN_AUDIO_PATH, f) for f in os.listdir(BROKEN_AUDIO_PATH) if f.endswith(".wav")]
    for file in broken_files:
        X.append(extract_mfcc(file))
        y.append(1)  

    # Process augmented broken audio samples
    print("Processing augmented broken audio samples...")
    for file in os.listdir(AUGMENTED_AUDIO_PATH):
        if file.endswith(".wav"):
            X.append(extract_mfcc(os.path.join(AUGMENTED_AUDIO_PATH, file)))
            y.append(1)  

    # Process unseen normal audio samples
    print("Processing unseen normal audio samples...")
    for file in os.listdir(UNSEEN_NORMAL_AUDIO_PATH):
        if file.endswith(".wav"):
            X.append(extract_mfcc(os.path.join(UNSEEN_NORMAL_AUDIO_PATH, file)))
            y.append(0)  

    # Process unseen broken audio samples
    print("Processing unseen broken audio samples...")
    for file in os.listdir(UNSEEN_BROKEN_AUDIO_PATH):
        if file.endswith(".wav"):
            augment_audio(os.path.join(UNSEEN_BROKEN_AUDIO_PATH, file), UNSEEN_AUGMENTED_BROKEN_AUDIO_PATH)

    # Process unseen augmented broken audio samples
    print("Processing unseen augmented broken audio samples...")
    for file in os.listdir(UNSEEN_AUGMENTED_BROKEN_AUDIO_PATH):
        if file.endswith(".wav"):
            X.append(extract_mfcc(os.path.join(UNSEEN_AUGMENTED_BROKEN_AUDIO_PATH, file)))
            y.append(1)  

    # Convert to NumPy arrays and save
    X, y = np.array(X), np.array(y)

    np.save(os.path.join(output_path, "features.npy"), X)
    np.save(os.path.join(output_path, "labels.npy"), y)
    print(f"Features and labels saved to {output_path}")

if __name__ == "__main__":
    save_features_and_labels()
