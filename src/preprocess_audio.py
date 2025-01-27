import os
import librosa
import numpy as np
import soundfile as sf

# Paths
NORMAL_AUDIO_PATH = "data/normal_audio_samples"
BROKEN_AUDIO_PATH = "data/broken_audio_samples"
AUGMENTED_AUDIO_PATH = "data/augmented_broken_audio_samples"
OUTPUT_PATH = "data/processed_data"

# Parameters
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_TIME_STEPS = 100

# Augmentation Functions
def add_noise(audio, noise_factor=0.010):
    noise = np.random.normal(0, 1, len(audio))
    return audio + noise_factor * noise

def reverse_audio(audio):
    return audio[::-1]

def dynamic_range_compression(audio, threshold=0.8):
    return np.tanh(audio * threshold)

def clip_audio(audio, clipping_factor=0.95):
    return np.clip(audio, -clipping_factor, clipping_factor)

def scale_amplitude(audio, factor=0.8):
    return audio * factor

def insert_silence(audio, sr, silence_duration=0.7):
    silence = np.zeros(int(sr * silence_duration))
    return np.concatenate((silence, audio, silence))

def apply_eq(audio, lowcut, highcut, sr):
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(fft), 1 / sr)
    fft[(freqs < lowcut) | (freqs > highcut)] = 0
    return np.fft.ifft(fft).real

def frequency_mask(audio, sr, freq_range=(1000, 3000)):
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(fft), 1 / sr)
    fft[(freqs > freq_range[0]) & (freqs < freq_range[1])] = 0
    return np.fft.ifft(fft).real

# RIR-related functions commented out
# def apply_reverb(audio, rir_audio):
#     rir_audio = rir_audio / np.sqrt(np.sum(rir_audio ** 2))
#     return np.convolve(audio, rir_audio, mode="full")[: len(audio)]

def augment_audio(file_path, output_path):
    """
    Applies augmentations to a given audio file and saves the results.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Augmentations
        augmentations = {
            "noisy": add_noise(audio),
            "reversed": reverse_audio(audio),
            "compressed": dynamic_range_compression(audio),
            "sped_up": librosa.effects.time_stretch(audio, rate=1.25),
            "eq": apply_eq(audio, lowcut=300, highcut=3000, sr=sr),
            "clipped": clip_audio(audio),
            "scaled": scale_amplitude(audio, factor=0.8),
            "silenced": insert_silence(audio, sr),
        }

        # Frequency masking
        augmentations["freq_masked"] = frequency_mask(audio, sr)

        # RIR augmentation skipped
        # rir_file = "data/rir_file.wav"
        # if os.path.exists(rir_file):
        #     rir_audio, _ = librosa.load(rir_file, sr=SAMPLE_RATE)
        #     augmentations["reverb"] = apply_reverb(audio, rir_audio)

        # Save all augmentations
        for aug_name, aug_audio in augmentations.items():
            sf.write(os.path.join(output_path, f"{aug_name}_{os.path.basename(file_path)}"), aug_audio, sr)

    except Exception as e:
        print(f"Error augmenting {file_path}: {e}")

# Feature Extraction
def extract_mfcc(audio_file, n_mfcc=N_MFCC, max_time_steps=MAX_TIME_STEPS):
    """
    Extract MFCC features from an audio file.
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
        mfcc_features = extract_mfcc(file)
        X.append(mfcc_features)
        y.append(0)  # Label: 0 for normal

    # Process broken audio samples
    print("Processing broken audio samples...")
    broken_files = [os.path.join(BROKEN_AUDIO_PATH, f) for f in os.listdir(BROKEN_AUDIO_PATH) if f.endswith(".wav")]
    for file in broken_files:
        mfcc_features = extract_mfcc(file)
        X.append(mfcc_features)
        y.append(1)  # Label: 1 for broken

    # Process augmented broken audio samples
    print("Processing augmented broken audio samples...")
    broken_files = [os.path.join(BROKEN_AUDIO_PATH, f) for f in os.listdir(BROKEN_AUDIO_PATH) if f.endswith(".wav")]
    for file in broken_files:
        augment_audio(file, AUGMENTED_AUDIO_PATH)

    augmented_files = [os.path.join(AUGMENTED_AUDIO_PATH, f) for f in os.listdir(AUGMENTED_AUDIO_PATH) if f.endswith(".wav")]
    for file in augmented_files:
        mfcc_features = extract_mfcc(file)
        X.append(mfcc_features)
        y.append(1)  # Label: 1 for augmented broken

    # Convert to NumPy arrays and save
    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(output_path, "features.npy"), X)
    np.save(os.path.join(output_path, "labels.npy"), y)
    print(f"Features and labels saved to {output_path}")

if __name__ == "__main__":
    save_features_and_labels()
