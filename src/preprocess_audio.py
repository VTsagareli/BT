import librosa
import noisereduce as nr
import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Modify segment_length to 3 seconds (3000 milliseconds)
def split_audio(audio_file, segment_length=3000, output_path="data/audio_samples/"):
    audio = AudioSegment.from_wav(audio_file)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    # Create output directory for split files if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        segment_file_name = f"{file_name}_segment_{i // segment_length + 1}.wav"
        segment_path = os.path.join(output_path, segment_file_name)
        segment.export(segment_path, format="wav")
        segments.append(segment_path)
    
    return segments

def clean_audio(audio_file, output_path="data/cleaned_audio_samples/"):
    y, sr = librosa.load(audio_file, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
    
    # Create output directory for cleaned audio if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_file = os.path.join(output_path, os.path.basename(audio_file))
    sf.write(output_file, reduced_noise, sr)  # Use soundfile.write to save the audio file
    return output_file

def extract_mfcc_sequence(audio_file, n_mfcc=40, max_time_steps=100):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate to ensure consistent shape
    if mfcc.shape[1] < max_time_steps:
        # Pad with zeros if time steps are fewer
        pad_width = max_time_steps - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if time steps exceed the maximum
        mfcc = mfcc[:, :max_time_steps]
    
    return mfcc


def save_features_and_labels(output_path="data/processed_data/", n_mfcc=40):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    X, y = [], []

    # Process broken audio files
    broken_audio_files = [os.path.join("data/broken_audio/", f) for f in os.listdir("data/broken_audio/")]
    for original_file in broken_audio_files:
        # Split into segments
        split_files = split_audio(original_file)
        for split_file in split_files:
            # Clean audio
            cleaned_audio = clean_audio(split_file)
            # Extract MFCC features
            mfcc_features = extract_mfcc_sequence(cleaned_audio, n_mfcc=n_mfcc)
            X.append(mfcc_features)
            y.append(1)  # Label: 1 for broken

    # Process normal audio files
    normal_audio_files = [os.path.join("data/normal_audio/", f) for f in os.listdir("data/normal_audio/")]
    for original_file in normal_audio_files:
        # Split into segments
        split_files = split_audio(original_file)
        for split_file in split_files:
            # Clean audio
            cleaned_audio = clean_audio(split_file)
            # Extract MFCC features
            mfcc_features = extract_mfcc_sequence(cleaned_audio, n_mfcc=n_mfcc)
            X.append(mfcc_features)
            y.append(0)  # Label: 0 for normal

    # Convert to NumPy arrays and save
    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(output_path, "features.npy"), X)
    np.save(os.path.join(output_path, "labels.npy"), y)
    print(f"Features and labels saved to {output_path}")

if __name__ == "__main__":
    save_features_and_labels()
