import os
import librosa
import soundfile as sf
import shutil  # For folder cleanup

# Paths
RAW_BROKEN_AUDIO_PATH = "data/broken_audio"
RAW_NORMAL_AUDIO_PATH = "data/normal_audio"
CLEANED_BROKEN_AUDIO_PATH = "data/broken_audio_samples"
CLEANED_NORMAL_AUDIO_PATH = "data/normal_audio_samples"
PROCESSED_DATA_PATH = "data/processed_data"
MODELS_PATH = "models"

# Parameters
SAMPLE_RATE = 22050
CHUNK_DURATION = 3  # in seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

def split_audio(audio, sample_rate, chunk_duration=CHUNK_DURATION):
    """
    Splits audio into smaller chunks of specified duration.
    """
    num_samples_per_chunk = sample_rate * chunk_duration
    chunks = [
        audio[i : i + num_samples_per_chunk]
        for i in range(0, len(audio), num_samples_per_chunk)
        if len(audio[i : i + num_samples_per_chunk]) == num_samples_per_chunk
    ]
    return chunks

def clean_audio(input_path, output_path):
    """
    Cleans audio files from input_path and saves them to output_path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in os.listdir(input_path):
        # Skip hidden files
        if file_name.startswith('.'):
            print(f"Skipping hidden file: {file_name}")
            continue

        file_path = os.path.join(input_path, file_name)
        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            # Save cleaned audio directly
            sf.write(os.path.join(output_path, file_name), audio, sr)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def clean_and_split_audio(input_path, output_path):
    """
    Cleans and splits audio files from input_path and saves them to output_path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in os.listdir(input_path):
        # Skip hidden files
        if file_name.startswith('.'):
            print(f"Skipping hidden file: {file_name}")
            continue

        file_path = os.path.join(input_path, file_name)
        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            chunks = split_audio(audio, sr)
            for i, chunk in enumerate(chunks):
                output_file_name = f"{os.path.splitext(file_name)[0]}_chunk_{i}.wav"
                sf.write(os.path.join(output_path, output_file_name), chunk, sr)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def clean_output_folders(folders):
    """
    Deletes all files in the specified folders to ensure a clean slate.
    """
    for folder in folders:
        if os.path.exists(folder):
            print(f"Cleaning folder: {folder}")
            shutil.rmtree(folder)
            os.makedirs(folder)  # Recreate the folder
        else:
            os.makedirs(folder)
            print(f"Folder created: {folder}")

def main():
    # Folders to be cleaned
    folders_to_clean = [
        CLEANED_BROKEN_AUDIO_PATH, 
        CLEANED_NORMAL_AUDIO_PATH, 
        PROCESSED_DATA_PATH, 
        MODELS_PATH
    ]

    # Clean output folders
    print("Cleaning output folders...")
    clean_output_folders(folders_to_clean)

    # Clean and split broken audio
    print("Cleaning and splitting broken audio samples...")
    clean_and_split_audio(RAW_BROKEN_AUDIO_PATH, CLEANED_BROKEN_AUDIO_PATH)

    # Clean normal audio (no splitting)
    print("Cleaning normal audio samples...")
    clean_audio(RAW_NORMAL_AUDIO_PATH, CLEANED_NORMAL_AUDIO_PATH)

    print("Cleaning and processing completed.")

if __name__ == "__main__":
    main()
