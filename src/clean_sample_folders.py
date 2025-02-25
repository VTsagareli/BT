import os
import shutil

# Directories to handle automatically
DEFAULT_FOLDERS = [
    "data/broken_audio_samples",
    "data/normal_audio_samples",
    "data/augmented_broken_audio_samples",
    "data/processed_data",
    "models",
    "data/unseen_broken_audio_samples",
    "data/unseen_normal_audio_samples",
    "data/unseen_augmented_broken_audio_samples",
    "data/unseen_broken_audio",
    "data/unseen_normal_audio"
]

# Directories that require user confirmation
CONFIRM_FOLDERS = [
    "data/normal_audio",
    "data/broken_audio"
]

def clean_folder(folder):
    """
    Deletes all files in the specified folder and recreates it.
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
        print(f"Cleaned and recreated folder: {folder}")
    else:
        os.makedirs(folder)
        print(f"Folder created: {folder}")

def confirm_and_clean():
    """
    Asks the user for confirmation before cleaning certain folders.
    """
    for folder in CONFIRM_FOLDERS:
        user_input = input(f"Do you want to clean the folder '{folder}'? (y/n): ").strip().lower()
        if user_input == 'y':
            clean_folder(folder)
        else:
            print(f"Skipping folder: {folder}")

def main():
    print("Cleaning default folders...")
    for folder in DEFAULT_FOLDERS:
        clean_folder(folder)

    print("\nRequesting confirmation for additional folders...")
    confirm_and_clean()

    print("\nAll necessary folders have been cleaned and recreated successfully.")

if __name__ == "__main__":
    main()
