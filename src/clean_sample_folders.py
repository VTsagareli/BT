import os
import shutil

# Directories to be deleted
dirs_to_delete = [
    "data/augmented_broken_audio_samples",
    "data/broken_audio_samples",
    "data/normal_audio_samples"
]

def delete_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Deleted: {directory}")
                os.makedirs(directory)  # Recreate the empty folder after deletion
                print(f"Recreated: {directory}")
            except Exception as e:
                print(f"Error deleting {directory}: {e}")
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    delete_directories(dirs_to_delete)
    print("Deletion completed successfully.")
