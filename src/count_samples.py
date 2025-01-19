import os

# Directories to count files in
directories = {
    "broken_audio_samples": "data/broken_audio_samples/",
    "augmented_broken_audio_samples": "data/augmented_broken_audio_samples/",
    "normal_audio_samples": "data/normal_audio_samples/"
}

# Count files in each directory
def count_samples(directories):
    for name, path in directories.items():
        if os.path.exists(path):
            num_files = len([f for f in os.listdir(path) if f.endswith(".wav")])
            print(f"{name}: {num_files} samples")
        else:
            print(f"{name}: Directory not found")

if __name__ == "__main__":
    count_samples(directories)
