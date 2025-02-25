import subprocess

def run_script(script_name):
    """
    Runs a Python script and handles any errors.
    """
    print(f"\nRunning {script_name}...")
    try:
        subprocess.run(["python", f"src/{script_name}"], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        exit(1)

def main():
    print("Starting the end-to-end audio classification pipeline...\n")

    # Step 1: Clean and sample audio data
    run_script("cleaning_and_sampling.py")

    # Step 2: Preprocess audio (augmentations + feature extraction)
    run_script("preprocess_audio.py")

    # Step 3: Split data for training and testing
    run_script("split_data.py")

    # Step 4a: Train the CNN model
    run_script("train_cnn_model.py")

    # Step 4b: Train classic ML models
    run_script("train_ml_models.py")

    # Step 5: Evaluate all trained models
    run_script("evaluate_model.py")

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()