import os
import subprocess
import sys

def main():
    """
    Launches the training with optimal hyperparameters inferred from the Sphere Encoder paper.
    
    Paper Settings for CIFAR-10 (Small Image 32x32):
    - Latent Dim: 16x16x8 = 2048
    - Epochs: 5000+ (We start with 1000 for practicality)
    - Batch Size: 256 (implied from standard practices)
    - Steps per Epoch: ~200 (CIFAR-10 has 50k images, 50000/256 ~= 195)
    
    This script assumes 'training.py' handles the logic.
    """
    
    # Optimal Configuration
    cmd = [
        sys.executable, "training.py",
        "--dataset", "cifar10",
        "--batch_size", "256",
        "--epochs", "1000",
        "--steps", "200", # Approx 1 epoch of CIFAR-10
        "--no-resume", # Start fresh because model architecture/dim changed
        # Add eval and checkpoint frequency (epochs)
        "--eval-frequency", "100",
        "--checkpoint-frequency", "100"
    ]
    
    cmd = [
        sys.executable, "training.py",
        "--dataset", "flowers102",
        "--batch_size", "64",
        "--epochs", "1000",
        "--steps", "128", # Approx 1 epoch of flowers102 (8200 images)
        
        # Add eval and checkpoint frequency (epochs)
        "--eval-frequency", "50",
        "--checkpoint-frequency", "100",
        "--no-resume" # Start fresh because model architecture/dim changed
    ]


    print(f"Starting Optimal Training Run...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")

if __name__ == "__main__":
    main()
