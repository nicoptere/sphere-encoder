
import os
import torchvision
import torchvision.transforms as transforms
from model import Config

def download_cifar10():
    # Target directory from Config
    data_dir = Config.dataset_path
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading and preparing CIFAR-10 dataset in: {data_dir}")
    
    # Use torchvision to download and extract
    # This will handle the download, verification and extraction automatically
    try:
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, 
            train=True,
            download=True, 
            transform=transforms.ToTensor()
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, 
            train=False,
            download=True, 
            transform=transforms.ToTensor()
        )
        
        print("\nSuccessfully downloaded and installed CIFAR-10.")
        print(f"Dataset location: {os.path.abspath(os.path.join(data_dir, 'cifar-10-batches-py'))}")
        
        print("\nTo train on this dataset, run:")
        print("python training.py --dataset cifar10")
        
    except Exception as e:
        print(f"\nError downloading CIFAR-10: {e}")

if __name__ == "__main__":
    download_cifar10()
