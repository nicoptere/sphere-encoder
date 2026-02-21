
import os
import requests
import zipfile
import io
from tqdm import tqdm

def download_and_unzip():
    # URL for FFHQ 64x64 ZIP on Hugging Face
    url = "https://huggingface.co/datasets/Dmini/FFHQ-64x64/resolve/main/ffhq-64x64.zip"
    
    # Target directory
    target_dir = os.path.join("data", "ffhq_64")
    zip_path = os.path.join("data", "ffhq_64.zip")
    
    os.makedirs("data", exist_ok=True)
    
    print(f"Downloading faces dataset (ZIP format) from:\n{url}")
    
    # Download with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 1024):
                size = file.write(data)
                bar.update(size)
                
        print("\nDownload complete. Unzipping...")
        
        # Unzip
        # We unzip to data/ffhq_64/images/ for compatibility
        extract_to = os.path.join(target_dir, "images")
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to show progress
            members = zip_ref.namelist()
            for member in tqdm(members, desc="Extracting"):
                # Clean up path if needed, but FFHQ zip is usually flat or simple
                zip_ref.extract(member, extract_to)
                
        print(f"\nSuccessfully extracted dataset to: {target_dir}")
        print("Cleaning up ZIP file...")
        os.remove(zip_path)
        
        print("\nTo train on this dataset, run:")
        print(f"python training.py --dataset {target_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    download_and_unzip()
