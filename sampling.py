
import torch
import torch.nn as nn
from model import SphereEncoder, Config
from torchvision.utils import save_image
import os
import argparse
import math
from PIL import Image
import torchvision.transforms as transforms
import torchvision

def load_image(path, size=32):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0)

def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to find latest
        import glob
        # We need a default to search checkpoints if not provided
        search_dir = './checkpoints'
        if not os.path.exists(search_dir):
            print(f"Directory {search_dir} does not exist. Please specify --checkpoint.")
            return

        files = glob.glob(os.path.join(search_dir, "checkpoint_*.pth"))
        if files:
            latest = os.path.join(search_dir, "checkpoint_latest.pth")
            checkpoint_path = latest if os.path.exists(latest) else max(files, key=os.path.getctime)
        else:
            print("No checkpoint found.")
            return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer Config from Checkpoint
    ckpt_config = checkpoint.get('config', {})
    if isinstance(ckpt_config, dict):
        config = Config(**ckpt_config)
    else:
        # If it's already a Config object or something else
        config = ckpt_config
    
    # Ensure device is correct
    config.device = device
    
    # Load Model with correct config
    model = SphereEncoder(config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Determine Model Name for output folder
    # Use parent directory name if available, otherwise filename without ext
    abs_path = os.path.abspath(checkpoint_path)
    parts = abs_path.split(os.sep)
    if 'results' in parts:
        model_name = parts[parts.index('results') + 1]
    else:
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    output_dir = os.path.join(args.dest_folder, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    # Reconstruction Mode
    if args.input:
        print(f"Reconstructing image: {args.input}")
        original = load_image(args.input, size=config.image_size).to(device)
        
        with torch.no_grad():
            v = model.encode(original)
            
            outputs = [original]
            for s in [1, 2, 3, 4]:
                recon = model.decode_multistep(v, steps=s)
                outputs.append(recon)
                print(f"Reconstructed with {s} steps.")
            
            # Save Comparison
            grid = torch.cat(outputs, dim=0)
            grid = grid * 0.5 + 0.5
            
            # Sanitize filename
            input_base = os.path.basename(args.input)
            clean_name = "".join([c if c.isalnum() else "_" for c in os.path.splitext(input_base)[0]])
            save_filename = f"{clean_name}_recon.png"
            
            save_path = os.path.join(output_dir, save_filename)
            # Horizontal layout: Original then steps 1, 2, 3, 4
            save_image(grid, save_path, nrow=5)
            print(f"Saved comparison to {save_path}")
            
    else:
        # Generation Mode
        print(f"Generating {args.num_samples} random samples...")
        with torch.no_grad():
            z = torch.randn(args.num_samples, config.latent_dim, device=device)
            v = torch.nn.functional.normalize(z, p=2, dim=1)
            v = v * math.sqrt(config.latent_dim)
            
            all_steps = []
            for s in [1, 2, 3, 4]:
                print(f"Generating with {s} steps...")
                generated = model.decode_multistep(v, steps=s)
                generated = generated * 0.5 + 0.5
                all_steps.append(generated)
            
            # One column per noise (num_samples columns), rows are steps 1, 2, 3, 4
            full_grid = torch.cat(all_steps, dim=0)
            save_path = os.path.join(output_dir, 'samples_multistep.png')
            save_image(full_grid, save_path, nrow=args.num_samples)
            print(f"Saved consolidated samples to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--dest_folder', type=str, default='samples', help='Parent directory for outputs')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--input', type=str, default=None, help='Input image for reconstruction')
    args = parser.parse_args()
    sample(args)
