
import torch
import torch.nn as nn
from model import SphereEncoder, Config
from torchvision.utils import save_image
import os
import argparse
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
    config = Config()
    device = config.device
    
    # Load Model
    model = SphereEncoder(config).to(device)
    
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to find latest
        import glob
        if not os.path.exists(config.checkpoint_dir):
            print(f"Checkpoint directory {config.checkpoint_dir} does not exist.")
            return

        files = glob.glob(os.path.join(config.checkpoint_dir, "checkpoint_*.pth"))
        if files:
            # Prefer checkpoint_latest if exists, else max time
            latest = os.path.join(config.checkpoint_dir, "checkpoint_latest.pth")
            if os.path.exists(latest):
                checkpoint_path = latest
            else:
                checkpoint_path = max(files, key=os.path.getctime)
        else:
            print("No checkpoint found.")
            return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    steps_list = [1, 2, 4]
    if args.steps != 1: # user specified something specific? 
        # Actually user said "implement 1,2, 4 steps".
        # If user passes --steps 10, maybe we just do 10?
        # But for now let's default to [1, 2, 4] unless overridden?
        # The parser default is 1. If user leaves default, we might execute [1, 2, 4]?
        pass

    # Reconstruction Mode
    if args.input:
        print(f"Reconstructing image: {args.input}")
        original = load_image(args.input, size=config.image_size).to(device)
        
        with torch.no_grad():
            v = model.encode(original)
            
            outputs = [original]
            for s in [1, 2, 4]:
                recon = model.decode_multistep(v, steps=s)
                outputs.append(recon)
                print(f"Reconstructed with {s} steps.")
            
            # Save Comparison
            grid = torch.cat(outputs, dim=0)
            grid = grid * 0.5 + 0.5
            save_path = args.output.replace('.png', '_recon.png')
            save_image(grid, save_path, nrow=4)
            print(f"Saved comparison to {save_path}")
            
    else:
        # Generation Mode
        print("Generating random samples...")
        with torch.no_grad():
            z = torch.randn(args.num_samples, config.latent_dim, device=device)
            v = torch.nn.functional.normalize(z, p=2, dim=1)
            
            for s in [1, 2, 4]:
                print(f"Generating with {s} steps...")
                generated = model.decode_multistep(v, steps=s)
                generated = generated * 0.5 + 0.5
                
                save_path = args.output.replace('.png', f'_step_{s}.png')
                save_image(generated, save_path, nrow=int(args.num_samples**0.5))
                print(f"Saved samples to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='sample.png', help='Output image path base name')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=1, help='(Deprecated) Steps are now 1, 2, 4 by default')
    parser.add_argument('--input', type=str, default=None, help='Input image for reconstruction')
    args = parser.parse_args()
    sample(args)
