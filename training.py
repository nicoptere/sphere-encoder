
import os
import argparse
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import SphereEncoder, Config
import glob

def find_latest_run(dataset_name):
    # Find all directories in results/ that start with dataset_name
    search_path = os.path.join('results', f"{dataset_name}_*")
    runs = glob.glob(search_path)
    if not runs:
        return None
    
    # Sort by creation time (or name if timestamp format allows)
    # Our format is YYYYMMDD_HHMMSS, so alphabetical sort works for latest
    runs.sort()
    return runs[-1]

def get_dataset(config, args):
    dataset_name = args.dataset.lower()
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    transform = None # Will set after loading sample or finding size?
    # Actually we need transform to load... 
    # But for standard datasets like CIFAR/Flowers we know default size, 
    # OR we can Resize to a target if specified? 
    # User said "infer image size ... from data". 
    # This implies we load raw first? Or we use the dataset's native size.
    
    # Heuristic: 
    # If standard dataset, use its default. 
    # If folder, check first image.
    
    if dataset_name == 'cifar10':
        native_size = 32
        transform = transforms.Compose(transform_list) # Native 32x32
        dataset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform)
        
    elif 'flower' in dataset_name: # Flowers102
        native_size = 256 # Usually large, we can/should Resize to common 256 or 128? 
        # Let's target 256 as previous request.
        transform = transforms.Compose([transforms.Resize((256, 256))] + transform_list)
        dataset = torchvision.datasets.Flowers102(root=config.dataset_path, split='test', download=True, transform=transform)
    else:
        # Fallback for custom folder or ImageFolder?
        # Assuming arguments might point to a path?
        if os.path.exists(args.dataset):
             # Try ImageFolder
             # We need to determine size first? 
             # Just picking a safe size for now or erroring out?
             # Let's default to resize 128 if unknown.
             transform = transforms.Compose([transforms.Resize((128, 128))] + transform_list)
             dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset

def infer_config(dataset, args):
    # Peek at first sample
    sample, _ = dataset[0]
    c, h, w = sample.shape
    
    # Defaults
    if args.model == 'vit':
        # Paper recommendations for ViT:
        latent_dim = 2048
        patch_size = 2
        embed_dim = 512
        depth = 6
        num_heads = 8
        
        if h >= 256: 
            patch_size = 8
            latent_dim = 32 * 32 * 64
            embed_dim = 1024
            depth = 24
            num_heads = 16
        elif h >= 64:
            patch_size = 4
            latent_dim = 4096
    else:
        # Defaults for ConvNet
        # 512 is generally enough for ConvNet on 32x32-128x128
        latent_dim = 512
        patch_size = 2 # Unused but keep for safety
        embed_dim = 512
        depth = 6
        num_heads = 8
        
        if h >= 256: latent_dim = 1024

    # Apply Overrides from Args
    if args.latent_dim is not None: latent_dim = args.latent_dim
    if args.patch_size is not None: patch_size = args.patch_size
    if args.embed_dim is not None: embed_dim = args.embed_dim
    if args.depth is not None: depth = args.depth
    if args.num_heads is not None: num_heads = args.num_heads

    print(f"Inferred configuration: Size={h}x{w}, Model={args.model}, Latent={latent_dim}")
    
    config = Config()
    config.model_type = args.model
    config.image_size = h
    config.patch_size = patch_size
    config.embed_dim = embed_dim
    config.depth = depth
    config.num_heads = num_heads
    
    config.channels = c
    config.latent_dim = latent_dim
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.iterations_per_epoch = args.steps
    config.eval_frequency = args.eval_frequency
    config.checkpoint_frequency = args.checkpoint_frequency
    config.learning_rate = args.lr
    config.sigma_max = args.sigma_max
    
    dataset_name = os.path.basename(args.dataset).replace('.', '')
    
    # Resume Logic
    if args.resume:
        latest_run = find_latest_run(dataset_name)
        if latest_run:
            print(f"Resuming from latest run: {latest_run}")
            config.results_dir = latest_run
            config.checkpoint_dir = os.path.join(config.results_dir, 'checkpoints')
            config.images_dir = os.path.join(config.results_dir, 'images')
            return config
    
    # New Run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_name}_{timestamp}"
    
    config.results_dir = os.path.join('results', run_name)
    config.checkpoint_dir = os.path.join(config.results_dir, 'checkpoints')
    config.images_dir = os.path.join(config.results_dir, 'images')
    
    return config

def train(args):
    # Dummy config to help get_dataset finding paths? 
    # We pass args mostly.
    
    # 1. Load Dataset (to infer size)
    temp_config = Config() # Defaults
    dataset = get_dataset(temp_config, args)
    
    # 2. Configure
    config = infer_config(dataset, args)
    
    # Create Dirs
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.images_dir, exist_ok=True)
    
    print(f"Results will be saved to: {config.results_dir}")
    
    # 3. Model & Loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    model = SphereEncoder(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    start_epoch = 0
    
    # Load Checkpoint if Resuming
    if args.resume and os.path.exists(config.checkpoint_dir):
        checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_latest.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['state_dict'])
            # We don't load optimizer/scheduler state strictly unless required, 
            # but usually good practice. For now, let's keep it simple or user might complain about errors if structures change.
            # But wait, user wants to resume training.
            try:
                if 'optimizer' in checkpoint:
                   optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epoch' in checkpoint:
                   start_epoch = checkpoint['epoch']
                   print(f"Resuming from Epoch {start_epoch}")
            except Exception as e:
                print(f"Warning: Could not load optimizer or epoch state: {e}")
                
            # If user increased epochs, we continue.
            # config.epochs is already set from args.epochs
            # config.iterations_per_epoch is set from args.steps
            
    
    # 4. Train Loop
    print("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        data_iter = iter(dataloader)
        
        for i in range(config.iterations_per_epoch):
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, _ = next(data_iter)
                
            images = images.to(config.device)
            optimizer.zero_grad()
            
            # --- Paper Implementation: Noise Injection ---
            # 1. Clean Latent
            v_clean = model.encode(images)
            
            # 2. Generate Noise Parameters
            # Jittering Sigma: r ~ U(0, 1)
            # sigma = r * sigma_max
            r = torch.rand(1, device=config.device) 
            sigma = r * config.sigma_max
            
            # Random direction e ~ N(0, I)
            e = torch.randn_like(v_clean)
            
            # 3. Large Noise Latent (v_NOISY)
            # v_NOISY = f(v + sigma * e)
            v_noisy = torch.nn.functional.normalize(v_clean + sigma * e, p=2, dim=1)
            # Fix: Scale to radius sqrt(L) matches model.encode output
            v_noisy = v_noisy * math.sqrt(config.latent_dim)
            
            # 4. Small Noise Latent (v_noisy_sub)
            # s ~ U(0, 0.5)
            # sigma_sub = s * sigma
            s = torch.rand(1, device=config.device) * 0.5
            sigma_sub = s * sigma
            v_noisy_sub = torch.nn.functional.normalize(v_clean + sigma_sub * e, p=2, dim=1)
            # Fix: Scale to radius sqrt(L)
            v_noisy_sub = v_noisy_sub * math.sqrt(config.latent_dim)
            
            # 5. Compute Loss
            # Returns: l_pix_recon, l_pix_con, l_lat_con, recon_sub
            l_recon, l_pix_con, l_lat_con, recon = model.compute_loss(images, v_noisy, v_noisy_sub, v_clean)
            
            # Weights from Paper (CIFAR-10 / Small Image)
            # Table 16: L_pix-recon=1.0, L_pix-con=0.5, L_lat-con=0.1
            loss = 1.0 * l_recon + 0.5 * l_pix_con + 0.1 * l_lat_con
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Debug log spread (first iter of batch)
            if i == 0:
                print(f"  Losses: Total={loss.item():.4f} | Recon={l_recon.item():.4f} | PixCon={l_pix_con.item():.4f} | LatCon={l_lat_con.item():.4f}")
            
        print(f"Epoch [{epoch+1}/{config.epochs}] Loss: {running_loss/config.iterations_per_epoch:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % config.checkpoint_frequency == 0 or (epoch + 1) == config.epochs:
            checkpoint_data = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'config': config.__dict__ # Save config for sampling!
            }
            # Save latest for resume
            torch.save(checkpoint_data, os.path.join(config.checkpoint_dir, "checkpoint_latest.pth"))
            # Save periodic for history
            torch.save(checkpoint_data, os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"  Saved checkpoint at epoch {epoch+1}")
        
        # Sample (1, 2, 4 steps)
        if (epoch + 1) % config.eval_frequency == 0 or (epoch + 1) == config.epochs:
            with torch.no_grad():
                model.eval()
                # Reconstruction
                recon = model(images[:4].to(config.device))
                if isinstance(recon, tuple): recon = recon[0] # Handle tuple return if forward changed
                save_image(torch.cat([images[:4], recon]) * 0.5 + 0.5, 
                           os.path.join(config.images_dir, f"epoch_{epoch+1}_recon.png"), nrow=4)
                
                # Generation (1 Step only)
                z = torch.randn(8, config.latent_dim, device=config.device)
                v = torch.nn.functional.normalize(z, p=2, dim=1)
                # Scale v to match the sphere radius used in training!
                v = v * math.sqrt(config.latent_dim)
                
                gen = model.decode_multistep(v, steps=1)
                save_image(gen * 0.5 + 0.5, 
                           os.path.join(config.images_dir, f"epoch_{epoch+1}_sample.png"), nrow=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (cifar10, flowers102) or path')
    parser.add_argument('--model', type=str, default='convnet', choices=['convnet', 'vit'], help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Total number of epochs')
    parser.add_argument('--steps', type=int, default=100, help='Steps (iterations) per epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Model Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=None, help='Latent dimension (overrides inferred)')
    parser.add_argument('--sigma_max', type=float, default=0.2, help='Noise injection sigma_max')
    
    # ViT specific
    parser.add_argument('--patch_size', type=int, default=None, help='ViT patch size (overrides inferred)')
    parser.add_argument('--embed_dim', type=int, default=None, help='ViT embedding dimension (overrides inferred)')
    parser.add_argument('--depth', type=int, default=None, help='ViT depth (overrides inferred)')
    parser.add_argument('--num_heads', type=int, default=None, help='ViT number of heads (overrides inferred)')
    
    # Maintenance
    parser.add_argument('--eval-frequency', type=int, default=100, help='Evaluation frequency (in epochs)')
    parser.add_argument('--checkpoint-frequency', type=int, default=100, help='Checkpoint frequency (in epochs)')
    
    # Resume flag: Default True
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from latest run')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Start a new run')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    train(args)
