
import os
import argparse
import datetime
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
    
    latent_dim = 128
    if h >= 256: latent_dim = 512
    elif h >= 128: latent_dim = 256
    elif h >= 64: latent_dim = 256
    
    print(f"Inferred configuration from data: Size={h}x{w}, Channels={c}")
    print(f"Selecting Latent Dim: {latent_dim}")
    
    config = Config()
    config.image_size = h
    config.channels = c
    config.latent_dim = latent_dim
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.iterations_per_epoch = args.steps
    
    dataset_name = os.path.basename(args.dataset).replace('.', '')
    
    # Resume Logic
    if args.resume:
        latest_run = find_latest_run(dataset_name)
        if latest_run:
            print(f"Resuming from latest run: {latest_run}")
            config.results_dir = latest_run
            config.checkpoint_dir = os.path.join(config.results_dir, 'checkpoints')
            return config
    
    # New Run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_name}_{timestamp}"
    
    config.results_dir = os.path.join('results', run_name)
    config.checkpoint_dir = os.path.join(config.results_dir, 'checkpoints')
    
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
            recon, _, _ = model(images, training=True)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{config.epochs}] Loss: {running_loss/config.iterations_per_epoch:.4f}")
        
        # Save Checkpoint
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'config': config.__dict__ # Save config for sampling!
        }, os.path.join(config.checkpoint_dir, "checkpoint_latest.pth"))
        
        # Sample (1, 2, 4 steps)
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                model.eval()
                # Reconstruction
                recon = model(images[:4].to(config.device), training=False)
                save_image(torch.cat([images[:4], recon]) * 0.5 + 0.5, 
                           os.path.join(config.results_dir, f"epoch_{epoch+1}_recon.png"), nrow=4)
                
                # Generation (1, 2, 4 Steps)
                z = torch.randn(8, config.latent_dim, device=config.device)
                v = torch.nn.functional.normalize(z, p=2, dim=1)
                
                for steps in [1, 2, 4]:
                    gen = model.decode_multistep(v, steps=steps)
                    save_image(gen * 0.5 + 0.5, 
                               os.path.join(config.results_dir, f"epoch_{epoch+1}_step_{steps}.png"), nrow=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (cifar10, flowers102) or path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Total number of epochs')
    parser.add_argument('--steps', type=int, default=100, help='Steps (iterations) per epoch')
    
    # Resume flag: Default True
    # We use a trick: --resume defaults to True. --no-resume sets it to False.
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from latest run')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Start a new run')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    train(args)
