
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import math

# Configuration
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    learning_rate = 1e-3
    latent_dim = 128  # Dimension of the spherical latent space
    epochs = 10
    iterations_per_epoch = 100
    dataset_path = './data'
    output_path = './results'
    sigma_max = 0.2  # Maximum noise magnitude for training

    def __init__(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
config = Config()

# Model Definition
class SphereEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(SphereEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 32x32 -> 1x1 (flattened)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )

        # Decoder: Latent -> 32x32
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.Tanh() # Output ranges [-1, 1]
        )

    def encode(self, x):
        z = self.encoder(x)
        # Normalize to unit sphere
        v = torch.nn.functional.normalize(z, p=2, dim=1)
        return v

    def decode(self, v):
        x = self.decoder_input(v)
        x = x.view(-1, 128, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x, training=True):
        # 1. Encode
        v = self.encode(x)
        
        # 2. Add Noise (during training only)
        if training:
            # Random noise magnitude sigma ~ U(0, sigma_max)
            # Sample one sigma per batch element or one for the whole batch? 
            # Paper mentions "where sigma is jittered magnitude". Let's do per-batch for simplicity/efficiency.
            sigma = torch.rand(v.shape[0], 1, device=v.device) * config.sigma_max
            
            # Random direction e ~ N(0, I)
            e = torch.randn_like(v)
            
            # Add noise: v_noisy = v + sigma * e
            v_noisy = v + sigma * e
            
            # 3. Re-project to Sphere
            v_reprojected = torch.nn.functional.normalize(v_noisy, p=2, dim=1)
            
            # 4. Decode
            recon = self.decode(v_reprojected)
            return recon, v, v_reprojected
        else:
            # During inference, just decode the exact latent
            recon = self.decode(v)
            return recon

# Data Preparation
def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])
    
    # Download CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False,
                                           download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

# Training Loop
def train():
    print(f"Using device: {config.device}")
    
    model = SphereEncoder(latent_dim=config.latent_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    trainloader, testloader = get_dataloader()
    
    print("Starting training...")
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        
        # Create an iterator to manually control the number of iterations
        data_iter = iter(trainloader)
        
        for i in range(config.iterations_per_epoch):
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(trainloader)
                images, _ = next(data_iter)
                
            images = images.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass with noise injection
            recon_images, v, v_noisy = model(images, training=True)
            
            loss = criterion(recon_images, images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / config.iterations_per_epoch
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}")
        
        # Save check-in images and test
        test(model, epoch + 1, testloader)

    print("Training finished.")
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.output_path, 'sphere_encoder.pth'))

# Testing and Generation
def test(model, epoch, testloader):
    model.eval()
    with torch.no_grad():
        # 1. Reconstruction Test
        # Get a batch of test images
        data_iter = iter(testloader)
        real_images, _ = next(data_iter)
        real_images = real_images.to(config.device)
        
        recon_images = model(real_images, training=False)
        
        # Save comparison
        comparison = torch.cat([real_images[:8], recon_images[:8]])
        # Denormalize for saving: [-1, 1] -> [0, 1]
        comparison = comparison * 0.5 + 0.5
        save_image(comparison, os.path.join(config.output_path, f'epoch_{epoch}_reconstruction.png'))
        
        # 2. Generation Test (Sampling from Sphere)
        # Sample random points on the sphere
        # z ~ N(0, I)
        z = torch.randn(16, config.latent_dim, device=config.device)
        # v = z / ||z||
        v = torch.nn.functional.normalize(z, p=2, dim=1)
        
        generated_images = model.decode(v)
        generated_images = generated_images * 0.5 + 0.5
        save_image(generated_images, os.path.join(config.output_path, f'epoch_{epoch}_generated.png'), nrow=4)

if __name__ == "__main__":
    train()
