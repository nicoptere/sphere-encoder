
import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.nn.functional as F

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Defaults (will be overwritten by args)
    image_size = 32
    channels = 3
    dataset_path = './data'
    
    # Training
    batch_size = 128
    learning_rate = 1e-3 # Restored to 1e-3 for faster convergence with ConvNet
    latent_dim = 2048
    epochs = 1000
    iterations_per_epoch = 100
    
    # Model
    sigma_max = 0.2  # Restored to 0.2 (Original working value). High noise prevents early learning.
    
    # ViT Hyperparameters
    patch_size = 2
    embed_dim = 512
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0
    
    # Switch
    model_type = 'convnet' # 'convnet' or 'vit'
    
    # Paths (will be overwritten)
    checkpoint_dir = './checkpoints'
    results_dir = './results'
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],
            vgg[4:9],
            vgg[9:16],
            vgg[16:23]
        ])
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # x, y assumed to be [-1, 1], normalize to [0, 1] then to VGG mean/std
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MLPMixer(nn.Module):
    def __init__(self, num_patches, dim, depth=4):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ))
            
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) # Simple token mixing via linear layers
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTSphereEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.patch_embed = PatchEmbed(
            img_size=config.image_size, 
            patch_size=config.patch_size, 
            in_chans=config.channels, 
            embed_dim=config.embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        self.encoder_blocks = nn.ModuleList([
            ViTBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ])
        
        # Bottleneck: MLP-Mixer + RMSNorm
        # Paper: "We insert 4-layer MLP-Mixers ... in the end of the encoder"
        # Since our output is (B, N, D), we can just flatten or use mixer on tokens
        # The paper says "flattens z into a vector of dimension L"
        # We need to map (N, D) -> (L) where L = 2048?
        # N = (32/2)^2 = 256 patches. D = 512. Total = 131072.
        # This is huge. Paper suggests L = 16x16x8 = 2048.
        # So we need a projection layer.
        
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        self.to_latent = nn.Linear(config.embed_dim * num_patches, config.latent_dim)
        
        # Decoder
        self.from_latent = nn.Linear(config.latent_dim, config.embed_dim * num_patches)
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Recon Block
        # Map tokens back to patches
        self.head = nn.Linear(config.embed_dim, config.patch_size**2 * config.channels)
        
        # Perceptual Loss
        self.perceptual_loss = VGGPerceptualLoss()

    def encode(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for blk in self.encoder_blocks:
            x = blk(x)
            
        x = self.encoder_norm(x)
        x = x.flatten(1)
        z = self.to_latent(x)
        
        # Sphere Normalization (RMS Norm)
        # v = z / RMS(z) -> z / sqrt(mean(z^2))
        # Wait, RMS norm usually keeps magnitude 1?
        # Paper: "projects it onto a sphere with radius sqrt(L) via RMS normalization"
        # v = f(z) = z / sqrt(mean(z^2))
        # ||v|| = sqrt(sum(v_i^2)) = sqrt(L * mean(v^2)) = sqrt(L)
        # Standard normalize p=2 gives norm 1.
        # If we use p=2, radius is 1. Standard RMS Norm gives radius sqrt(L).
        # Let's stick to unit sphere consistency with previous code if possible, 
        # or follow paper strictly. Paper says "radius sqrt(L)".
        # Let's use standard L2 normalize (radius 1) for simplicity with cosine similarity, 
        # unless radius matters for noise.
        # Eq 11: alpha depends on sigma_max.
        # If we use L2 norm (radius 1), then sigma_max is relative to 1.
        # If radius is sqrt(L), sigma_max scales.
        # Let's stick to unit sphere (radius 1) and use the tan(alpha) logic which is scale invariant.
        
        v = torch.nn.functional.normalize(z, p=2, dim=1)
        return v

    def decode(self, v):
        x = self.from_latent(v)
        x = x.view(-1, self.patch_embed.num_patches, self.config.embed_dim)
        
        # Add decoder pos embed? Usually symmetric.
        x = x + self.pos_embed # Reuse or new? Paper implies symmetry. Let's reuse for simplicity/param sharing or new.
        # Let's make new parameter for decoder
        if not hasattr(self, 'decoder_pos_embed'):
             self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.config.embed_dim)).to(v.device)
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
            
        x = self.decoder_norm(x)
        x = self.head(x)
        
        # Reshape to Image: (B, N, P*P*C) -> (B, C, H, W)
        B, N, C = x.shape
        H = W = self.config.image_size
        P = self.config.patch_size
        x = x.transpose(1, 2) # (B, C, N) -> (B, P*P*C, N)
        # (B, C, P, P, H/P, W/P)
        x = x.view(B, self.config.channels, P, P, H//P, W//P)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, self.config.channels, H, W)
        
        return torch.tanh(x) # Match [-1, 1] range

    def decode_multistep(self, v, steps=1):
        x = self.decode(v)
        for i in range(steps - 1):
            v_new = self.encode(x)
            x = self.decode(v_new)
        return x

    def forward(self, x, training=True):
        v = self.encode(x)
        return self.decode(v), v

    def compute_loss(self, x, v_noisy, v_noisy_sub, v_clean):
        recon_sub = self.decode(v_noisy_sub)
        l_pix_recon = F.l1_loss(recon_sub, x) + self.perceptual_loss(recon_sub, x)
        
        recon_large = self.decode(v_noisy)
        target_pix = recon_sub.detach() 
        l_pix_con = F.l1_loss(recon_large, target_pix) + self.perceptual_loss(recon_large, target_pix)
        
        v_rec = self.encode(recon_large)
        l_lat_con = 1.0 - F.cosine_similarity(v_clean, v_rec).mean()
        
        return l_pix_recon, l_pix_con, l_lat_con, recon_sub


class ConvSphereEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim
        
        # Encoder: 32x32 -> 1x1 (flattened)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(config.channels, 32, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.encoder_linear = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder: Latent -> 32x32
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, config.channels, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.Tanh() # Output ranges [-1, 1]
        )
        
        # Perceptual Loss
        self.perceptual_loss = VGGPerceptualLoss()

    def encode(self, x):
        features = self.encoder_conv(x)
        features = features.flatten(1)
        z = self.encoder_linear(features)
        v = torch.nn.functional.normalize(z, p=2, dim=1)
        # Fix: Scale to radius sqrt(L) to maintain variance ~1 and match noise magnitude
        v = v * math.sqrt(self.config.latent_dim)
        return v
        
    def decode(self, v):
        x = self.decoder_input(v)
        x = x.view(-1, 128, 4, 4)
        x = self.decoder_conv(x)
        return x

    def decode_multistep(self, v, steps=1):
        x = self.decode(v)
        for i in range(steps - 1):
            v_new = self.encode(x)
            x = self.decode(v_new)
        return x

    def forward(self, x, training=True):
        v = self.encode(x)
        return self.decode(v), v

    def compute_loss(self, x, v_noisy, v_noisy_sub, v_clean):
        # Same loss logic as ViT, ensuring compatibility with training loop
        recon_sub = self.decode(v_noisy_sub)
        l_pix_recon = F.l1_loss(recon_sub, x) + self.perceptual_loss(recon_sub, x)
        
        recon_large = self.decode(v_noisy)
        target_pix = recon_sub.detach() 
        l_pix_con = F.l1_loss(recon_large, target_pix) + self.perceptual_loss(recon_large, target_pix)
        
        v_rec = self.encode(recon_large)
        l_lat_con = 1.0 - F.cosine_similarity(v_clean, v_rec).mean()
        
        return l_pix_recon, l_pix_con, l_lat_con, recon_sub

class SphereEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_type == 'vit':
            print("Initializing ViT Sphere Encoder")
            self.model = ViTSphereEncoder(config)
        else:
            print("Initializing ConvNet Sphere Encoder")
            self.model = ConvSphereEncoder(config)
            
    def encode(self, x): return self.model.encode(x)
    def decode(self, v): return self.model.decode(v)
    def decode_multistep(self, v, steps=1): return self.model.decode_multistep(v, steps)
    def forward(self, x, training=True): return self.model(x, training)
    def compute_loss(self, x, vn, vns, vc): return self.model.compute_loss(x, vn, vns, vc)
