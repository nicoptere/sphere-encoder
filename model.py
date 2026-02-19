
import torch
import torch.nn as nn
import math

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Defaults (will be overwritten by args)
    image_size = 32
    channels = 3
    dataset_path = './data'
    
    # Training
    batch_size = 128
    learning_rate = 1e-3
    latent_dim = 128
    epochs = 100
    iterations_per_epoch = 100
    
    # Model
    sigma_max = 0.2
    
    # Paths (will be overwritten)
    checkpoint_dir = './checkpoints'
    results_dir = './results'
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class SphereEncoder(nn.Module):
    def __init__(self, config):
        super(SphereEncoder, self).__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        # Calculate number of downsampling layers needed
        # We want to go from image_size -> 1x1 or 4x4?
        # Let's say we target ~4x4 at the bottleneck before flattening.
        # image_size / 2^layers = 4  =>  2^layers = image_size / 4
        # layers = log2(image_size) - 2
        
        # Example: 32 -> 32/4 = 8. log2(8)=3 layers? 
        # Actually 32 -> 16 -> 8 -> 4 (3 layers stride 2). 
        # Previous 32x32 model used 4 layers: 32->16->8->4->Flatten->Linear.
        
        # Consistent logic:
        # Size | Layers | Feature Map Size History
        # 32   | 3      | 32->16->8->4
        # 64   | 4      | 64->32->16->8->4
        # 128  | 5      | ...->4
        # 256  | 6      | ...->4
        
        num_layers = int(math.log2(config.image_size) - 2)
        if num_layers < 2: num_layers = 2 # Minimum 2 layers
        
        # Dynamic Encoder Construction
        layers = []
        in_channels = config.channels # 3
        # Initial filters? 
        # 32 -> 64 -> 128 -> 256 -> 512 ...
        current_filters = 32
        
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, current_filters, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(current_filters))
            layers.append(nn.LeakyReLU(0.2))
            
            in_channels = current_filters
            if current_filters < 512: # Cap filters at 512
                current_filters *= 2
                
        self.encoder_conv = nn.Sequential(*layers)
        
        # Calculate feature map size
        # If we targeted 4x4, it should be 4x4.
        self.feature_map_size = 4 
        self.final_conv_channels = in_channels
        
        self.encoder_linear = nn.Linear(self.final_conv_channels * self.feature_map_size * self.feature_map_size, self.latent_dim)
        
        # Decoder Construction
        self.decoder_input = nn.Linear(self.latent_dim, self.final_conv_channels * self.feature_map_size * self.feature_map_size)
        
        decoder_layers = []
        
        # We need to reverse the channel progression
        # e.g., for 32x32: 128 -> 64 -> 32 -> 3
        
        # We constructed encoder: 3->32->64->128. 
        # Decoder should go: 128->64->32->3. 
        
        # Reconstruct channel list used in encoder
        enc_channels = []
        c = 32
        for i in range(num_layers):
            enc_channels.append(c)
            if c < 512: c *= 2
        
        # enc_channels for 32x32 (3 layers): [32, 64, 128]
        # Decoder input channels: 128
        
        reversed_channels = enc_channels[::-1] # [128, 64, 32]
        
        current_in = self.final_conv_channels # 128
        
        for i, out_channels in enumerate(reversed_channels[1:]): # Loop [64, 32]
            decoder_layers.append(nn.ConvTranspose2d(current_in, out_channels, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            decoder_layers.append(nn.ReLU())
            current_in = out_channels
            
        # Final layer to image channels
        # Last channel in reversed was 32. current_in is 32.
        # We need to map 32 -> 3.
        # But wait, we missed one step? 
        # Encoder: 3 -> 32 (stride 2) -> 64 (stride 2) -> 128 (stride 2). 3 layers.
        # Decoder: 128 -> 64 (stride 2, up) -> 32 (stride 2, up) -> 3 (stride 2, up).
        
        # My loop above: `reversed_channels` = [128, 64, 32]. 
        # `reversed_channels[1:]` = [64, 32].
        # Iter 1: 128 -> 64.
        # Iter 2: 64 -> 32.
        # After loop: current_in = 32.
        # Final layer below: 32 -> 3. Correct.
        
        decoder_layers.append(nn.ConvTranspose2d(current_in, config.channels, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Tanh())
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        features = self.encoder_conv(x)
        features = features.flatten(1)
        z = self.encoder_linear(features)
        v = torch.nn.functional.normalize(z, p=2, dim=1)
        return v
        
    def decode(self, v):
        x = self.decoder_input(v)
        x = x.view(-1, self.final_conv_channels, self.feature_map_size, self.feature_map_size)
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
        if training:
            sigma = torch.rand(v.shape[0], 1, device=v.device) * self.config.sigma_max
            e = torch.randn_like(v)
            v_reprojected = torch.nn.functional.normalize(v + sigma * e, p=2, dim=1)
            return self.decode(v_reprojected), v, v_reprojected
        else:
            return self.decode(v)
