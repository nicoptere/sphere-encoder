import torch
import os
import argparse
from model import SphereEncoder, Config

def export_onnx(checkpoint_path, output_dir):
    device = torch.device('cpu')
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint if available, else use defaults
    config_dict = checkpoint.get('config', {})
    config = Config(**config_dict)
    config.device = 'cpu'
    
    model = SphereEncoder(config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export Encoder
    # Input: (B, C, H, W)
    dummy_input = torch.randn(1, config.channels, config.image_size, config.image_size)
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    
    # We want to export only the internal encoder to avoid messy logic in forward
    # but SphereEncoder wraps it. We can access the inner model.
    inner_model = model.model
    
    print("Exporting Encoder...")
    torch.onnx.export(
        inner_model, 
        dummy_input, 
        encoder_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['latent'],
        dynamic_axes={'input': {0: 'batch_size'}, 'latent': {0: 'batch_size'}},
        # We need to tell ONNX which method to call if it's not forward
        # Actually inner_model.encode is what we want. 
        # But torch.onnx.export usually calls forward.
    )
    
    # Alternative: Wrap encoder and decoder for clean export
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model.encode(x)

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, v):
            return self.model.decode(v)

    wrap_encoder = EncoderWrapper(inner_model)
    wrap_decoder = DecoderWrapper(inner_model)
    
    print("Exporting Encoder (Wrapped)...")
    torch.onnx.export(
        wrap_encoder, 
        dummy_input, 
        encoder_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['latent'],
        dynamic_axes={'input': {0: 'batch_size'}, 'latent': {0: 'batch_size'}}
    )

    print("Exporting Decoder...")
    dummy_latent = torch.randn(1, config.latent_dim)
    decoder_path = os.path.join(output_dir, "decoder.onnx")
    torch.onnx.export(
        wrap_decoder, 
        dummy_latent, 
        decoder_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['latent'],
        output_names=['output'],
        dynamic_axes={'latent': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Successfully exported to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=r'results\cifar10_20260219_182845\checkpoints\checkpoint_latest.pth', help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default=r'frontend\public\model', help='Output directory for ONNX models')
    args = parser.parse_args()
    
    export_onnx(args.checkpoint, args.output_dir)
