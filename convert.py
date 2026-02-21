
import torch
import os
import argparse
import json
from model import SphereEncoder, Config

def export_onnx(checkpoint_path, output_root):
    device = torch.device('cpu')
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint if available, else use defaults
    config_dict = checkpoint.get('config', {})
    config = Config(**config_dict)
    config.device = 'cpu'
    
    # Infer dataset name from checkpoint directory or filename
    # results/ffhq_64_TIMESTAMP/checkpoints/checkpoint_latest.pth -> ffhq_64
    parent_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    run_name = os.path.basename(parent_dir)
    # Split by timestamp (usually YYYYMMDD_HHMMSS)
    dataset_name = run_name.split('_202')[0] if '_202' in run_name else run_name
    
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    model = SphereEncoder(config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 1. Export Metadata
    metadata = {
        "dataset": dataset_name,
        "latent_dim": config.latent_dim,
        "image_size": config.image_size,
        "channels": config.channels,
        "model_type": config.model_type
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {os.path.join(output_dir, 'metadata.json')}")

    # 2. Export Encoder
    dummy_input = torch.randn(1, config.channels, config.image_size, config.image_size)
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    
    inner_model = model.model
    
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
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default=r'frontend\public\model', help='Root output directory')
    args = parser.parse_args()
    
    export_onnx(args.checkpoint, args.output_dir)
