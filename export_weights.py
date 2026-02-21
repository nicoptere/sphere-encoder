
import torch
import torch.nn as nn
import json
import os
import argparse
import numpy as np
from model import SphereEncoder, Config

def fold_batchnorm_conv_transpose(conv, bn):
    """
    Folds BatchNorm parameters into ConvTranspose2d weight and bias.
    """
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels).to(w.device)
    
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight.data
    beta = bn.bias.data
    eps = bn.eps
    
    scale = gamma / torch.sqrt(var + eps)
    
    # Weight shape: (In, Out, K, K)
    w_folded = w * scale.view(1, -1, 1, 1)
    b_folded = (b - mean) * scale + beta
    
    return w_folded, b_folded

def fold_batchnorm_conv2d(conv, bn):
    """
    Folds BatchNorm parameters into Conv2d weight and bias.
    """
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels).to(w.device)
    
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight.data
    beta = bn.bias.data
    eps = bn.eps
    
    scale = gamma / torch.sqrt(var + eps)
    
    # Weight shape: (Out, In, K, K)
    w_folded = w * scale.view(-1, 1, 1, 1)
    b_folded = (b - mean) * scale + beta
    
    return w_folded, b_folded

def export_model(checkpoint_path, output_dir):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint['state_dict']
    ckpt_config = checkpoint.get('config', {})
    
    latent_dim = ckpt_config.get('latent_dim', 512)
    image_size = ckpt_config.get('image_size', 64)
    model_type = ckpt_config.get('model_type', 'convnet')
    
    config = Config(image_size=image_size, latent_dim=latent_dim, model_type=model_type)
    model = SphereEncoder(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    full_model = model.model
    os.makedirs(output_dir, exist_ok=True)
    
    weights_binary = bytearray()
    
    # --- ENCODER ---
    encoder_meta = []
    enc_seq = full_model.encoder_conv
    i = 0
    while i < len(enc_seq):
        layer = enc_seq[i]
        if isinstance(layer, nn.Conv2d):
            conv = layer
            next_layer = enc_seq[i+1] if i+1 < len(enc_seq) else None
            
            w = conv.weight.data
            b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
            
            if isinstance(next_layer, nn.BatchNorm2d):
                print(f"Folding BN into {conv}")
                w, b = fold_batchnorm_conv2d(conv, next_layer)
                i += 2 
            else:
                i += 1
                
            w_np = w.numpy() # (Out, In, K, K)
            b_np = b.numpy()
            
            encoder_meta.append({
                "name": f"encoder_conv_{len(encoder_meta)}",
                "type": "conv2d",
                "in_channels": w_np.shape[1],
                "out_channels": w_np.shape[0],
                "kernel_size": w_np.shape[2],
                "stride": 2,
                "padding": 1,
                "weight_offset": len(weights_binary),
                "weight_size": w_np.nbytes,
                "bias_offset": len(weights_binary) + w_np.nbytes,
                "bias_size": b_np.nbytes,
                "activation": "leaky_relu"
            })
            weights_binary.extend(w_np.tobytes())
            weights_binary.extend(b_np.tobytes())
        else:
            i += 1

    # Encoder Linear
    enc_lin_w = full_model.encoder_linear.weight.data.numpy() # (Latent, Flat)
    enc_lin_b = full_model.encoder_linear.bias.data.numpy()
    encoder_meta.append({
        "name": "encoder_linear",
        "type": "linear",
        "in_features": enc_lin_w.shape[1],
        "out_features": enc_lin_w.shape[0],
        "weight_offset": len(weights_binary),
        "weight_size": enc_lin_w.nbytes,
        "bias_offset": len(weights_binary) + enc_lin_w.nbytes,
        "bias_size": enc_lin_b.nbytes
    })
    weights_binary.extend(enc_lin_w.tobytes())
    weights_binary.extend(enc_lin_b.tobytes())

    # --- DECODER ---
    decoder_meta = []
    # 1. Decoder Input (Linear)
    lin_w = full_model.decoder_input.weight.data.numpy() # (Flat, Latent)
    lin_b = full_model.decoder_input.bias.data.numpy()
    
    decoder_meta.append({
        "name": "linear_input",
        "type": "linear",
        "in_features": lin_w.shape[1],
        "out_features": lin_w.shape[0],
        "weight_offset": len(weights_binary),
        "weight_size": lin_w.nbytes,
        "bias_offset": len(weights_binary) + lin_w.nbytes,
        "bias_size": lin_b.nbytes
    })
    weights_binary.extend(lin_w.tobytes())
    weights_binary.extend(lin_b.tobytes())
    
    # 2. ConvTranspose layers
    conv_seq = full_model.decoder_conv
    i = 0
    while i < len(conv_seq):
        layer = conv_seq[i]
        if isinstance(layer, nn.ConvTranspose2d):
            conv = layer
            next_layer = conv_seq[i+1] if i+1 < len(conv_seq) else None
            w = conv.weight.data
            b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
            
            if isinstance(next_layer, nn.BatchNorm2d):
                print(f"Folding BN into {conv}")
                w, b = fold_batchnorm_conv_transpose(conv, next_layer)
                i += 2 
            else:
                i += 1
                
            w_np = w.numpy() # (In, Out, K, K)
            b_np = b.numpy()
            
            decoder_meta.append({
                "name": f"conv_transpose_{len(decoder_meta)}",
                "type": "conv_transpose",
                "in_channels": w_np.shape[0],
                "out_channels": w_np.shape[1],
                "kernel_size": w_np.shape[2],
                "stride": 2,
                "padding": 1,
                "weight_offset": len(weights_binary),
                "weight_size": w_np.nbytes,
                "bias_offset": len(weights_binary) + w_np.nbytes,
                "bias_size": b_np.nbytes,
                "activation": "relu"
            })
            weights_binary.extend(w_np.tobytes())
            weights_binary.extend(b_np.tobytes())
        else:
            i += 1

    # Patch activation for the last decoder layer
    decoder_meta[-1]["activation"] = "tanh"

    # Save
    with open(os.path.join(output_dir, "model_weights.bin"), "wb") as f:
        f.write(weights_binary)
        
    metadata = {
        "image_size": config.image_size,
        "latent_dim": config.latent_dim,
        "final_channels": full_model.final_channels,
        "encoder_layers": encoder_meta,
        "decoder_layers": decoder_meta
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Export complete. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    export_model(args.checkpoint, args.output_dir)
