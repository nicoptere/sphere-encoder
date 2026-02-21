
import torch
import torch.nn as nn
import json
import os
import argparse
import numpy as np
from model import SphereEncoder, Config

def fold_batchnorm_conv_transpose(conv, bn):
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels).to(w.device)
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight.data
    beta = bn.bias.data
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    w_folded = w * scale.view(1, -1, 1, 1)
    b_folded = (b - mean) * scale + beta
    return w_folded, b_folded

def fold_batchnorm_conv2d(conv, bn):
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels).to(w.device)
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight.data
    beta = bn.bias.data
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    w_folded = w * scale.view(-1, 1, 1, 1)
    b_folded = (b - mean) * scale + beta
    return w_folded, b_folded

def quantize_linear(data, mode):
    """
    Quantizes data to f32, f16, q8, or q4.
    Returns (quantized_bytes, scale, data_type)
    """
    if mode == "f32":
        return data.astype(np.float32).tobytes(), 1.0, "f32"
    
    if mode == "f16":
        return data.astype(np.float16).tobytes(), 1.0, "f16"
    
    if mode == "q8":
        # Symmetric quantization: scale = max(abs(data)) / 127
        max_val = np.max(np.abs(data))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0
        
        q_data = np.round(data / scale).clip(-128, 127).astype(np.int8)
        return q_data.tobytes(), float(scale), "q8"
    
    if mode == "q4":
        # 4-bit symmetric quantization: scale = max(abs(data)) / 7
        max_val = np.max(np.abs(data))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 7.0
        
        q_data = np.round(data / scale).clip(-8, 7).astype(np.int8)
        # Pack into uint8: low 4 bits = weight 0, high 4 bits = weight 1
        q_data = q_data.flatten()
        if q_data.size % 2 != 0:
            q_data = np.pad(q_data, (0, 1), 'constant')
        
        # Shift values to 0-15 range for packing
        packed = (q_data[0::2] + 8) | ((q_data[1::2] + 8) << 4)
        return packed.astype(np.uint8).tobytes(), float(scale), "q4"
    
    return data.astype(np.float32).tobytes(), 1.0, "f32"

def export_model(checkpoint_path, output_dir, quant_mode="f32"):
    print(f"Loading checkpoint: {checkpoint_path} (Mode: {quant_mode})")
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
    
    def process_layer(layer_name, layer_type, w_torch, b_torch, meta_list, **kwargs):
        w_np = w_torch.numpy()
        b_np = b_torch.numpy().astype(np.float32) # Biases usually stay f32 or f16
        
        # Quantize weights
        q_bytes, scale, dtype = quantize_linear(w_np, quant_mode)
        
        meta = {
            "name": layer_name,
            "type": layer_type,
            "weight_offset": len(weights_binary),
            "weight_size": len(q_bytes),
            "weight_dtype": dtype,
            "weight_scale": scale,
            "bias_offset": len(weights_binary) + len(q_bytes),
            "bias_size": b_np.nbytes,
            **kwargs
        }
        meta_list.append(meta)
        weights_binary.extend(q_bytes)
        weights_binary.extend(b_np.tobytes())

    # --- ENCODER ---
    encoder_meta = []
    enc_seq = full_model.encoder_conv
    i = 0
    while i < len(enc_seq):
        layer = enc_seq[i]
        if isinstance(layer, nn.Conv2d):
            conv = layer
            next_layer = enc_seq[i+1] if i+1 < len(enc_seq) else None
            w, b = conv.weight.data, (conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels))
            if isinstance(next_layer, nn.BatchNorm2d):
                w, b = fold_batchnorm_conv2d(conv, next_layer)
                i += 2 
            else: i += 1
            
            process_layer(f"enc_conv_{len(encoder_meta)}", "conv2d", w, b, encoder_meta,
                          in_channels=w.shape[1], out_channels=w.shape[0], kernel_size=w.shape[2], stride=2, padding=1, activation="leaky_relu")
        else: i += 1

    process_layer("encoder_linear", "linear", full_model.encoder_linear.weight.data, full_model.encoder_linear.bias.data, encoder_meta,
                  in_features=full_model.encoder_linear.in_features, out_features=full_model.encoder_linear.out_features)

    # --- DECODER ---
    decoder_meta = []
    process_layer("linear_input", "linear", full_model.decoder_input.weight.data, full_model.decoder_input.bias.data, decoder_meta,
                  in_features=full_model.decoder_input.in_features, out_features=full_model.decoder_input.out_features)
    
    conv_seq = full_model.decoder_conv
    i = 0
    while i < len(conv_seq):
        layer = conv_seq[i]
        if isinstance(layer, nn.ConvTranspose2d):
            conv = layer
            next_layer = conv_seq[i+1] if i+1 < len(conv_seq) else None
            w, b = conv.weight.data, (conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels))
            if isinstance(next_layer, nn.BatchNorm2d):
                w, b = fold_batchnorm_conv_transpose(conv, next_layer)
                i += 2 
            else: i += 1
            
            process_layer(f"dec_conv_{len(decoder_meta)}", "conv_transpose", w, b, decoder_meta,
                          in_channels=w.shape[0], out_channels=w.shape[1], kernel_size=w.shape[2], stride=2, padding=1, activation="relu")
        else: i += 1
    decoder_meta[-1]["activation"] = "tanh"

    # Save
    suffix = f"_{quant_mode}" if quant_mode != "f32" else ""
    with open(os.path.join(output_dir, f"model_weights{suffix}.bin"), "wb") as f:
        f.write(weights_binary)
        
    metadata = {
        "image_size": config.image_size,
        "latent_dim": config.latent_dim,
        "final_channels": full_model.final_channels,
        "quant_mode": quant_mode,
        "encoder_layers": encoder_meta,
        "decoder_layers": decoder_meta
    }
    with open(os.path.join(output_dir, f"model_meta{suffix}.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Export complete: {quant_mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="f32", choices=["f32", "f16", "q8", "q4"])
    args = parser.parse_args()
    export_model(args.checkpoint, args.output_dir, args.mode)
