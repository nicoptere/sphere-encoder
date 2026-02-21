
// WebGPU Kernels for Sphere Encoder with Dequantization Support

export const DEQUANT_HELPERS = `
fn get_weight_f32(idx: u32, scale: f32, mode: u32) -> f32 {
    // mode: 0=f32, 1=f16, 2=q8, 3=q4
    if (mode == 0u) {
        return bitcast<f32>(weights_u32[idx]);
    } else if (mode == 1u) {
        // f16: weights_u32 is array<u32>, each contains two f16
        // actually give me vec2<f32>(f16_low, f16_high)
        let word = weights_u32[idx / 2u];
        let unpacked = unpack2x16float(word);
        return select(unpacked.x, unpacked.y, idx % 2u == 1u);
    } else if (mode == 2u) {
        // q8: weights_u32 is array<u32>, each contains 4 int8
        let word = weights_u32[idx / 4u];
        let byte_idx = idx % 4u;
        let val_u8 = (word >> (byte_idx * 8u)) & 0xFFu;
        let val_i8 = select(i32(val_u8), i32(val_u8) - 256, val_u8 >= 128u);
        return f32(val_i8) * scale;
    } else if (mode == 3u) {
        // q4: weights_u32 is array<u32>, each contains 8 nibbles
        let byte_idx = idx / 2u;
        let word = weights_u32[byte_idx / 4u];
        let b = (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
        let nibble = select(b & 0xFu, b >> 4u, idx % 2u == 1u);
        return (f32(nibble) - 8.0) * scale;
    }
    return 0.0;
}
`;

export const LINEAR_WGSL = `
struct Params {
    in_features: u32,
    out_features: u32,
    weight_scale: f32,
    weight_mode: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights_u32: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

${DEQUANT_HELPERS}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.out_features) { return; }

    var sum: f32 = bias[row];
    for (var i: u32 = 0u; i < params.in_features; i = i + 1u) {
        let w = get_weight_f32(row * params.in_features + i, params.weight_scale, params.weight_mode);
        sum = sum + input[i] * w;
    }
    output[row] = sum;
}
`;

export const CONV_TRANSPOSE_WGSL = `
struct Params {
    in_channels: u32,
    out_channels: u32,
    in_size: u32,
    out_size: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    weight_scale: f32,
    weight_mode: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights_u32: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

${DEQUANT_HELPERS}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let out_c = global_id.z;

    if (out_x >= params.out_size || out_y >= params.out_size || out_c >= params.out_channels) { return; }

    var sum: f32 = bias[out_c];
    let k_size = params.kernel_size;
    let stride = params.stride;
    let padding = params.padding;
    let in_size = params.in_size;

    for (var in_c: u32 = 0u; in_c < params.in_channels; in_c = in_c + 1u) {
        for (var ky: u32 = 0u; ky < k_size; ky = ky + 1u) {
            let py = out_y + padding;
            if (py < ky) { continue; }
            let tmp_y = py - ky;
            if (tmp_y % stride != 0u) { continue; }
            let in_y = tmp_y / stride;
            if (in_y >= in_size) { continue; }

            for (var kx: u32 = 0u; kx < k_size; kx = kx + 1u) {
                let px = out_x + padding;
                if (px < kx) { continue; }
                let tmp_x = px - kx;
                if (tmp_x % stride != 0u) { continue; }
                let in_x = tmp_x / stride;
                if (in_x >= in_size) { continue; }

                let input_idx = (in_c * in_size + in_y) * in_size + in_x;
                let weight_idx = (((in_c * params.out_channels + out_c) * k_size + ky) * k_size) + kx;
                
                let w = get_weight_f32(weight_idx, params.weight_scale, params.weight_mode);
                sum = sum + input[input_idx] * w;
            }
        }
    }

    let out_idx = (out_c * params.out_size + out_y) * params.out_size + out_x;
    output[out_idx] = sum;
}
`;

export const CONV2D_WGSL = `
struct Params {
    in_channels: u32,
    out_channels: u32,
    in_size: u32,
    out_size: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    weight_scale: f32,
    weight_mode: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights_u32: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

${DEQUANT_HELPERS}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let out_c = global_id.z;

    if (out_x >= params.out_size || out_y >= params.out_size || out_c >= params.out_channels) { return; }

    var sum: f32 = bias[out_c];
    let k_size = params.kernel_size;
    let stride = params.stride;
    let padding = params.padding;
    let in_size = params.in_size;

    for (var in_c: u32 = 0u; in_c < params.in_channels; in_c = in_c + 1u) {
        for (var ky: u32 = 0u; ky < k_size; ky = ky + 1u) {
            let in_y = out_y * stride + ky;
            if (in_y < padding || in_y >= in_size + padding) { continue; }
            let real_in_y = in_y - padding;

            for (var kx: u32 = 0u; kx < k_size; kx = kx + 1u) {
                let in_x = out_x * stride + kx;
                if (in_x < padding || in_x >= in_size + padding) { continue; }
                let real_in_x = in_x - padding;

                let input_idx = (in_c * in_size + real_in_y) * in_size + real_in_x;
                let weight_idx = (((out_c * params.in_channels + in_c) * k_size + ky) * k_size) + kx;
                
                let w = get_weight_f32(weight_idx, params.weight_scale, params.weight_mode);
                sum = sum + input[input_idx] * w;
            }
        }
    }

    let out_idx = (out_c * params.out_size + out_y) * params.out_size + out_x;
    output[out_idx] = sum;
}
`;

export const LEAKY_RELU_WGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&data)) {
        let x = data[idx];
        data[idx] = select(0.2 * x, x, x > 0.0);
    }
}
`;

export const RELU_WGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&data)) {
        data[idx] = max(0.0, data[idx]);
    }
}
`;

export const TANH_WGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&data)) {
        data[idx] = tanh(data[idx]);
    }
}
`;
