
import {
    LINEAR_WGSL, CONV_TRANSPOSE_WGSL, RELU_WGSL, TANH_WGSL,
    CONV2D_WGSL, LEAKY_RELU_WGSL
} from './Kernels';
import { WeightLoader, ModelMetadata, LayerMeta } from './WeightLoader';

export class WebGPUEngine {
    private device: GPUDevice | null = null;
    private metadata: ModelMetadata | null = null;

    private encoderWeights: GPUBuffer[] = [];
    private encoderBiases: GPUBuffer[] = [];
    private decoderWeights: GPUBuffer[] = [];
    private decoderBiases: GPUBuffer[] = [];

    private pipelines: Record<string, GPUComputePipeline> = {};

    constructor() { }

    async init(modelPath: string, quantMode: string = "f32") {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter found");
        this.device = await adapter.requestDevice();

        const loader = new WeightLoader();
        await loader.load(modelPath, quantMode);
        this.metadata = loader.getMetadata();

        // Create Pipelines
        const createPipeline = (code: string) => this.device!.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device!.createShaderModule({ code }), entryPoint: 'main' }
        });

        this.pipelines['linear'] = createPipeline(LINEAR_WGSL);
        this.pipelines['conv_transpose'] = createPipeline(CONV_TRANSPOSE_WGSL);
        this.pipelines['conv2d'] = createPipeline(CONV2D_WGSL);
        this.pipelines['relu'] = createPipeline(RELU_WGSL);
        this.pipelines['leaky_relu'] = createPipeline(LEAKY_RELU_WGSL);
        this.pipelines['tanh'] = createPipeline(TANH_WGSL);

        // Clear existing buffers
        this.encoderWeights = []; this.encoderBiases = [];
        this.decoderWeights = []; this.decoderBiases = [];

        for (const layer of this.metadata.encoder_layers) {
            const { w, b } = this.createBuffers(loader, layer);
            this.encoderWeights.push(w);
            this.encoderBiases.push(b);
        }

        for (const layer of this.metadata.decoder_layers) {
            const { w, b } = this.createBuffers(loader, layer);
            this.decoderWeights.push(w);
            this.decoderBiases.push(b);
        }
    }

    private createBuffers(loader: WeightLoader, layer: LayerMeta) {
        if (!this.device) throw new Error("No device");

        const wData = loader.getWeightData(layer.weight_offset, layer.weight_size);
        const bData = loader.getBiasData(layer.bias_offset, layer.bias_size);

        // Pad weight buffer to multiple of 4 bytes for storage/u32 access
        const paddedWSize = (wData.byteLength + 3) & ~3;
        const w = this.device.createBuffer({
            size: paddedWSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        const wMapped = w.getMappedRange();
        new Uint8Array(wMapped).set(new Uint8Array(wData));
        w.unmap();

        const b = this.device.createBuffer({
            size: bData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(b.getMappedRange()).set(bData);
        b.unmap();

        return { w, b };
    }

    private getModeEnum(mode: string): number {
        switch (mode) {
            case "f32": return 0;
            case "f16": return 1;
            case "q8": return 2;
            case "q4": return 3;
            default: return 0;
        }
    }

    async encode(pixels: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.metadata) throw new Error("Engine not initialized");

        const commandEncoder = this.device.createCommandEncoder();
        let currentSize = this.metadata.image_size;

        const inputBuffer = this.device.createBuffer({
            size: pixels.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(inputBuffer, 0, pixels);
        let currentInput = inputBuffer;

        const modeEnum = this.getModeEnum(this.metadata.quant_mode);

        for (let i = 0; i < this.metadata.encoder_layers.length; i++) {
            const layer = this.metadata.encoder_layers[i];
            let outSize = currentSize;
            let pipeline: GPUComputePipeline;

            if (layer.type === 'conv2d') {
                outSize = currentSize / 2;
                pipeline = this.pipelines['conv2d'];
            } else if (layer.type === 'linear') {
                outSize = 1;
                pipeline = this.pipelines['linear'];
            } else continue;

            const outCount = (layer.out_channels || layer.out_features!) * outSize * outSize;
            const outputBuffer = this.device.createBuffer({
                size: outCount * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            // Params: in, out, inSize, outSize, k, s, p, scale, mode (9 values)
            // But uniforms usually want alignment. 10 floats/uints is fine.
            const paramData = new ArrayBuffer(40);
            const u32 = new Uint32Array(paramData);
            const f32 = new Float32Array(paramData);

            if (layer.type === 'conv2d') {
                u32[0] = layer.in_channels!;
                u32[1] = layer.out_channels!;
                u32[2] = currentSize;
                u32[3] = outSize;
                u32[4] = layer.kernel_size!;
                u32[5] = layer.stride!;
                u32[6] = layer.padding!;
                f32[7] = layer.weight_scale;
                u32[8] = modeEnum;
            } else {
                u32[0] = layer.in_features!;
                u32[1] = layer.out_features!;
                f32[2] = layer.weight_scale;
                u32[3] = modeEnum;
            }

            const paramBuffer = this.device.createBuffer({
                size: paramData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(paramBuffer, 0, paramData);

            const bindGroup = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: paramBuffer } },
                    { binding: 1, resource: { buffer: currentInput } },
                    { binding: 2, resource: { buffer: this.encoderWeights[i] } },
                    { binding: 3, resource: { buffer: this.encoderBiases[i] } },
                    { binding: 4, resource: { buffer: outputBuffer } }
                ]
            });

            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            if (layer.type === 'conv2d') pass.dispatchWorkgroups(Math.ceil(outSize / 8), Math.ceil(outSize / 8), layer.out_channels!);
            else pass.dispatchWorkgroups(Math.ceil(layer.out_features! / 64));
            pass.end();

            if (layer.activation === 'leaky_relu') {
                const actPass = commandEncoder.beginComputePass();
                actPass.setPipeline(this.pipelines['leaky_relu']);
                actPass.setBindGroup(0, this.device.createBindGroup({
                    layout: this.pipelines['leaky_relu'].getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
                }));
                actPass.dispatchWorkgroups(Math.ceil(outCount / 64));
                actPass.end();
            }

            currentInput = outputBuffer;
            currentSize = outSize;
        }

        const readBuffer = this.device.createBuffer({ size: currentInput.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        commandEncoder.copyBufferToBuffer(currentInput, 0, readBuffer, 0, currentInput.size);
        this.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        return result;
    }

    async decode(latent: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.metadata) throw new Error("Engine not initialized");

        const commandEncoder = this.device.createCommandEncoder();
        let currentSize = 4;

        const inputBuffer = this.device.createBuffer({ size: latent.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(inputBuffer, 0, latent);
        let currentInput = inputBuffer;

        const modeEnum = this.getModeEnum(this.metadata.quant_mode);

        for (let i = 0; i < this.metadata.decoder_layers.length; i++) {
            const layer = this.metadata.decoder_layers[i];
            let outSize = currentSize;
            let pipeline: GPUComputePipeline;

            if (layer.type === 'linear') {
                outSize = 4;
                pipeline = this.pipelines['linear'];
            } else if (layer.type === 'conv_transpose') {
                outSize = currentSize * 2;
                pipeline = this.pipelines['conv_transpose'];
            } else continue;

            const outCount = (layer.out_channels || layer.out_features!) * outSize * outSize;
            const outputBuffer = this.device.createBuffer({
                size: outCount * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            const paramData = new ArrayBuffer(40);
            const u32 = new Uint32Array(paramData);
            const f32 = new Float32Array(paramData);

            if (layer.type === 'linear') {
                u32[0] = layer.in_features!;
                u32[1] = layer.out_features!;
                f32[2] = layer.weight_scale;
                u32[3] = modeEnum;
            } else {
                u32[0] = layer.in_channels!;
                u32[1] = layer.out_channels!;
                u32[2] = currentSize;
                u32[3] = outSize;
                u32[4] = layer.kernel_size!;
                u32[5] = layer.stride!;
                u32[6] = layer.padding!;
                f32[7] = layer.weight_scale;
                u32[8] = modeEnum;
            }

            const paramBuffer = this.device.createBuffer({ size: paramData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this.device.queue.writeBuffer(paramBuffer, 0, paramData);

            const bindGroup = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: paramBuffer } },
                    { binding: 1, resource: { buffer: currentInput } },
                    { binding: 2, resource: { buffer: this.decoderWeights[i] } },
                    { binding: 3, resource: { buffer: this.decoderBiases[i] } },
                    { binding: 4, resource: { buffer: outputBuffer } }
                ]
            });

            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            if (layer.type === 'linear') pass.dispatchWorkgroups(Math.ceil(layer.out_features! / 64));
            else pass.dispatchWorkgroups(Math.ceil(outSize / 8), Math.ceil(outSize / 8), layer.out_channels!);
            pass.end();

            if (layer.activation === 'relu' || layer.activation === 'tanh') {
                const actPass = commandEncoder.beginComputePass();
                actPass.setPipeline(this.pipelines[layer.activation]);
                actPass.setBindGroup(0, this.device.createBindGroup({
                    layout: this.pipelines[layer.activation].getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
                }));
                actPass.dispatchWorkgroups(Math.ceil(outCount / 64));
                actPass.end();
            }

            currentInput = outputBuffer;
            currentSize = outSize;
        }

        const readBuffer = this.device.createBuffer({ size: currentInput.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        commandEncoder.copyBufferToBuffer(currentInput, 0, readBuffer, 0, currentInput.size);
        this.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        return result;
    }

    getMetadata() { return this.metadata; }
}
