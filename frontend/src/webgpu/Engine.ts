
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

    // Pipelines
    private pipelines: Record<string, GPUComputePipeline> = {};

    constructor() { }

    async init(modelPath: string) {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter found");
        this.device = await adapter.requestDevice();

        const loader = new WeightLoader();
        await loader.load(modelPath);
        this.metadata = loader.getMetadata();

        // Create Pipelines
        this.pipelines['linear'] = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code: LINEAR_WGSL }), entryPoint: 'main' }
        });
        this.pipelines['conv_transpose'] = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code: CONV_TRANSPOSE_WGSL }), entryPoint: 'main' }
        });
        this.pipelines['conv2d'] = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code: CONV2D_WGSL }), entryPoint: 'main' }
        });
        this.pipelines['relu'] = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code: RELU_WGSL }), entryPoint: 'main' }
        });
        this.pipelines['leaky_relu'] = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code: LEAKY_RELU_WGSL }), entryPoint: 'main' }
        });
        this.pipelines['tanh'] = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code: TANH_WGSL }), entryPoint: 'main' }
        });

        // Initialize Encoder Buffers
        for (const layer of this.metadata.encoder_layers) {
            const { w, b } = this.createWeightBuffers(loader, layer);
            this.encoderWeights.push(w);
            this.encoderBiases.push(b);
        }

        // Initialize Decoder Buffers
        for (const layer of this.metadata.decoder_layers) {
            const { w, b } = this.createWeightBuffers(loader, layer);
            this.decoderWeights.push(w);
            this.decoderBiases.push(b);
        }
    }

    private createWeightBuffers(loader: WeightLoader, layer: LayerMeta) {
        if (!this.device) throw new Error("No device");
        const wData = loader.getWeightData(layer.weight_offset, layer.weight_size);
        const bData = loader.getWeightData(layer.bias_offset, layer.bias_size);

        const w = this.device.createBuffer({
            size: wData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(w.getMappedRange()).set(wData);
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
            } else {
                continue;
            }

            const outputByteSize = (layer.out_channels || layer.out_features!) * outSize * outSize * 4;
            const outputBuffer = this.device.createBuffer({
                size: outputByteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            // Params
            const paramData = new Uint32Array(8);
            if (layer.type === 'conv2d') {
                paramData[0] = layer.in_channels!;
                paramData[1] = layer.out_channels!;
                paramData[2] = currentSize;
                paramData[3] = outSize;
                paramData[4] = layer.kernel_size!;
                paramData[5] = layer.stride!;
                paramData[6] = layer.padding!;
            } else {
                paramData[0] = layer.in_features!;
                paramData[1] = layer.out_features!;
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
            if (layer.type === 'conv2d') {
                pass.dispatchWorkgroups(Math.ceil(outSize / 8), Math.ceil(outSize / 8), layer.out_channels!);
            } else {
                pass.dispatchWorkgroups(Math.ceil(layer.out_features! / 64));
            }
            pass.end();

            if (layer.activation === 'leaky_relu') {
                const actPipeline = this.pipelines['leaky_relu'];
                const actBindGroup = this.device.createBindGroup({
                    layout: actPipeline.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
                });
                const actPass = commandEncoder.beginComputePass();
                actPass.setPipeline(actPipeline);
                actPass.setBindGroup(0, actBindGroup);
                actPass.dispatchWorkgroups(Math.ceil((outputByteSize / 4) / 64));
                actPass.end();
            }

            currentInput = outputBuffer;
            currentSize = outSize;
        }

        const readBuffer = this.device.createBuffer({
            size: currentInput.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
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
        let currentSize = 4; // Model starts at 4x4 after Linear

        const inputBuffer = this.device.createBuffer({
            size: latent.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(inputBuffer, 0, latent);

        let currentInput = inputBuffer;

        for (let i = 0; i < this.metadata.decoder_layers.length; i++) {
            const layer = this.metadata.decoder_layers[i];
            let outSize = currentSize;
            let pipeline: GPUComputePipeline;

            if (layer.type === 'linear') {
                outSize = 4; // Reshaped to (C, 4, 4)
                pipeline = this.pipelines['linear'];
            } else if (layer.type === 'conv_transpose') {
                outSize = currentSize * 2;
                pipeline = this.pipelines['conv_transpose'];
            } else {
                continue;
            }

            const outputByteSize = (layer.out_channels || layer.out_features!) * outSize * outSize * 4;
            const outputBuffer = this.device.createBuffer({
                size: outputByteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            const paramData = new Uint32Array(8);
            if (layer.type === 'linear') {
                paramData[0] = layer.in_features!;
                paramData[1] = layer.out_features!;
            } else {
                paramData[0] = layer.in_channels!;
                paramData[1] = layer.out_channels!;
                paramData[2] = currentSize;
                paramData[3] = outSize;
                paramData[4] = layer.kernel_size!;
                paramData[5] = layer.stride!;
                paramData[6] = layer.padding!;
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
                    { binding: 2, resource: { buffer: this.decoderWeights[i] } },
                    { binding: 3, resource: { buffer: this.decoderBiases[i] } },
                    { binding: 4, resource: { buffer: outputBuffer } }
                ]
            });

            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            if (layer.type === 'linear') {
                pass.dispatchWorkgroups(Math.ceil(layer.out_features! / 64));
            } else {
                pass.dispatchWorkgroups(Math.ceil(outSize / 8), Math.ceil(outSize / 8), layer.out_channels!);
            }
            pass.end();

            if (layer.activation === 'relu' || layer.activation === 'tanh') {
                const actPipeline = this.pipelines[layer.activation];
                const actBindGroup = this.device.createBindGroup({
                    layout: actPipeline.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: { buffer: outputBuffer } }]
                });
                const actPass = commandEncoder.beginComputePass();
                actPass.setPipeline(actPipeline);
                actPass.setBindGroup(0, actBindGroup);
                actPass.dispatchWorkgroups(Math.ceil((outputByteSize / 4) / 64));
                actPass.end();
            }

            currentInput = outputBuffer;
            currentSize = outSize;
        }

        const readBuffer = this.device.createBuffer({
            size: currentInput.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        commandEncoder.copyBufferToBuffer(currentInput, 0, readBuffer, 0, currentInput.size);
        this.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        return result;
    }

    getMetadata() { return this.metadata; }
}
