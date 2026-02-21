
export interface LayerMeta {
    name: string;
    type: string;
    in_features?: number;
    out_features?: number;
    in_channels?: number;
    out_channels?: number;
    kernel_size?: number;
    stride?: number;
    padding?: number;
    weight_offset: number;
    weight_size: number;
    weight_dtype: string;
    weight_scale: number;
    bias_offset: number;
    bias_size: number;
    activation: string;
}

export interface ModelMetadata {
    image_size: number;
    latent_dim: number;
    final_channels: number;
    quant_mode: string;
    encoder_layers: LayerMeta[];
    decoder_layers: LayerMeta[];
}

export class WeightLoader {
    private binaryData: ArrayBuffer | null = null;
    private metadata: ModelMetadata | null = null;

    async load(modelPath: string, quantMode: string = "f32") {
        const suffix = quantMode === "f32" ? "" : `_${quantMode}`;
        const metaUrl = `${modelPath}/model_meta${suffix}.json`;
        const binUrl = `${modelPath}/model_weights${suffix}.bin`;

        console.log(`WeightLoader: Loading ${quantMode} from ${modelPath}`);
        const [metaResp, binResp] = await Promise.all([
            fetch(metaUrl),
            fetch(binUrl)
        ]);

        if (!metaResp.ok || !binResp.ok) {
            console.error(`WeightLoader Error: Meta: ${metaResp.status}, Bin: ${binResp.status}`);
            throw new Error(`Failed to load weights for mode ${quantMode} at ${modelPath}`);
        }

        this.metadata = await metaResp.json();
        this.binaryData = await binResp.arrayBuffer();

        console.log(`Loaded WebGPU weights (${quantMode}). Binary size: ${(this.binaryData.byteLength / 1024 / 1024).toFixed(2)} MB`);
    }

    getMetadata(): ModelMetadata {
        if (!this.metadata) throw new Error("Metadata not loaded");
        return this.metadata;
    }

    getWeightData(offset: number, size: number): ArrayBuffer {
        if (!this.binaryData) throw new Error("Binary data not loaded");
        // We return the raw buffer slice, the engine will handle typing
        return this.binaryData.slice(offset, offset + size);
    }

    getBiasData(offset: number, size: number): Float32Array {
        if (!this.binaryData) throw new Error("Binary data not loaded");
        return new Float32Array(this.binaryData, offset, size / 4);
    }
}
