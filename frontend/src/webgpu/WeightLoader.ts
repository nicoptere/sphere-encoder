
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
    bias_offset: number;
    bias_size: number;
    activation: string;
}

export interface ModelMetadata {
    image_size: number;
    latent_dim: number;
    final_channels: number;
    encoder_layers: LayerMeta[];
    decoder_layers: LayerMeta[];
}

export class WeightLoader {
    private binaryData: ArrayBuffer | null = null;
    private metadata: ModelMetadata | null = null;

    async load(modelPath: string) {
        console.log(`WeightLoader: Loading from ${modelPath}`);
        const [metaResp, binResp] = await Promise.all([
            fetch(`${modelPath}/model_meta.json`),
            fetch(`${modelPath}/model_weights.bin`)
        ]);

        if (!metaResp.ok || !binResp.ok) {
            console.error(`WeightLoader Error: Meta: ${metaResp.status}, Bin: ${binResp.status}`);
            throw new Error(`Failed to load weights from ${modelPath}`);
        }

        const contentType = metaResp.headers.get("Content-Type");
        if (contentType && !contentType.includes("application/json")) {
            const text = await metaResp.text();
            console.error(`WeightLoader Error: Expected JSON, got ${contentType}. Content start: ${text.substring(0, 100)}`);
            throw new Error(`Expected JSON metadata but got ${contentType}`);
        }

        this.metadata = await metaResp.json();
        this.binaryData = await binResp.arrayBuffer();

        console.log(`Loaded WebGPU weights for ${this.metadata?.image_size}x${this.metadata?.image_size} model.`);
    }

    getMetadata(): ModelMetadata {
        if (!this.metadata) throw new Error("Metadata not loaded");
        return this.metadata;
    }

    getWeightData(offset: number, size: number): Float32Array {
        if (!this.binaryData) throw new Error("Binary data not loaded");
        return new Float32Array(this.binaryData, offset, size / 4);
    }
}
