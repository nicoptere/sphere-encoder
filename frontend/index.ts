import * as ort from 'onnxruntime-web';

const LATENT_DIM = 512;
const IMAGE_SIZE = 32;

let encoderSession: ort.InferenceSession | null = null;
let decoderSession: ort.InferenceSession | null = null;

const inputCanvas = document.getElementById('inputCanvas') as HTMLCanvasElement;
const outputCanvas = document.getElementById('outputCanvas') as HTMLCanvasElement;
const sampleBtn = document.getElementById('sampleBtn') as HTMLButtonElement;
const uploadBtn = document.getElementById('uploadBtn') as HTMLButtonElement;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const statusEl = document.getElementById('status') as HTMLDivElement;

const inputCtx = inputCanvas.getContext('2d')!;
const outputCtx = outputCanvas.getContext('2d')!;

// Configure ONNX wasm paths
ort.env.wasm.wasmPaths = {
    'ort-wasm-simd-threaded.mjs': '/ort-assets/ort-wasm-simd-threaded.js',
    'ort-wasm-simd-threaded.jsep.mjs': '/ort-assets/ort-wasm-simd-threaded.jsep.js',
    'ort-wasm-simd-threaded.asyncify.mjs': '/ort-assets/ort-wasm-simd-threaded.asyncify.js',
    'ort-wasm-simd-threaded.jspi.mjs': '/ort-assets/ort-wasm-simd-threaded.jspi.js',
    'ort-wasm-simd-threaded.wasm': '/ort-assets/ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.jsep.wasm': '/ort-assets/ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.asyncify.wasm': '/ort-assets/ort-wasm-simd-threaded.asyncify.wasm',
    'ort-wasm-simd-threaded.jspi.wasm': '/ort-assets/ort-wasm-simd-threaded.jspi.wasm',
    'ort-wasm.wasm': '/ort-assets/ort-wasm.wasm',
    'ort-wasm-simd.wasm': '/ort-assets/ort-wasm-simd.wasm',
};
ort.env.wasm.proxy = true;
ort.env.wasm.numThreads = 1;

async function init() {
    try {
        statusEl.textContent = 'Loading models...';
        // Path relative to public folder in Vite
        encoderSession = await ort.InferenceSession.create('/model/encoder.onnx');
        decoderSession = await ort.InferenceSession.create('/model/decoder.onnx');
        statusEl.textContent = 'Models loaded! Ready.';
        sampleBtn.disabled = false;
    } catch (e) {
        statusEl.textContent = 'Error loading models: ' + e;
        console.error(e);
    }
}

// Utility to draw data to canvas
function drawToCanvas(ctx: CanvasRenderingContext2D, data: Float32Array) {
    const imageData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        // data is [-1, 1], convert to [0, 255]
        // data order is C, H, W (3, 32, 32)
        const r = (data[i] * 0.5 + 0.5) * 255;
        const g = (data[i + IMAGE_SIZE * IMAGE_SIZE] * 0.5 + 0.5) * 255;
        const b = (data[i + 2 * IMAGE_SIZE * IMAGE_SIZE] * 0.5 + 0.5) * 255;

        const idx = i * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

// Utility to get tensor from image
function getTensorFromCanvas(ctx: CanvasRenderingContext2D): ort.Tensor {
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const data = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        const idx = i * 4;
        // [0, 255] -> [-1, 1]
        data[i] = (imageData.data[idx] / 255) * 2 - 1;
        data[i + IMAGE_SIZE * IMAGE_SIZE] = (imageData.data[idx + 1] / 255) * 2 - 1;
        data[i + 2 * IMAGE_SIZE * IMAGE_SIZE] = (imageData.data[idx + 2] / 255) * 2 - 1;
    }
    return new ort.Tensor('float32', data, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

async function runInference(inputData: Float32Array | null = null) {
    if (!encoderSession || !decoderSession) return;

    let latent: ort.Tensor;

    if (inputData) {
        // Reconstruction mode
        const inputTensor = new ort.Tensor('float32', inputData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
        const encoderResults = await encoderSession.run({ input: inputTensor });
        latent = encoderResults.latent;
    } else {
        // Sampling mode: Random noise normalized to sphere
        const noise = new Float32Array(LATENT_DIM);
        let sumSq = 0;
        for (let i = 0; i < LATENT_DIM; i++) {
            noise[i] = Math.random() * 2 - 1; // Simplistic random, N(0,1) would be better
            // Box-Muller for N(0,1)
            const u1 = Math.random();
            const u2 = Math.random();
            const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            noise[i] = z0;
            sumSq += z0 * z0;
        }
        const norm = Math.sqrt(sumSq);
        const radius = Math.sqrt(LATENT_DIM);
        for (let i = 0; i < LATENT_DIM; i++) {
            noise[i] = (noise[i] / norm) * radius;
        }
        latent = new ort.Tensor('float32', noise, [1, LATENT_DIM]);
    }

    const decoderResults = await decoderSession.run({ latent: latent });
    const output = decoderResults.output.data as Float32Array;

    drawToCanvas(outputCtx, output);
}

sampleBtn.addEventListener('click', () => {
    runInference();
});

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        inputCtx.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
        inputCtx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
        const tensor = getTensorFromCanvas(inputCtx);
        runInference(tensor.data as Float32Array);
    };
    img.src = URL.createObjectURL(file);
});

init();
