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
const loopToggle = document.getElementById('loopToggle') as HTMLInputElement;
const genTimeEl = document.getElementById('genTime') as HTMLDivElement;
const statusEl = document.getElementById('status') as HTMLDivElement;

const cornerTL = document.getElementById('corner-tl') as HTMLCanvasElement;
const cornerTR = document.getElementById('corner-tr') as HTMLCanvasElement;
const cornerBL = document.getElementById('corner-bl') as HTMLCanvasElement;
const cornerBR = document.getElementById('corner-br') as HTMLCanvasElement;
const interpPad = document.getElementById('interpPad') as HTMLDivElement;
const interpCursor = document.getElementById('interpCursor') as HTMLDivElement;
const interpResult = document.getElementById('interpResult') as HTMLCanvasElement;
const randomizeInterp = document.getElementById('randomizeInterp') as HTMLButtonElement;

const inputCtx = inputCanvas.getContext('2d')!;
const outputCtx = outputCanvas.getContext('2d')!;
const cornerCtxs = [
    cornerTL.getContext('2d')!,
    cornerTR.getContext('2d')!,
    cornerBL.getContext('2d')!,
    cornerBR.getContext('2d')!
];
const interpResultCtx = interpResult.getContext('2d')!;

// Utility to draw image center-cropped and resized
function drawImageCenterCrop(ctx: CanvasRenderingContext2D, img: HTMLImageElement) {
    const minDim = Math.min(img.width, img.height);
    const sx = (img.width - minDim) / 2;
    const sy = (img.height - minDim) / 2;
    ctx.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    ctx.drawImage(img, sx, sy, minDim, minDim, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
}

let cornerLatents: Float32Array[] = [
    new Float32Array(LATENT_DIM),
    new Float32Array(LATENT_DIM),
    new Float32Array(LATENT_DIM),
    new Float32Array(LATENT_DIM)
];

let isLooping = false;
let sessionBusy = false;
let pendingInterpolation = false;

// Configure ONNX wasm paths
ort.env.wasm.wasmPaths = {
    'ort-wasm-simd-threaded.jsep.wasm': '/ort-assets/ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.jsep.js': '/ort-assets/ort-wasm-simd-threaded.jsep.js',
    'ort-wasm-simd-threaded.wasm': '/ort-assets/ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.js': '/ort-assets/ort-wasm-simd-threaded.js',
};
ort.env.wasm.proxy = true;
ort.env.wasm.numThreads = 1;

async function init() {
    try {
        statusEl.textContent = 'Loading models...';
        // Path relative to public folder in Vite
        const sessionOptions: ort.InferenceSession.SessionOptions = {
            executionProviders: ['webgpu', 'wasm'],
        };
        encoderSession = await ort.InferenceSession.create('/model/encoder.onnx', sessionOptions);
        decoderSession = await ort.InferenceSession.create('/model/decoder.onnx', sessionOptions);
        statusEl.textContent = 'Models loaded! Ready.';
        sampleBtn.disabled = false;

        // Setup interpolation
        await setupInterpolationCorners();
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
    if (!encoderSession || !decoderSession || sessionBusy) return;

    sessionBusy = true;
    try {
        const startTime = performance.now();
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

        const endTime = performance.now();
        genTimeEl.textContent = `${(endTime - startTime).toFixed(1)} ms`;

        drawToCanvas(outputCtx, output);
    } finally {
        sessionBusy = false;
    }
}

async function loop() {
    if (!isLooping) return;
    await runInference();
    // Use requestAnimationFrame for smooth looping or just a microtask if we want "as fast as possible"
    // requestAnimationFrame is better for browser responsiveness
    requestAnimationFrame(loop);
}

sampleBtn.addEventListener('click', () => {
    runInference();
});

loopToggle.addEventListener('change', () => {
    isLooping = loopToggle.checked;
    if (isLooping) {
        loop();
    }
});

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        drawImageCenterCrop(inputCtx, img);
        const tensor = getTensorFromCanvas(inputCtx);
        runInference(tensor.data as Float32Array);
    };
    img.src = URL.createObjectURL(file);
});

init();

// --- Interpolation Logic ---

async function fetchRandomFlowerImage(): Promise<HTMLImageElement> {
    const idx = Math.floor(Math.random() * 10) + 1;
    const url = `/images/flower (${idx}).jpg`;
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = (e) => reject(e);
        img.src = url;
    });
}

async function fetchAndEncodeCorner(i: number) {
    const img = await fetchRandomFlowerImage();
    const ctx = cornerCtxs[i];
    drawImageCenterCrop(ctx, img);
    const tensor = getTensorFromCanvas(ctx);
    const results = await encoderSession!.run({ input: tensor });
    cornerLatents[i] = results.latent.data as Float32Array;
}

async function randomizeCorner(i: number) {
    if (!encoderSession || sessionBusy) return;
    sessionBusy = true;
    try {
        await fetchAndEncodeCorner(i);
    } catch (e) {
        console.error(`Failed to randomize corner ${i}:`, e);
    } finally {
        sessionBusy = false;
        updateInterpolation();
    }
}

async function setupInterpolationCorners() {
    if (!encoderSession || sessionBusy) return;
    sessionBusy = true;
    statusEl.textContent = 'Encoding corners...';
    try {
        for (let i = 0; i < 4; i++) {
            await fetchAndEncodeCorner(i);
        }
    } catch (e) {
        console.error("Failed to setup corners:", e);
    } finally {
        sessionBusy = false;
        statusEl.textContent = 'Models loaded! Ready.';
        updateInterpolation();
    }
}

async function updateInterpolation() {
    if (!decoderSession) return;
    if (sessionBusy) {
        pendingInterpolation = true;
        return;
    }

    sessionBusy = true;
    try {
        const px = (parseFloat(interpCursor.style.left) || 50) / 100;
        const py = (parseFloat(interpCursor.style.top) || 50) / 100;

        const x = px;
        const y = py;

        const interpolated = new Float32Array(LATENT_DIM);
        let sumSq = 0;
        for (let i = 0; i < LATENT_DIM; i++) {
            // Bilinear interpolation
            const top = (1 - x) * cornerLatents[0][i] + x * cornerLatents[1][i];
            const bottom = (1 - x) * cornerLatents[2][i] + x * cornerLatents[3][i];
            const val = (1 - y) * top + y * bottom;
            interpolated[i] = val;
            sumSq += val * val;
        }

        // Renormalize to sphere radius sqrt(L)
        const norm = Math.sqrt(sumSq);
        const radius = Math.sqrt(LATENT_DIM);
        for (let i = 0; i < LATENT_DIM; i++) {
            interpolated[i] = (interpolated[i] / norm) * radius;
        }

        const latent = new ort.Tensor('float32', interpolated, [1, LATENT_DIM]);
        const decoderResults = await decoderSession.run({ latent: latent });
        const output = decoderResults.output.data as Float32Array;
        drawToCanvas(interpResultCtx, output);
    } finally {
        sessionBusy = false;
        if (pendingInterpolation) {
            pendingInterpolation = false;
            updateInterpolation();
        }
    }
}

function handlePadInteraction(e: MouseEvent | TouchEvent) {
    const rect = interpPad.getBoundingClientRect();
    let clientX, clientY;

    if ('touches' in e) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }

    let x = (clientX - rect.left) / rect.width;
    let y = (clientY - rect.top) / rect.height;

    // Constrain
    x = Math.max(0, Math.min(1, x));
    y = Math.max(0, Math.min(1, y));

    interpCursor.style.left = `${x * 100}%`;
    interpCursor.style.top = `${y * 100}%`;

    updateInterpolation();
}

interpPad.addEventListener('mousedown', (e) => {
    handlePadInteraction(e);
    const onMouseMove = (moveEvent: MouseEvent) => handlePadInteraction(moveEvent);
    const onMouseUp = () => {
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
});

randomizeInterp.addEventListener('click', () => {
    setupInterpolationCorners();
});

[cornerTL, cornerTR, cornerBL, cornerBR].forEach((canvas, i) => {
    canvas.addEventListener('click', () => randomizeCorner(i));
    canvas.style.cursor = 'pointer';
});
