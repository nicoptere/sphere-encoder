
import { WebGPUEngine } from './src/webgpu/Engine';

let LATENT_DIM = 512;
let IMAGE_SIZE = 32;
let MODEL_PATH = 'models/ffhq_64';

let engine: WebGPUEngine | null = null;

const inputCanvas = document.getElementById('inputCanvas') as HTMLCanvasElement;
const outputCanvas = document.getElementById('outputCanvas') as HTMLCanvasElement;
const sampleBtn = document.getElementById('sampleBtn') as HTMLButtonElement;
const uploadBtn = document.getElementById('uploadBtn') as HTMLButtonElement;
const reconstructBtn = document.getElementById('reconstructBtn') as HTMLButtonElement;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const loopToggle = document.getElementById('loopToggle') as HTMLInputElement;
const roamToggle = document.getElementById('roamToggle') as HTMLInputElement;
const lengthSlider = document.getElementById('lengthSlider') as HTMLInputElement;
const speedSlider = document.getElementById('speedSlider') as HTMLInputElement;
const genTimeEl = document.getElementById('genTime') as HTMLDivElement;
const encTimeEl = document.getElementById('encTime') as HTMLDivElement;
const statusEl = document.getElementById('status') as HTMLDivElement;

const cornerTL = document.getElementById('corner-tl') as HTMLCanvasElement;
const cornerTR = document.getElementById('corner-tr') as HTMLCanvasElement;
const cornerBL = document.getElementById('corner-bl') as HTMLCanvasElement;
const cornerBR = document.getElementById('corner-br') as HTMLCanvasElement;
const interpPad = document.getElementById('interpPad') as HTMLDivElement;
const interpCursor = document.getElementById('interpCursor') as HTMLDivElement;
const interpResult = document.getElementById('interpResult') as HTMLCanvasElement;
const randomizeInterp = document.getElementById('randomizeInterp') as HTMLButtonElement;

const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true })!;
const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true })!;
const cornerCtxs = [
    cornerTL.getContext('2d', { willReadFrequently: true })!,
    cornerTR.getContext('2d', { willReadFrequently: true })!,
    cornerBL.getContext('2d', { willReadFrequently: true })!,
    cornerBR.getContext('2d', { willReadFrequently: true })!
];
const interpResultCtx = interpResult.getContext('2d', { willReadFrequently: true })!;

// Utility to draw image center-cropped and resized
function drawImageCenterCrop(ctx: CanvasRenderingContext2D, img: HTMLImageElement) {
    const minDim = Math.min(img.width, img.height);
    const sx = (img.width - minDim) / 2;
    const sy = (img.height - minDim) / 2;
    ctx.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    ctx.drawImage(img, sx, sy, minDim, minDim, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
}

let cornerLatents: Float32Array[] = [];
let isLooping = false;
let isRoaming = false;
let sessionBusy = false;
let pendingInterpolation = false;

let sourceLatent: Float32Array | null = null;
let orbitA: Float32Array;
let orbitB: Float32Array;

let currentCornerIndices = [-1, -1, -1, -1];
let targetInterpX = 0.5;
let targetInterpY = 0.5;
let currentInterpX = 0.5;
let currentInterpY = 0.5;

async function init() {
    try {
        statusEl.textContent = 'Initializing WebGPU engine...';
        engine = new WebGPUEngine();
        await engine.init(MODEL_PATH);
        const metadata = engine.getMetadata()!;

        LATENT_DIM = metadata.latent_dim;
        IMAGE_SIZE = metadata.image_size;

        console.log(`WebGPU Model Loaded: ${IMAGE_SIZE}x${IMAGE_SIZE}, Latent: ${LATENT_DIM}`);

        // Re-initialize arrays
        cornerLatents = Array.from({ length: 4 }, () => new Float32Array(LATENT_DIM));
        orbitA = new Float32Array(LATENT_DIM);
        orbitB = new Float32Array(LATENT_DIM);

        [inputCanvas, outputCanvas, interpResult, ...cornerCtxs.map(ctx => ctx.canvas)].forEach(canvas => {
            canvas.width = IMAGE_SIZE;
            canvas.height = IMAGE_SIZE;
        });

        statusEl.textContent = 'Syncing corners...';
        await setupInterpolationCorners();

        statusEl.textContent = 'Ready.';
        sampleBtn.disabled = false;
        reconstructBtn.disabled = false;

        if (!loopStarted) {
            loopStarted = true;
            masterLoop();
        }
    } catch (e) {
        statusEl.textContent = 'Initialization error: ' + e;
        console.error(e);
    }
}

let loopStarted = false;

function drawToCanvas(ctx: CanvasRenderingContext2D, data: Float32Array) {
    const imageData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
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

function getPixelsFromCanvas(ctx: CanvasRenderingContext2D): Float32Array {
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const data = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        const idx = i * 4;
        data[i] = (imageData.data[idx] / 255) * 2 - 1;
        data[i + IMAGE_SIZE * IMAGE_SIZE] = (imageData.data[idx + 1] / 255) * 2 - 1;
        data[i + 2 * IMAGE_SIZE * IMAGE_SIZE] = (imageData.data[idx + 2] / 255) * 2 - 1;
    }
    return data;
}

function setSourceLatent(data: Float32Array) {
    sourceLatent = new Float32Array(data);
    const radius = Math.sqrt(LATENT_DIM);
    for (let i = 0; i < LATENT_DIM; i++) {
        orbitA[i] = Math.sqrt(-2.0 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2.0 * Math.PI * Math.random());
        orbitB[i] = Math.sqrt(-2.0 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2.0 * Math.PI * Math.random());
    }
    // Simple orthonormalization
    let dotA = 0;
    for (let i = 0; i < LATENT_DIM; i++) dotA += orbitA[i] * orbitA[i];
    const normA = Math.sqrt(dotA);
    for (let i = 0; i < LATENT_DIM; i++) orbitA[i] /= normA;

    let dotAB = 0;
    for (let i = 0; i < LATENT_DIM; i++) dotAB += orbitA[i] * orbitB[i];
    for (let i = 0; i < LATENT_DIM; i++) orbitB[i] -= dotAB * orbitA[i];

    let dotB = 0;
    for (let i = 0; i < LATENT_DIM; i++) dotB += orbitB[i] * orbitB[i];
    const normB = Math.sqrt(dotB);
    for (let i = 0; i < LATENT_DIM; i++) orbitB[i] /= normB;

    for (let i = 0; i < LATENT_DIM; i++) {
        orbitA[i] *= radius;
        orbitB[i] *= radius;
    }
}

async function runInference(pixels: Float32Array | null = null) {
    if (!engine || sessionBusy) return;
    sessionBusy = true;
    try {
        const startTime = performance.now();
        let latent: Float32Array;

        if (pixels) {
            const encStart = performance.now();
            latent = await engine.encode(pixels);
            const encEnd = performance.now();
            encTimeEl.textContent = `${(encEnd - encStart).toFixed(1)} ms`;
            setSourceLatent(latent);
        } else {
            encTimeEl.textContent = '';
            latent = new Float32Array(LATENT_DIM);
            let sumSq = 0;
            for (let i = 0; i < LATENT_DIM; i++) {
                const z = Math.sqrt(-2.0 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2.0 * Math.PI * Math.random());
                latent[i] = z;
                sumSq += z * z;
            }
            const norm = Math.sqrt(sumSq);
            const radius = Math.sqrt(LATENT_DIM);
            for (let i = 0; i < LATENT_DIM; i++) latent[i] = (latent[i] / norm) * radius;
            setSourceLatent(latent);
        }

        const output = await engine.decode(latent);
        const endTime = performance.now();
        genTimeEl.textContent = `${(endTime - startTime).toFixed(1)} ms`;
        drawToCanvas(outputCtx, output);
    } finally {
        sessionBusy = false;
    }
}

async function masterLoop() {
    const dx = targetInterpX - currentInterpX;
    const dy = targetInterpY - currentInterpY;
    const interpMoved = Math.abs(dx) > 0.0001 || Math.abs(dy) > 0.0001;

    if (interpMoved) {
        currentInterpX += dx * 0.1;
        currentInterpY += dy * 0.1;
        interpCursor.style.left = `${currentInterpX * 100}%`;
        interpCursor.style.top = `${currentInterpY * 100}%`;
    }

    if (!sessionBusy && engine) {
        if (interpMoved || pendingInterpolation) {
            pendingInterpolation = false;
            await runInterpolationUpdate();
        } else if (isRoaming && sourceLatent) {
            await runRoamFrame();
        } else if (isLooping) {
            await runInference();
        }
    }
    requestAnimationFrame(masterLoop);
}

async function runRoamFrame() {
    if (!sourceLatent || !engine) return;
    sessionBusy = true;
    try {
        const startTime = performance.now();
        const angle = startTime * 0.001 * parseFloat(speedSlider.value);
        const length = parseFloat(lengthSlider.value);
        const radius = Math.sqrt(LATENT_DIM);

        const morphed = new Float32Array(LATENT_DIM);
        let sumSq = 0;
        const cosL = Math.cos(length);
        const sinL = Math.sin(length);
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);

        for (let i = 0; i < LATENT_DIM; i++) {
            const val = sourceLatent[i] * cosL + (orbitA[i] * cosA + orbitB[i] * sinA) * sinL;
            morphed[i] = val;
            sumSq += val * val;
        }

        const norm = Math.sqrt(sumSq);
        for (let i = 0; i < LATENT_DIM; i++) morphed[i] = (morphed[i] / norm) * radius;

        const output = await engine.decode(morphed);
        genTimeEl.textContent = `${(performance.now() - startTime).toFixed(1)} ms`;
        drawToCanvas(outputCtx, output);
    } finally {
        sessionBusy = false;
    }
}

async function runInterpolationUpdate() {
    if (!engine) return;
    sessionBusy = true;
    try {
        const x = currentInterpX;
        const y = currentInterpY;
        const interpolated = new Float32Array(LATENT_DIM);
        let sumSq = 0;
        for (let i = 0; i < LATENT_DIM; i++) {
            const top = (1 - x) * cornerLatents[0][i] + x * cornerLatents[1][i];
            const bottom = (1 - x) * cornerLatents[2][i] + x * cornerLatents[3][i];
            const val = (1 - y) * top + y * bottom;
            interpolated[i] = val;
            sumSq += val * val;
        }
        const norm = Math.sqrt(sumSq);
        const radius = Math.sqrt(LATENT_DIM);
        for (let i = 0; i < LATENT_DIM; i++) interpolated[i] = (interpolated[i] / norm) * radius;

        const output = await engine.decode(interpolated);
        drawToCanvas(interpResultCtx, output);
    } finally {
        sessionBusy = false;
    }
}

sampleBtn.addEventListener('click', () => runInference());
reconstructBtn.addEventListener('click', () => runInference(getPixelsFromCanvas(inputCtx)));
loopToggle.addEventListener('change', () => isLooping = loopToggle.checked);
roamToggle.addEventListener('change', () => isRoaming = roamToggle.checked);
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
        drawImageCenterCrop(inputCtx, img);
        runInference(getPixelsFromCanvas(inputCtx));
    };
    img.src = URL.createObjectURL(file);
});

async function fetchRandomFlowerImage(exclude: number[] = []): Promise<{ img: HTMLImageElement, idx: number }> {
    let idx: number;
    do { idx = Math.floor(Math.random() * 10) + 1; } while (exclude.includes(idx));
    const url = `/images/flower (${idx}).jpg`;
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve({ img, idx });
        img.onerror = reject;
        img.src = url;
    });
}

async function fetchAndEncodeCorner(i: number) {
    const exclude = currentCornerIndices.filter((_, index) => index !== i);
    const { img, idx } = await fetchRandomFlowerImage(exclude);
    currentCornerIndices[i] = idx;
    drawImageCenterCrop(cornerCtxs[i], img);
    cornerLatents[i] = await engine!.encode(getPixelsFromCanvas(cornerCtxs[i]));
}

async function setupInterpolationCorners() {
    for (let i = 0; i < 4; i++) await fetchAndEncodeCorner(i);
}

const modelSelect = document.getElementById('modelSelect') as HTMLSelectElement;
if (modelSelect) {
    modelSelect.addEventListener('change', () => {
        MODEL_PATH = `models/${modelSelect.value}`;
        console.log(`Model selection changed to: ${MODEL_PATH}`);
        init();
    });
    MODEL_PATH = `models/${modelSelect.value}`;
    console.log(`Initial model path from select: ${MODEL_PATH}`);
}

init();

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

randomizeInterp.addEventListener('click', () => setupInterpolationCorners().then(() => pendingInterpolation = true));
cornerCtxs.forEach((ctx, i) => {
    ctx.canvas.addEventListener('click', () => {
        if (!sessionBusy) fetchAndEncodeCorner(i).then(() => pendingInterpolation = true);
    });
    ctx.canvas.style.cursor = 'pointer';
});

function handlePadInteraction(e: MouseEvent | TouchEvent) {
    const rect = interpPad.getBoundingClientRect();
    const cx = ('touches' in e) ? e.touches[0].clientX : e.clientX;
    const cy = ('touches' in e) ? e.touches[0].clientY : e.clientY;
    targetInterpX = Math.max(0, Math.min(1, (cx - rect.left) / rect.width));
    targetInterpY = Math.max(0, Math.min(1, (cy - rect.top) / rect.height));
}
