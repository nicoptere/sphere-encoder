
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
        const quantSelect = document.getElementById('quantSelect') as HTMLSelectElement;
        const quantMode = quantSelect?.value || 'f32';

        statusEl.textContent = `Initializing WebGPU engine (${quantMode})...`;
        engine = new WebGPUEngine();
        await engine.init(MODEL_PATH, quantMode);
        const metadata = engine.getMetadata()!;

        LATENT_DIM = metadata.latent_dim;
        IMAGE_SIZE = metadata.image_size;

        console.log(`WebGPU Model Loaded: ${IMAGE_SIZE}x${IMAGE_SIZE}, Latent: ${LATENT_DIM}, Mode: ${quantMode}`);

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
}

async function updateLatentRecon() {
    if (!engine || sessionBusy) return;
    sessionBusy = true;
    const pixels = getPixelsFromCanvas(inputCtx);
    const encStart = performance.now();
    const latent = await engine.encode(pixels);
    const encEnd = performance.now();

    encTimeEl.textContent = `${(encEnd - encStart).toFixed(1)}ms (Enc)`;
    setSourceLatent(latent);

    const decStart = performance.now();
    const output = await engine.decode(latent);
    const decEnd = performance.now();
    drawToCanvas(outputCtx, output);
    genTimeEl.textContent = `${(decEnd - decStart).toFixed(1)}ms (Dec)`;

    sessionBusy = false;
}

async function randomizeLatent() {
    if (!engine || sessionBusy) return;
    sessionBusy = true;
    const latent = new Float32Array(LATENT_DIM);
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
    const start = performance.now();
    const output = await engine.decode(latent);
    const end = performance.now();
    drawToCanvas(outputCtx, output);
    genTimeEl.textContent = `${(end - start).toFixed(1)}ms`;
    sessionBusy = false;
}

async function setupInterpolationCorners() {
    if (!engine) return;
    for (let i = 0; i < 4; i++) {
        const latent = new Float32Array(LATENT_DIM);
        let sumSq = 0;
        for (let j = 0; j < LATENT_DIM; j++) {
            const z = Math.sqrt(-2.0 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2.0 * Math.PI * Math.random());
            latent[j] = z;
            sumSq += z * z;
        }
        const norm = Math.sqrt(sumSq);
        const radius = Math.sqrt(LATENT_DIM);
        for (let j = 0; j < LATENT_DIM; j++) latent[j] = (latent[j] / norm) * radius;

        cornerLatents[i] = latent;
        const output = await engine.decode(latent);
        drawToCanvas(cornerCtxs[i], output);
    }
    pendingInterpolation = true;
}

async function runInterpolation(x: number, y: number) {
    if (!engine || sessionBusy) return;
    sessionBusy = true;

    const latent = new Float32Array(LATENT_DIM);
    const wTL = (1 - x) * (1 - y);
    const wTR = x * (1 - y);
    const wBL = (1 - x) * y;
    const wBR = x * y;

    let sumSq = 0;
    for (let i = 0; i < LATENT_DIM; i++) {
        const val = cornerLatents[0][i] * wTL + cornerLatents[1][i] * wTR + cornerLatents[2][i] * wBL + cornerLatents[3][i] * wBR;
        latent[i] = val;
        sumSq += val * val;
    }
    const norm = Math.sqrt(sumSq);
    const radius = Math.sqrt(LATENT_DIM);
    for (let i = 0; i < LATENT_DIM; i++) latent[i] = (latent[i] / norm) * radius;

    const output = await engine.decode(latent);
    drawToCanvas(interpResultCtx, output);
    sessionBusy = false;
    pendingInterpolation = false;
}

async function masterLoop() {
    const loop = async () => {
        if (!engine) { requestAnimationFrame(loop); return; }

        if (isLooping || isRoaming) {
            if (!sessionBusy) {
                let latent: Float32Array;
                if (isRoaming && sourceLatent) {
                    const time = performance.now() * 0.001 * parseFloat(speedSlider.value);
                    const length = parseFloat(lengthSlider.value);
                    latent = new Float32Array(LATENT_DIM);
                    let sumSq = 0;
                    for (let i = 0; i < LATENT_DIM; i++) {
                        const val = sourceLatent[i] + length * (orbitA[i] * Math.cos(time) + orbitB[i] * Math.sin(time));
                        latent[i] = val;
                        sumSq += val * val;
                    }
                    const norm = Math.sqrt(sumSq);
                    const radius = Math.sqrt(LATENT_DIM);
                    for (let i = 0; i < LATENT_DIM; i++) latent[i] = (latent[i] / norm) * radius;
                } else {
                    latent = new Float32Array(LATENT_DIM);
                    for (let i = 0; i < LATENT_DIM; i++) latent[i] = Math.sqrt(-2 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2 * Math.PI * Math.random());
                    const norm = Math.sqrt(latent.reduce((a, b) => a + b * b, 0));
                    const radius = Math.sqrt(LATENT_DIM);
                    for (let i = 0; i < LATENT_DIM; i++) latent[i] = (latent[i] / norm) * radius;
                }

                sessionBusy = true;
                const output = await engine.decode(latent);
                drawToCanvas(outputCtx, output);
                sessionBusy = false;
            }
        }

        if (pendingInterpolation || Math.abs(currentInterpX - targetInterpX) > 0.01 || Math.abs(currentInterpY - targetInterpY) > 0.01) {
            currentInterpX += (targetInterpX - currentInterpX) * 0.1;
            currentInterpY += (targetInterpY - currentInterpY) * 0.1;
            interpCursor.style.left = `${currentInterpX * 100}%`;
            interpCursor.style.top = `${currentInterpY * 100}%`;
            await runInterpolation(currentInterpX, currentInterpY);
        }

        requestAnimationFrame(loop);
    };
    loop();
}

sampleBtn.addEventListener('click', randomizeLatent);
reconstructBtn.addEventListener('click', updateLatentRecon);
randomizeInterp.addEventListener('click', setupInterpolationCorners);

loopToggle.addEventListener('change', () => { isLooping = loopToggle.checked; });
roamToggle.addEventListener('change', () => { isRoaming = roamToggle.checked; });

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (re) => {
            const img = new Image();
            img.onload = () => {
                drawImageCenterCrop(inputCtx, img);
                updateLatentRecon();
            };
            img.src = re.target?.result as string;
        };
        reader.readAsDataURL(file);
    }
});

const modelSelect = document.getElementById('modelSelect') as HTMLSelectElement;
const quantSelect = document.getElementById('quantSelect') as HTMLSelectElement;

if (modelSelect) {
    modelSelect.addEventListener('change', () => {
        MODEL_PATH = `models/${modelSelect.value}`;
        console.log(`Model selection changed to: ${MODEL_PATH}`);
        init();
    });
    MODEL_PATH = `models/${modelSelect.value}`;
}

if (quantSelect) {
    quantSelect.addEventListener('change', () => {
        console.log(`Quantization changed to: ${quantSelect.value}`);
        init();
    });
}

interpPad.addEventListener('mousedown', (e) => {
    const rect = interpPad.getBoundingClientRect();
    const update = (me: MouseEvent) => {
        targetInterpX = Math.max(0, Math.min(1, (me.clientX - rect.left) / rect.width));
        targetInterpY = Math.max(0, Math.min(1, (me.clientY - rect.top) / rect.height));
    };
    const up = () => {
        window.removeEventListener('mousemove', update);
        window.removeEventListener('mouseup', up);
    };
    window.addEventListener('mousemove', update);
    window.addEventListener('mouseup', up);
    update(e);
});

init();
