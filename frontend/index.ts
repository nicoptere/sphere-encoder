import './src/css/index.scss';
import { WebGPUEngine } from './src/webgpu/Engine';

let LATENT_DIM = 512;
let IMAGE_SIZE = 32;
let MODEL_PATH = 'models/cifar10';

let engine: WebGPUEngine | null = null;

const inputCanvas = document.getElementById('inputCanvas') as HTMLCanvasElement;
const outputCanvas = document.getElementById('outputCanvas') as HTMLCanvasElement;
const randomCanvas = document.getElementById('randomCanvas') as HTMLCanvasElement;
const sampleBtn = document.getElementById('sampleBtn') as HTMLButtonElement;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const loopToggle = document.getElementById('loopToggle') as HTMLInputElement;
const roamToggle = document.getElementById('roamToggle') as HTMLInputElement;
const lengthSlider = document.getElementById('lengthSlider') as HTMLInputElement;
const speedSlider = document.getElementById('speedSlider') as HTMLInputElement;
const genTimeEl = document.getElementById('genTime') as HTMLDivElement;
const encTimeEl = document.getElementById('encTime') as HTMLDivElement;
const randomTimeEl = document.getElementById('randomTime') as HTMLDivElement;
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
const randomCtx = randomCanvas.getContext('2d', { willReadFrequently: true })!;
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
let nextCornerIndex = 0;
let targetInterpX = 0.5;
let targetInterpY = 0.5;
let currentInterpX = 0.5;
let currentInterpY = 0.5;

let allSampleUrls: string[] = [];

async function init() {
    const statusContainer = document.querySelector('.status-container') as HTMLDivElement;
    try {
        statusContainer.classList.remove('ready');
        sessionBusy = true; // Block loops during engine swap
        const quantSelect = document.getElementById('quantSelect') as HTMLSelectElement;
        const quantMode = quantSelect?.value || 'f32';

        statusEl.textContent = `Initializing WebGPU engine (${quantMode})...`;
        const newEngine = new WebGPUEngine();
        await newEngine.init(MODEL_PATH, quantMode);

        // Only assign and unblock once init is successful
        engine = newEngine;
        const metadata = engine.getMetadata()!;

        LATENT_DIM = metadata.latent_dim;
        IMAGE_SIZE = metadata.image_size;

        console.log(`WebGPU Model Loaded: ${IMAGE_SIZE}x${IMAGE_SIZE}, Latent: ${LATENT_DIM}, Mode: ${quantMode}`);

        // Re-initialize arrays
        cornerLatents = Array.from({ length: 4 }, () => new Float32Array(LATENT_DIM));
        orbitA = new Float32Array(LATENT_DIM);
        orbitB = new Float32Array(LATENT_DIM);

        [inputCanvas, outputCanvas, randomCanvas, interpResult, ...cornerCtxs.map(ctx => ctx.canvas)].forEach(canvas => {
            canvas.width = IMAGE_SIZE;
            canvas.height = IMAGE_SIZE;
        });

        statusEl.textContent = 'Syncing corners...';
        await setupInterpolationCorners();

        statusEl.textContent = 'Ready.';
        statusContainer.classList.add('ready');
        sampleBtn.disabled = false;
        sessionBusy = false; // Unblock

        randomizeLatent(); // Generate initial random image

        if (!loopStarted) {
            loopStarted = true;
            masterLoop();
        }
    } catch (e) {
        statusEl.textContent = 'Initialization error: ' + e;
        statusContainer.classList.add('ready'); // Hide throbber on error too
        console.error(e);
        sessionBusy = false;
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
    try {
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
    } finally {
        sessionBusy = false;
    }
}

async function randomizeLatent() {
    if (!engine || sessionBusy) return;
    sessionBusy = true;
    try {
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
        drawToCanvas(randomCtx, output);
        if (randomTimeEl) randomTimeEl.textContent = `${(end - start).toFixed(1)}ms`;
    } finally {
        sessionBusy = false;
    }
}

async function setupInterpolationCorners() {
    if (!engine || allSampleUrls.length === 0) return;

    // Pick 4 random unique samples
    const shuffled = [...allSampleUrls].sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, 4);

    for (let i = 0; i < 4; i++) {
        const url = selected[i];
        const img = new Image();
        await new Promise((resolve) => {
            img.onload = resolve;
            img.src = url;
        });

        drawImageCenterCrop(cornerCtxs[i], img);
        const pixels = getPixelsFromCanvas(cornerCtxs[i]);
        const latent = await engine.encode(pixels);
        cornerLatents[i] = latent;
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
    let lastFpsUpdate = performance.now();
    let frames = 0;

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
                const start = performance.now();
                const output = await engine.decode(latent);
                const end = performance.now();
                drawToCanvas(randomCtx, output);

                frames++;
                const now = performance.now();
                if (now - lastFpsUpdate > 500) {
                    const fps = (frames * 1000) / (now - lastFpsUpdate);
                    const ms = (now - lastFpsUpdate) / frames;
                    if (randomTimeEl) randomTimeEl.textContent = `${fps.toFixed(1)} FPS / ${ms.toFixed(1)}ms`;
                    frames = 0;
                    lastFpsUpdate = now;
                }

                sessionBusy = false;
            }
        } else {
            // Reset FPS tracking when not active to avoid huge spikes on resume
            frames = 0;
            lastFpsUpdate = performance.now();
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
randomizeInterp.addEventListener('click', setupInterpolationCorners);

function resetAnimations() {
    isLooping = false;
    isRoaming = false;
    loopToggle.checked = false;
    roamToggle.checked = false;
}

loopToggle.addEventListener('change', () => {
    isLooping = loopToggle.checked;
    if (isLooping) {
        isRoaming = false;
        roamToggle.checked = false;
    }
});

roamToggle.addEventListener('change', () => {
    isRoaming = roamToggle.checked;
    if (isRoaming) {
        isLooping = false;
        loopToggle.checked = false;
    }
});

const sourceUploadGroup = document.getElementById('sourceUploadGroup') as HTMLDivElement;
const thumbnailGrid = document.getElementById('thumbnailGrid') as HTMLDivElement;

async function handleImageUrl(url: string) {
    if (!engine) return;
    const img = new Image();
    img.onload = async () => {
        // Update main reconstruction
        drawImageCenterCrop(inputCtx, img);
        updateLatentRecon();

        // Also update interpolation corner
        const cornerIdx = nextCornerIndex;
        drawImageCenterCrop(cornerCtxs[cornerIdx], img);
        const pixels = getPixelsFromCanvas(cornerCtxs[cornerIdx]);
        const latent = await engine.encode(pixels);
        cornerLatents[cornerIdx] = latent;

        nextCornerIndex = (nextCornerIndex + 1) % 4;
        pendingInterpolation = true;
    };
    img.src = url;
}

// Populate Thumbnails
// Add FFHQ images
for (let i = 1; i <= 10; i++) {
    const url = `images/ffhq (${i}).png`;
    allSampleUrls.push(url);
    const img = document.createElement('img');
    img.src = url;
    img.className = 'thumb';
    img.alt = `FFHQ ${i}`;
    img.addEventListener('click', () => handleImageUrl(url));
    thumbnailGrid.appendChild(img);
}

// Add Flower images
for (let i = 1; i <= 10; i++) {
    const url = `images/flower (${i}).jpg`;
    allSampleUrls.push(url);
    const img = document.createElement('img');
    img.src = url;
    img.className = 'thumb';
    img.alt = `Flower ${i}`;
    img.addEventListener('click', () => handleImageUrl(url));
    thumbnailGrid.appendChild(img);
}

function handleFile(file: File) {
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

fileInput.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) handleFile(file);
    fileInput.value = ''; // Reset to allow re-selection of same file
});

sourceUploadGroup.addEventListener('dragover', (e) => {
    e.preventDefault();
    sourceUploadGroup.classList.add('dragover');
});

sourceUploadGroup.addEventListener('dragleave', () => {
    sourceUploadGroup.classList.remove('dragover');
});

sourceUploadGroup.addEventListener('drop', (e) => {
    e.preventDefault();
    sourceUploadGroup.classList.remove('dragover');
    const file = e.dataTransfer?.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

inputCanvas.addEventListener('click', () => {
    fileInput.click();
});

const modelSelect = document.getElementById('modelSelect') as HTMLSelectElement;
const quantSelect = document.getElementById('quantSelect') as HTMLSelectElement;

if (modelSelect) {
    modelSelect.addEventListener('change', () => {
        resetAnimations();
        MODEL_PATH = `models/${modelSelect.value}`;
        console.log(`Model selection changed to: ${MODEL_PATH}`);
        init();
    });
    MODEL_PATH = `models/${modelSelect.value}`;
}

if (quantSelect) {
    quantSelect.addEventListener('change', () => {
        resetAnimations();
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
