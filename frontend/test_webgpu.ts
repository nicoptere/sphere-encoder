
import { WebGPUEngine } from './src/webgpu/Engine';

const statusEl = document.getElementById('status')!;
const canvas = document.getElementById('webgpuCanvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
const testBtn = document.getElementById('testBtn') as HTMLButtonElement;
const timeEl = document.getElementById('webgpuTime')!;
const latentDimEl = document.getElementById('latentDim')!;

let engine: WebGPUEngine | null = null;
let IMAGE_SIZE = 64;
let LATENT_DIM = 512;

async function init() {
    try {
        statusEl.textContent = 'Loading custom WebGPU engine...';
        engine = new WebGPUEngine();
        await engine.init('models/ffhq_64');

        const metadata = engine.getMetadata()!;
        IMAGE_SIZE = metadata.image_size;
        LATENT_DIM = metadata.latent_dim;

        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;
        latentDimEl.textContent = LATENT_DIM.toString();

        statusEl.textContent = 'WebGPU Engine Ready!';
        testBtn.disabled = false;
    } catch (e) {
        statusEl.textContent = 'Error: ' + e;
        console.error(e);
    }
}

async function runTest() {
    if (!engine) return;

    const latent = new Float32Array(LATENT_DIM);
    let sumSq = 0;
    for (let i = 0; i < LATENT_DIM; i++) {
        const u1 = Math.max(1e-10, Math.random());
        const u2 = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        latent[i] = z;
        sumSq += z * z;
    }
    const norm = Math.sqrt(sumSq);
    const radius = Math.sqrt(LATENT_DIM);
    for (let i = 0; i < LATENT_DIM; i++) {
        latent[i] = (latent[i] / norm) * radius;
    }

    statusEl.textContent = 'Running inference...';
    const start = performance.now();
    const output = await engine.decode(latent);
    const end = performance.now();

    timeEl.textContent = `${(end - start).toFixed(1)} ms`;
    statusEl.textContent = 'Done.';

    const imgData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        const r = (output[i] * 0.5 + 0.5) * 255;
        const g = (output[i + IMAGE_SIZE * IMAGE_SIZE] * 0.5 + 0.5) * 255;
        const b = (output[i + 2 * IMAGE_SIZE * IMAGE_SIZE] * 0.5 + 0.5) * 255;

        const idx = i * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
}

testBtn.addEventListener('click', runTest);
init();
