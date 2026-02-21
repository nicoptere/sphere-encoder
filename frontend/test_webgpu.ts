import { WebGPUEngine } from './src/webgpu/Engine';

const statusEl = document.getElementById('status')!;
const canvas = document.getElementById('webgpuCanvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
const testBtn = document.getElementById('testBtn') as HTMLButtonElement;
const timeEl = document.getElementById('webgpuTime')!;

const isSecureEl = document.getElementById('isSecure')!;
const hasGpuEl = document.getElementById('hasGpu')!;
const userAgentEl = document.getElementById('userAgent')!;
const adapterInfoEl = document.getElementById('adapterInfo')!;
const limitsInfoEl = document.getElementById('limitsInfo')!;
const featuresListEl = document.getElementById('featuresList')!;

let engine: WebGPUEngine | null = null;
let IMAGE_SIZE = 64;
let LATENT_DIM = 512;

function log(msg: string) {
    statusEl.textContent += msg + '\n';
    console.log(msg);
}

async function runDiagnostics() {
    statusEl.textContent = ''; // Clear
    log('--- System Diagnostics ---');

    // Environment
    const isSecure = window.isSecureContext;
    isSecureEl.textContent = isSecure ? 'YES' : 'NO';
    if (!isSecure) {
        document.getElementById('envCard')?.classList.add('error');
    }

    const hasGpu = !!navigator.gpu;
    hasGpuEl.textContent = hasGpu ? 'YES' : 'NO';
    if (!hasGpu) {
        document.getElementById('envCard')?.classList.add('error');
    }

    userAgentEl.textContent = navigator.userAgent;

    if (!hasGpu) {
        log('CRITICAL: navigator.gpu is undefined.');
        if (!isSecure) {
            log('WebGPU requires a SECURE CONTEXT (HTTPS). Accessing via HTTP or an insecure IP will hide WebGPU.');
        } else {
            log('Secure context detected, but WebGPU is still missing. This device/browser might not support it, or you may need to enable flags.');
        }
        return;
    }

    // Adapter
    log('Requesting WebGPU Adapter...');
    try {
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!adapter) {
            log('ERROR: No WebGPU Adapter found.');
            adapterInfoEl.innerHTML = '<div style="color: #ef4444">No adapter returned for request.</div>';
            document.getElementById('adapterCard')?.classList.add('error');
            return;
        }

        // Adapter Info
        const info = await adapter.requestAdapterInfo();
        adapterInfoEl.innerHTML = `
            <div class="info-row"><span class="info-label">Vendor</span><span class="info-value">${info.vendor || 'Unknown'}</span></div>
            <div class="info-row"><span class="info-label">Architecture</span><span class="info-value">${info.architecture || 'Unknown'}</span></div>
            <div class="info-row"><span class="info-label">Description</span><span class="info-value" style="font-size: 0.7rem">${info.description || 'Unknown'}</span></div>
        `;
        document.getElementById('adapterCard')?.classList.add('success');

        // Device & Limits
        log('Requesting WebGPU Device...');
        const device = await adapter.requestDevice();

        // Limits
        let limitsHtml = '';
        const importantLimits = [
            'maxStorageBufferBindingSize',
            'maxComputeWorkgroupStorageSize',
            'maxComputeInvocationsPerWorkgroup',
            'maxTextureDimension2D'
        ];

        for (const limit of importantLimits) {
            const val = device.limits[limit as keyof GPUDeviceLimits];
            limitsHtml += `<div class="info-row"><span class="info-label">${limit}</span><span class="info-value">${val}</span></div>`;
        }
        limitsInfoEl.innerHTML = limitsHtml;

        // Features
        let featuresHtml = '';
        device.features.forEach(feature => {
            featuresHtml += `<span class="tag">${feature}</span>`;
        });
        featuresListEl.innerHTML = featuresHtml || '<span class="info-muted">No extensions enabled</span>';

        log('WebGPU initialized successfully.');

        // Inference Engine
        log('Loading Inference Engine...');
        engine = new WebGPUEngine();
        await engine.init('models/ffhq_64');

        const metadata = engine.getMetadata()!;
        IMAGE_SIZE = metadata.image_size;
        LATENT_DIM = metadata.latent_dim;

        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;

        testBtn.disabled = false;
        log('Full engine ready.');

    } catch (err) {
        log(`CRITICAL ERROR during initialization: ${err}`);
        console.error(err);
    }
}

async function runTest() {
    if (!engine) return;

    const latent = new Float32Array(LATENT_DIM);
    let sumSq = 0;
    for (let i = 0; i < LATENT_DIM; i++) {
        const z = Math.sqrt(-2.0 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2.0 * Math.PI * Math.random());
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
runDiagnostics();
