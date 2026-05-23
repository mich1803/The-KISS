'use strict';

const CONFIG = {
  channelN: 16,
  targetSize: 64,
  hiddenN: 128,
  stepSize: 1.0,
  defaultFireRate: 0.5,
  preZoomGrowSteps: 180,
  models: {
    weak: {
      title: 'Weak model',
      url: 'assets/models/weak.json',
      source: 'pipeline/1_grow/kiss_log/the_kiss/1200.weights.h5'
    },
    good: {
      title: 'Good model',
      url: 'assets/models/good.json',
      source: 'pipeline/2_improve/kiss_log/1500.weights.h5'
    },
    zoom: {
      title: 'Zoom model',
      url: 'assets/models/zoom.json',
      source: 'pipeline/3_transition/the_kiss_zoom_transition_from_old_pool/1500.weights.h5'
    }
  }
};

const $ = (id) => document.getElementById(id);

const canvas = $('ncaCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: false });
ctx.imageSmoothingEnabled = false;

let state = null;
let activeModelKey = null;
let activeModel = null;
let modelCache = new Map();
let playing = false;
let rafId = null;
let stepCounter = 0;

function setStatus(text) { $('statusText').textContent = text; }
function setTitle(text) { $('activeModelTitle').textContent = text; }
function setStepCounter() { $('stepCounter').textContent = String(stepCounter); }

function setControlsEnabled(enabled) {
  $('playBtn').disabled = !enabled;
  $('pauseBtn').disabled = !enabled;
  $('restartBtn').disabled = !enabled;
  $('zoomBtn').disabled = !enabled;
}

function showHome() {
  pause();
  $('homeView').classList.remove('hidden');
  $('simView').classList.add('hidden');
}

function showSim() {
  $('homeView').classList.add('hidden');
  $('simView').classList.remove('hidden');
}

function createSeed(size = CONFIG.targetSize, channelN = CONFIG.channelN) {
  const x = new Float32Array(size * size * channelN);
  const center = Math.floor(size / 2);
  const base = (center * size + center) * channelN;
  for (let c = 3; c < channelN; c++) x[base + c] = 1.0;
  return x;
}

function clamp01(v) {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function render() {
  if (!state) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const size = CONFIG.targetSize;
  const C = CONFIG.channelN;
  const image = ctx.createImageData(size, size);
  const pixels = image.data;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const si = (y * size + x) * C;
      const pi = (y * size + x) * 4;
      const a = clamp01(state[si + 3]);
      const r = clamp01(1.0 - a + state[si + 0]);
      const g = clamp01(1.0 - a + state[si + 1]);
      const b = clamp01(1.0 - a + state[si + 2]);
      pixels[pi + 0] = Math.round(r * 255);
      pixels[pi + 1] = Math.round(g * 255);
      pixels[pi + 2] = Math.round(b * 255);
      pixels[pi + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function livingMask(x, size, C) {
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let xp = 0; xp < size; xp++) {
      let alive = false;
      for (let oy = -1; oy <= 1 && !alive; oy++) {
        const yy = y + oy;
        if (yy < 0 || yy >= size) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const xx = xp + ox;
          if (xx < 0 || xx >= size) continue;
          const a = x[(yy * size + xx) * C + 3];
          if (a > 0.1) { alive = true; break; }
        }
      }
      mask[y * size + xp] = alive ? 1 : 0;
    }
  }
  return mask;
}

function ncaStep(x, model, fireRate) {
  const size = model.targetSize || CONFIG.targetSize;
  const C = model.channelN || CONFIG.channelN;
  const HN = model.hiddenN || CONFIG.hiddenN;
  const W1 = model.W1;
  const b1 = model.b1;
  const W2 = model.W2;
  const b2 = model.b2;

  const preMask = livingMask(x, size, C);
  const next = new Float32Array(x);
  const perceptionN = C * 3;
  const p = new Float32Array(perceptionN);
  const hidden = new Float32Array(HN);
  const dx = new Float32Array(C);

  const sobelX = [
    -1/8, 0, 1/8,
    -2/8, 0, 2/8,
    -1/8, 0, 1/8
  ];
  const sobelY = [
    -1/8, -2/8, -1/8,
     0,    0,    0,
     1/8,  2/8,  1/8
  ];

  for (let y = 0; y < size; y++) {
    for (let xp = 0; xp < size; xp++) {
      p.fill(0);

      for (let c = 0; c < C; c++) {
        const centerValue = x[(y * size + xp) * C + c];
        let gx = 0;
        let gy = 0;
        let k = 0;
        for (let oy = -1; oy <= 1; oy++) {
          const yy = y + oy;
          for (let ox = -1; ox <= 1; ox++, k++) {
            const xx = xp + ox;
            if (yy < 0 || yy >= size || xx < 0 || xx >= size) continue;
            const v = x[(yy * size + xx) * C + c];
            gx += v * sobelX[k];
            gy += v * sobelY[k];
          }
        }
        const pi = c * 3;
        p[pi + 0] = centerValue;
        p[pi + 1] = gx;
        p[pi + 2] = gy;
      }

      for (let j = 0; j < HN; j++) {
        let sum = b1[j];
        for (let i = 0; i < perceptionN; i++) sum += p[i] * W1[i * HN + j];
        hidden[j] = sum > 0 ? sum : 0;
      }

      for (let c = 0; c < C; c++) {
        let sum = b2[c];
        for (let j = 0; j < HN; j++) sum += hidden[j] * W2[j * C + c];
        dx[c] = sum * CONFIG.stepSize;
      }

      if (Math.random() <= fireRate) {
        const base = (y * size + xp) * C;
        for (let c = 0; c < C; c++) next[base + c] = x[base + c] + dx[c];
      }
    }
  }

  const postMask = livingMask(next, size, C);
  for (let y = 0; y < size; y++) {
    for (let xp = 0; xp < size; xp++) {
      const alive = preMask[y * size + xp] && postMask[y * size + xp];
      if (!alive) {
        const base = (y * size + xp) * C;
        for (let c = 0; c < C; c++) next[base + c] = 0;
      }
    }
  }

  return next;
}

async function loadModel(key) {
  if (modelCache.has(key)) return modelCache.get(key);

  const spec = CONFIG.models[key];
  if (!spec) throw new Error(`Unknown model key: ${key}`);

  setStatus(`Loading ${spec.title}...`);
  const res = await fetch(spec.url, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Could not load ${spec.url}. Convert the .weights.h5 file first and place the JSON in assets/models/.`);
  }
  const raw = await res.json();

  const model = {
    name: raw.name || spec.title,
    source: raw.source || spec.source,
    targetSize: raw.target_size || raw.targetSize || CONFIG.targetSize,
    channelN: raw.channel_n || raw.channelN || CONFIG.channelN,
    hiddenN: raw.hidden_n || raw.hiddenN || CONFIG.hiddenN,
    W1: new Float32Array(raw.W1),
    b1: new Float32Array(raw.b1),
    W2: new Float32Array(raw.W2),
    b2: new Float32Array(raw.b2)
  };

  if (model.targetSize !== CONFIG.targetSize || model.channelN !== CONFIG.channelN) {
    console.warn('Model shape differs from CONFIG:', model);
  }

  modelCache.set(key, model);
  return model;
}

async function selectModel(key, reset = true) {
  pause();
  try {
    activeModel = await loadModel(key);
    activeModelKey = key;
    setTitle(activeModel.name || CONFIG.models[key].title);
    setStatus(`Loaded ${activeModel.name || CONFIG.models[key].title}. Source: ${activeModel.source || CONFIG.models[key].source}`);
    document.querySelectorAll('.model-card').forEach(card => {
      card.classList.toggle('active', card.dataset.model === key);
    });
    if (reset || !state) restart(false);
    setControlsEnabled(true);
    if (key !== 'zoom') $('zoomBtn').disabled = false;
    render();
  } catch (err) {
    console.error(err);
    setStatus(err.message);
    setControlsEnabled(false);
  }
}

function restart(shouldRender = true) {
  state = createSeed(CONFIG.targetSize, CONFIG.channelN);
  stepCounter = 0;
  setStepCounter();
  if (shouldRender) render();
}

function runSteps(n) {
  if (!state || !activeModel) return;
  const fireRate = Number($('fireRateSelect').value || CONFIG.defaultFireRate);
  for (let i = 0; i < n; i++) {
    state = ncaStep(state, activeModel, fireRate);
    stepCounter++;
  }
  setStepCounter();
}

function tick() {
  if (!playing) return;
  const speed = Number($('speedSelect').value || 4);
  runSteps(speed);
  render();
  rafId = requestAnimationFrame(tick);
}

function play() {
  if (!activeModel || !state) return;
  if (playing) return;
  playing = true;
  setStatus(`Running ${activeModel.name || activeModelKey}...`);
  rafId = requestAnimationFrame(tick);
}

function pause() {
  playing = false;
  if (rafId !== null) cancelAnimationFrame(rafId);
  rafId = null;
}

async function zoomOnFaces() {
  pause();
  try {
    if (!state) {
      const baseKey = modelCache.has('good') ? 'good' : 'weak';
      activeModel = await loadModel(baseKey);
      activeModelKey = baseKey;
      restart(false);
      for (let i = 0; i < CONFIG.preZoomGrowSteps; i++) {
        state = ncaStep(state, activeModel, CONFIG.defaultFireRate);
      }
      stepCounter = CONFIG.preZoomGrowSteps;
    }

    activeModel = await loadModel('zoom');
    activeModelKey = 'zoom';
    setTitle(activeModel.name || 'Zoom model');
    setStatus('Zoom model loaded. It starts from the current full-image state and zooms toward the faces.');
    document.querySelectorAll('.model-card').forEach(card => card.classList.remove('active'));
    $('zoomBtn').disabled = false;
    render();
    play();
  } catch (err) {
    console.error(err);
    setStatus(err.message);
  }
}

$('startBtn').addEventListener('click', showSim);
$('backBtn').addEventListener('click', showHome);
$('playBtn').addEventListener('click', play);
$('pauseBtn').addEventListener('click', pause);
$('restartBtn').addEventListener('click', () => { pause(); restart(true); setStatus('Restarted from the original training seed.'); });
$('zoomBtn').addEventListener('click', zoomOnFaces);

document.querySelectorAll('.model-card').forEach(card => {
  card.addEventListener('click', () => selectModel(card.dataset.model, true));
});

render();
