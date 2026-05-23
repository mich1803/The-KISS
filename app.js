const homePage = document.getElementById('home-page');
const simPage = document.getElementById('sim-page');
const startBtn = document.getElementById('start-btn');
const modelSelect = document.getElementById('model-select');
const zoomBtn = document.getElementById('zoom-btn');
const playBtn = document.getElementById('play-btn');
const pauseBtn = document.getElementById('pause-btn');
const restartBtn = document.getElementById('restart-btn');
const speedSelect = document.getElementById('speed-select');
const frameEl = document.getElementById('frame');
const modelInfo = document.getElementById('model-info');

const buildFrames = (basePath, maxStep) => {
  const frames = [];
  for (let i = 0; i <= maxStep; i += 10) {
    const step = String(i).padStart(4, '0');
    frames.push(`${basePath}/${step}_pool.jpg`);
  }
  return frames;
};

const models = {
  weak: {
    label: 'Weak model',
    weights: 'pipeline/1_grow/kiss_log/the_kiss/1200.weights.h5',
    seed: 'pipeline/1_grow/kiss_log/the_kiss/0000_pool.jpg',
    frames: buildFrames('pipeline/1_grow/kiss_log/the_kiss', 1200)
  },
  good: {
    label: 'Good model',
    weights: 'pipeline/2_improve/kiss_log/1500.weights.h5',
    seed: 'pipeline/1_grow/kiss_log/the_kiss/0000_pool.jpg',
    frames: buildFrames('pipeline/1_grow/kiss_log/the_kiss', 1200).concat(
      buildFrames('pipeline/2_improve/kiss_log', 1500)
        .map(path => path.replace('_pool.jpg', '').replace(/\/$/, ''))
    )
  },
  zoom: {
    label: 'Zoom transition model',
    weights: 'pipeline/3_transition/the_kiss_zoom_transition_from_old_pool/1500.weights.h5',
    seed: 'pipeline/3_transition/the_kiss_zoom_transition_from_old_pool/0000_pool.jpg',
    frames: buildFrames('pipeline/3_transition/the_kiss_zoom_transition_from_old_pool', 1500)
  }
};

// fix good model paths to available finetune frames
models.good.frames = [];
for (let i = 0; i <= 1500; i += 100) {
  const step = String(i).padStart(4, '0');
  models.good.frames.push(`pipeline/2_improve/kiss_log/finetune_batch_${step}.jpg`);
}

let currentModelKey = 'weak';
let frameIndex = 0;
let isPlaying = false;
let timerId = null;

function updateInfo() {
  const m = models[currentModelKey];
  modelInfo.textContent = `${m.label} • Weights: ${m.weights} • Seed: ${m.seed}`;
}

function renderFrame() {
  const m = models[currentModelKey];
  frameEl.src = m.frames[Math.min(frameIndex, m.frames.length - 1)] || m.seed;
}

function step() {
  const m = models[currentModelKey];
  frameIndex = Math.min(frameIndex + 1, m.frames.length - 1);
  renderFrame();
  if (frameIndex >= m.frames.length - 1) pause();
}

function play() {
  if (isPlaying) return;
  isPlaying = true;
  const fps = 3 * Number(speedSelect.value);
  timerId = setInterval(step, 1000 / fps);
}

function pause() {
  isPlaying = false;
  if (timerId) clearInterval(timerId);
}

function restart() {
  pause();
  frameIndex = 0;
  renderFrame();
}

function selectModel(key) {
  currentModelKey = key;
  restart();
  updateInfo();
}

startBtn.addEventListener('click', () => {
  homePage.classList.add('hidden');
  simPage.classList.remove('hidden');
  selectModel('weak');
});

modelSelect.addEventListener('change', (e) => selectModel(e.target.value));
zoomBtn.addEventListener('click', () => selectModel('zoom'));
playBtn.addEventListener('click', play);
pauseBtn.addEventListener('click', pause);
restartBtn.addEventListener('click', restart);
speedSelect.addEventListener('change', () => {
  if (isPlaying) {
    pause();
    play();
  }
});
