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

const CHANNEL_N = 16;
const GRID_SIZE = 64;

const models = {
  weak: { label: 'Weak model', weights: 'pipeline/1_grow/kiss_log/the_kiss/1200.weights.h5' },
  good: { label: 'Good model', weights: 'pipeline/2_improve/kiss_log/1500.weights.h5' },
  zoom: { label: 'Zoom transition model', weights: 'pipeline/3_transition/the_kiss_zoom_transition_from_old_pool/1500.weights.h5' }
};

let currentModelKey = 'weak';
let isPlaying = false;
let timerId = null;
let state, k1, b1, k2, b2;

const makeSeed = () => {
  const seed = tf.buffer([1, GRID_SIZE, GRID_SIZE, CHANNEL_N]);
  const c = Math.floor(GRID_SIZE / 2);
  for (let ch = 3; ch < CHANNEL_N; ch += 1) seed.set(1, 0, c, c, ch);
  return seed.toTensor();
};

function getLivingMask(x) {
  const alpha = x.slice([0, 0, 0, 3], [1, GRID_SIZE, GRID_SIZE, 1]);
  return tf.maxPool(alpha, 3, 1, 'same').greater(0.1);
}

function perceive(x) {
  const ident = tf.tensor2d([[0,0,0],[0,1,0],[0,0,0]]);
  const dx = tf.tensor2d([[ -1,0,1],[-2,0,2],[-1,0,1]]).div(8);
  const dy = dx.transpose();
  const basis = tf.stack([ident, dx, dy], -1).expandDims(2).tile([1,1,CHANNEL_N,1]);
  return tf.depthwiseConv2d(x, basis, 1, 'same');
}

function caStep() {
  tf.tidy(() => {
    const pre = getLivingMask(state);
    const y = perceive(state);
    const h = tf.relu(tf.add(tf.conv2d(y, k1, 1, 'same'), b1));
    const dx = tf.add(tf.conv2d(h, k2, 1, 'same'), b2);
    const update = tf.randomUniform([1, GRID_SIZE, GRID_SIZE, 1]).lessEqual(0.5).toFloat();
    let next = state.add(dx.mul(update));
    const post = getLivingMask(next);
    const life = pre.logicalAnd(post).toFloat();
    next = next.mul(life);
    state.dispose();
    state = next;
  });

  const rgba = tf.tidy(() => {
    const a = state.slice([0,0,0,3],[1,GRID_SIZE,GRID_SIZE,1]).clipByValue(0,1);
    const rgb = state.slice([0,0,0,0],[1,GRID_SIZE,GRID_SIZE,3]);
    return tf.onesLike(rgb).sub(a).add(rgb).clipByValue(0,1);
  });
  tf.browser.toPixels(rgba.squeeze(), frameEl);
  rgba.dispose();
}

function findWeights(datasets) {
  const byShape = (shape) => datasets.find(d => JSON.stringify(d.shape) === JSON.stringify(shape))?.value;
  return {
    k1: byShape([1, 1, 48, 128]),
    b1: byShape([128]),
    k2: byShape([1, 1, 128, 16]),
    b2: byShape([16])
  };
}

function walkGroup(group, out = []) {
  for (const key of Object.keys(group)) {
    const v = group[key];
    if (v && typeof v === 'object' && 'shape' in v && 'value' in v) out.push(v);
    else if (v && typeof v === 'object') walkGroup(v, out);
  }
  return out;
}

async function loadModelWeights(modelKey) {
  const res = await fetch(models[modelKey].weights);
  const buf = await res.arrayBuffer();
  const f = new hdf5.File(buf, modelKey + '.h5');
  const datasets = walkGroup(f.root);
  const w = findWeights(datasets);
  if (!w.k1 || !w.k2) throw new Error('Could not find NCA kernels in weights file');
  [k1, b1, k2, b2].forEach(t => t && t.dispose());
  k1 = tf.tensor(w.k1, [1,1,48,128]);
  b1 = tf.tensor(w.b1, [128]);
  k2 = tf.tensor(w.k2, [1,1,128,16]);
  b2 = tf.tensor(w.b2, [16]);
}

function pause() { isPlaying = false; if (timerId) clearInterval(timerId); }
function play() {
  if (isPlaying) return;
  isPlaying = true;
  timerId = setInterval(caStep, 1000 / (8 * Number(speedSelect.value)));
}
async function selectModel(key) {
  pause();
  currentModelKey = key;
  modelInfo.textContent = `Loading ${models[key].label}...`;
  await loadModelWeights(key);
  if (state) state.dispose();
  state = makeSeed();
  caStep();
  modelInfo.textContent = `${models[key].label} • Weights: ${models[key].weights} • Seed: centered alive cell`;
}

startBtn.addEventListener('click', async () => {
  homePage.classList.add('hidden');
  simPage.classList.remove('hidden');
  await selectModel('weak');
});
modelSelect.addEventListener('change', (e) => selectModel(e.target.value));
zoomBtn.addEventListener('click', () => selectModel('zoom'));
playBtn.addEventListener('click', play);
pauseBtn.addEventListener('click', pause);
restartBtn.addEventListener('click', () => selectModel(currentModelKey));
speedSelect.addEventListener('change', () => { if (isPlaying) { pause(); play(); } });
