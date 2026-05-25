# The KISS  
## Klimt-Inspired Self-organizing System with Neural Cellular Automata

**The KISS** is a Klimt-inspired self-organizing image generation project based on **Neural Cellular Automata** (NCA).  
The project explores how a small local neural rule, repeatedly applied on a 2D grid, can grow, refine, persist, and transform an image starting from a minimal seed state.

The project is inspired by the Distill article **Growing Neural Cellular Automata** by Mordvintsev et al. and extends the idea from simple emoji-like targets to a more demanding artistic reconstruction task inspired by Gustav Klimt's *The Kiss*.

The final system is organized as a three-stage NCA pipeline:

```text
seed
  ↓
Model 1: Grow
  ↓
Model 2: Improve / Persist
  ↓
Model 3: Transition / Zoom
```

A browser-based static web application is also included and hosted through GitHub Pages:

**Web demo:**  
https://mich1803.github.io/The-KISS/

---

## Project Idea

Neural Cellular Automata are differentiable dynamical systems where each cell in a grid stores a vector state and updates itself using only local information from nearby cells.

Instead of generating an image in one forward pass, an NCA learns a repeated local rule:

```text
local perception → neural update → new cell state
```

When applied many times, this local rule can produce global image-like structures.

In this project, the goal is to train NCAs that can:

1. grow a Klimt-inspired target image from a single active seed;
2. improve visual detail while preserving the ability to grow;
3. transition from a grown full image to a zoomed version focused on the faces;
4. run interactively in the browser as a real simulation.

---

## Relation to Growing Neural Cellular Automata

The project builds on:

> Mordvintsev, Randazzo, Niklasson, and Levin,  
> **Growing Neural Cellular Automata**, Distill, 2020.  
> https://distill.pub/2020/growing-ca/

The original work demonstrated that NCAs can grow emoji-like targets from a single seed and learn persistence/regeneration.

This project keeps the core idea but changes the target and training pipeline:

- the target is not a simple icon, but a more complex artistic image;
- visual detail and chromatic texture are more important;
- training must balance growth, reconstruction, and long-term persistence;
- the final system is split into multiple specialized models instead of relying on one single automaton.

---

## Repository Contents

The repository contains:

```text
The-KISS/
│
├── pipeline/
│   ├── 1_grow/
│   │   └── training notebook, logs, checkpoints for the first grow model
│   │
│   ├── 2_improve/
│   │   └── fine-tuning notebook, improved checkpoints, loss logs
│   │
│   └── 3_transition/
│       └── transition notebook and zoom-transition checkpoints
│
├── TheKISS_report.pdf
│
└── README.md
```

The repository includes the full Colab training notebooks, exported checkpoint models, and the browser application used to simulate the trained automata.

---

## Neural Cellular Automata Model

Each automaton state is a tensor:

```text
x ∈ R^(B × H × W × C)
```

where:

- `B` is the batch size;
- `H` and `W` are the spatial dimensions;
- `C` is the number of cell state channels.

In the main experiments:

```text
H = 64
W = 64
C = 16
```

The first four channels are interpreted as a premultiplied RGBA image:

```text
channels 0, 1, 2 → RGB
channel 3       → alpha
```

The remaining channels are hidden memory channels used internally by the automaton.

---

## Seed State

The model starts from a single living cell.

The initial state is zero everywhere except at the center of the grid:

```python
seed = np.zeros([H, W, CHANNEL_N], np.float32)
seed[H//2, W//2, 3:] = 1.0
```

This means that the alpha channel and all hidden channels from index `3` onward are activated only at the center.

The full image must grow from this minimal seed.

---

## Local Perception

Each cell perceives local information through convolutional filters.

The basic perception includes:

- identity;
- horizontal Sobel derivative;
- vertical Sobel derivative.

Some variants also include a Laplacian-like cue to improve detail sensitivity.

Conceptually:

```text
cell state
  ↓
identity / dx / dy / laplacian filters
  ↓
local perceived features
  ↓
1×1 neural update network
  ↓
state increment
```

The update rule is shared by all cells and applied repeatedly.

---

## Stochastic Cell Updates

The automata use a stochastic update mask controlled by the fire rate.

In most experiments:

```text
fire_rate = 0.5
```

This means that at each step, only a random subset of cells updates.

This makes the system more robust and less dependent on perfectly synchronous updates.

---


# Model 1: Grow

The first model is trained to grow the Klimt-inspired target from the seed state.

## Goal

The goal of Model 1 is:

```text
seed → full target image
```

The model must learn both global structure and local visual features.

## Main Training Parameters

Typical settings:

```text
target size      = 64 × 64
channels         = 16
batch size       = 8
pool size        = 1024
fire rate        = 0.5
target padding   = 0
rollout length   = U(64, 256)
```

## Pattern Pool

Training uses a pattern pool, as in the original Growing NCA work.

The pool stores many intermediate automaton states.  
At each training iteration:

1. a batch is sampled from the pool;
2. samples are ranked by current loss;
3. part of the batch is reset to the seed;
4. the model is rolled forward for a random number of steps;
5. the resulting states are written back into the pool.

This trains the model to both grow and persist.

## Batch Composition

For Model 1, each batch combines:

```text
seed samples + pool samples
```

The seed samples preserve the ability to grow from scratch.  
The pool samples train persistence and refinement.

A typical batch with size 8 is:

```text
4 seed states
4 pool states
```

## Loss Function

The loss combines:

- a mid-rollout loss;
- a final rollout loss;
- RGB reconstruction;
- alpha reconstruction;
- detail/gradient loss.

The total loss has the form:

```text
L = w_mid L_mid + w_final L_final
```

where:

```text
w_mid   = 0.2
w_final = 0.8
```

The pixel reconstruction loss separates RGB and alpha:

```text
L_pix = RGB_W · ||RGB - RGB_target||² + A_W · ||alpha - alpha_target||²
```

A key tuning decision was to assign more importance to RGB channels than to alpha:

```text
RGB_W = 2.5
A_W   = 0.5
```

This was important because early models could reproduce the coarse silhouette but lacked chromatic detail.

## Training Observations

Initial experiments using parameters close to the original emoji-based NCA setup produced poor detail.

After increasing the RGB weight and using longer rollouts, the model produced a recognizable reconstruction.  
However, training for too long caused a failure mode:

```text
the model became good at maintaining already formed states,
but forgot how to grow from the seed.
```

This made it necessary to stop at the last acceptable growing checkpoint.

The selected Model 1 checkpoint is used as the starting point for the improvement stage.

---

# Model 2: Improve / Persist

Model 2 improves the visual quality and long-term persistence of Model 1.

Unlike Model 1, this stage is not trained from a normal self-generated pool.  
Instead, the old grow model is frozen and used as a teacher to generate inputs.

## Goal

The goal of Model 2 is:

```text
old grow-model state → more detailed and more persistent state
```

while preserving the ability to start from the same seed pipeline.

## Teacher-Generated Inputs

The old grow model is loaded from a selected checkpoint and frozen.

For each training batch:

```text
seed
  ↓
frozen old model for N steps
  ↓
input state for the new model
```

where `N` is randomly sampled, for example:

```text
N ∈ [64, 256]
```

The student model then starts from these old-model states.

This avoids the problem of the new model poisoning its own pool with unstable states.

## Fine-Tuning Settings

Typical settings:

```text
teacher input steps = U(64, 256)
student rollout     = U(128, 512)
fire rate           = 0.5
RGB weight          = 3.0
alpha weight        = 0.35
detail weight       = 0.20
```

The longer rollout teaches persistence.

## Additional Regularization

Model 2 uses two regularization terms:

### 1. Standard L2 parameter regularization

This penalizes large weights:

```text
L2_param = ||θ||²
```

Typical weight:

```text
1e-6
```

### 2. Anchor regularization

This keeps the student close to the original grow model:

```text
L_anchor = ||θ_student - θ_teacher||²
```

Typical weight:

```text
1e-5
```

The anchor prevents the fine-tuned model from drifting too far from the original growth dynamics.

## Result

Model 2 gives better detail and more stable long-horizon behavior while preserving the growth pathway from the seed.

---

# Model 3: Transition / Zoom

Model 3 is a new model trained from scratch, not a fine-tuned version of the previous one.

## Goal

The goal of Model 3 is to learn a transition:

```text
full image state → zoomed target image
```

The model starts from states generated by Model 2 and transforms them into a second target, focused on a zoomed region of the image.

## Training Input Pool

The initial training pool is generated by rolling the improved model from the seed:

```text
seed
  ↓
Model 2
  ↓
full image state
  ↓
Model 3 training input
```

The transition model is then trained to map this distribution to the zoomed target.

## Training Strategy

During training, part of the batch can be refreshed with new states generated by Model 2.  
This keeps Model 3 tied to the real distribution produced by the previous model.

Typical settings:

```text
old model input steps = U(48, 256)
transition rollout    = U(64, 512)
batch size            = 8
fire rate             = 0.5
```

The transition model uses RGB, alpha, detail, and regularization losses.

## Result

Model 3 learns a state-to-state transformation:

```text
seed → Model 2 full image → Model 3 zoomed faces
```

This creates a pipeline of self-organizing image stages.

---

## Web Application

The repository includes a static web application that runs the trained NCA models directly in the browser.

The app is available at:

https://mich1803.github.io/The-KISS/

The web app allows the user to:

- start the automaton from the same seed used during training;
- choose between the weak and improved models;
- play and pause the simulation;
- restart from seed;
- change simulation speed;
- activate the zoom transition model;
- observe the NCA evolve in real time.

---


## Training Notebooks

The project includes Colab notebooks for each stage.

### Model 1: Grow

Notebook for training the first grow model from the seed.

Main purpose:

```text
seed → Klimt-inspired target
```

### Model 2: Improve

Notebook for teacher-driven fine-tuning.

Main purpose:

```text
old grow model output → improved detail and persistence
```

### Model 3: Transition

Notebook for training a new transition model from an old-model generated pool.

Main purpose:

```text
Model 2 generated state → zoomed target
```

Each notebook includes:

- model definition;
- seed initialization;
- training loop;
- loss functions;
- checkpoint export;
- qualitative visualizations;
- loss plots;
- generated videos.

---


## Main Lessons

The central empirical finding is that NCA training is a balance between:

```text
growth
detail
persistence
```

Optimizing only for reconstruction loss can produce visually good states but may destroy the ability to grow from the seed.

In practice:

- increasing RGB loss improves visual quality;
- detail/gradient loss helps reduce blur;
- longer rollouts improve persistence;
- too much persistence training can cause seed-growth forgetting;
- teacher-generated inputs help stabilize fine-tuning;
- transition models can be trained from old-model-generated state distributions.

---


## References

The project is based on and inspired by:

- Mordvintsev, A., Randazzo, E., Niklasson, E., and Levin, M.  
  **Growing Neural Cellular Automata.**  
  Distill, 2020.  
  https://distill.pub/2020/growing-ca

- Mordvintsev, A., Randazzo, E., Niklasson, E., Levin, M., and Greydanus, S.  
  **Differentiable Self-organizing Systems.**  
  Distill, 2020.  
  https://distill.pub/2020/selforg

- Niklasson, E., Mordvintsev, A., Randazzo, E., and Levin, M.  
  **Self-Organising Textures.**  
  Distill, 2021.  
  https://distill.pub/selforg/2021/textures

---


## Author

**Michele Magrini**  
Applied Mathematics
Deep Learning and Applied AI 2026  
Sapienza University of Rome
