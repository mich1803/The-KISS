# The KISS - Klimt-Inspired Self-organizing System with Neural Cellular Automata

## 1. Project Identity

**Research question:**  
Can a single conditioned Neural Cellular Automata model learn to grow and transform low-resolution representations of multiple Klimt paintings through local cell interactions?

**Short description:**  
The KISS is a Neural Cellular Automata project where a single conditioned model learns to grow low-resolution versions of selected Gustav Klimt paintings from an initial seed state. The system must support image growth, transition between paintings, and local condition maps painted by the user. The final output is both a research-oriented training/evaluation pipeline and an interactive web app.

**Main topic areas:**  
- Artificial life  
- Image generation  
- Representation learning  
- Optimization for deep learning  
- Optional: quantization  

---

## 2. Core Concept

The system is based on a **single conditional Neural Cellular Automata model**.

Each cell/pixel does **not** have its own independent model.  
Instead, all cells share the same neural update rule.

Each cell has a state vector:

```text
[R, G, B, A, h1, h2, ..., hN]
```

Where:

- `R, G, B` are the visible color channels.
- `A` is the alpha/liveness channel.
- `h1...hN` are hidden channels used as internal memory.

At every simulation step, each cell observes a local neighborhood and updates its internal state using the shared neural update rule.

The model is conditioned by the selected painting.  
The condition can be:

1. **Global condition**: all cells receive the same painting condition.
2. **Local condition map**: different regions/cells receive different painting conditions.

---

## 3. Dataset Organization

The user manually prepares the painting images.  
No preprocessing code is required for image creation, but the project should include utility code to load and validate the dataset.

Required folder structure:

```text
paintings/
  data_sources.txt

  64/
    kiss.png
    adele.png
    tree_of_life.png
    judith.png
    danae.png
```

### 3.1 Image Requirements

Each image must be:

- PNG format.
- RGB or RGBA.
- Exactly `64x64` in `paintings/64/`.
- Named using lowercase snake_case.
- Manually selected and downsampled before training.

---

## 4. Main Technical Goals

The project must implement:

1. A single conditional NCA model.
2. Training on multiple Klimt paintings.
3. Support for `64x64` images only.
4. Support for local neighborhoods as a training hyperparameter:
   - `3x3` neighborhood: 8 surrounding cells.
   - `5x5` neighborhood: 24 surrounding cells.
5. Growth from an initial seed state.
6. Transition from one painting condition to another.
7. Optional local condition map painted by the user.
8. Real-time or near-real-time visualization in a web app.
9. Training metrics and animated GIF previews.
10. Optional quantization experiment.

---

## 5. Proposed repository Structure

```text
The_KISS/
  The_KISS_project.md # this file, don't touch it
  README.md # i'll write it once the project is finished
  TUTORIAL.md # write this to guide the user to the training and the web app start
  requirements.txt
  pyproject.toml
  .gitignore

  paintings/
    data_sources.txt
    64/

  src/
    ... list of .py utils ... # everything well commented

  train_evaluate.ipynb # train and evaluate models on a notebook, give also a gif preview at the end, used for finding the perfect hyperparameters

  train_script.py # train the single final model needed by the web app using webapp/model_config.yaml

  webapp/
    model_config.yaml # single final model configuration used by train_script.py and by the web app
    saved_model/
      ... trained model and metadata ...
    ... everything needed to the web app ...
```

### 5.1 Configuration Logic

The project has two different configuration levels:

1. **Notebook-level experimentation**  
   `train_evaluate.ipynb` allows the user to test different hyperparameters, including the neighborhood size (`3x3` or `5x5`).

2. **Final web app model**  
   `webapp/model_config.yaml` defines the single model that will be trained by `train_script.py` and loaded by the web app.

The web app does **not** allow model selection.  
It always loads the single model specified in:

```text
webapp/model_config.yaml
```

---

## 6. Model Design

### 6.1 Conditional Neural Cellular Automata

The model should support:

- Variable number of hidden channels.
- Variable neighborhood size during training and experimentation.
- Global painting condition.
- Local painting condition is never encountered in the training.
- Stochastic cell updates.
- CPU and GPU inference.

Suggested model interface:

```python
class ConditionalNCA(nn.Module):
    def __init__(
        self,
        state_channels: int,
        hidden_channels: int,
        num_paintings: int,
        condition_dim: int,
        neighborhood_size: int,
        update_rate: float,
    ):
        ...

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        steps: int = 1,
        local_condition_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

### 6.2 State Channels

Default:

```text
RGB channels: 3
Alpha/liveness channel: 1
Hidden channels: 12
```

### 6.3 Condition Embedding

Each painting receives an ID:

```text
the_kiss -> 0
adele_bloch_bauer -> 1
tree_of_life -> 2
judith -> 3
danae -> 4
```

The model uses an embedding layer.

### 6.4 Global Condition

In Living Canvas mode, the whole canvas receives the same painting condition.

Example:

```text
condition = "the_kiss"
```

All cells are guided toward the same target painting.

### 6.5 Local Condition Map

In Paint the Genome mode, each cell may receive a different painting condition.

Example:

```text
left side: The Kiss
right side: Tree of Life
center: Adele Bloch-Bauer
```

The local condition map can be represented as:

```text
H x W integer map
```

---

## 7. Neighborhood Experiments

The model must support two neighborhood sizes during notebook experimentation:

### 7.1 3x3 Neighborhood

Each cell observes the 8 surrounding cells.

```text
x x x
x c x
x x x
```

This is the default and fastest mode.

### 7.2 5x5 Neighborhood

Each cell observes the 24 surrounding cells.

```text
x x x x x
x x x x x
x x c x x
x x x x x
x x x x x
```

This is slower but may improve global structure.

### 7.3 Research Hypothesis

Larger neighborhoods may improve global structural reconstruction but increase computational cost and may reduce the local self-organizing character of the model.

Suggested comparison inside `train_evaluate.ipynb`:

```text
64x64, 3x3
64x64, 5x5
```

The final web app does not expose this option.  
The selected neighborhood is fixed in `webapp/model_config.yaml`.

---

## 8. Training Objectives

The training system should support multiple objectives.

### 8.1 Growth Objective

The model starts from a seed state and grows toward a selected painting target.

```text
seed state -> NCA steps -> target painting
```

Loss:

```text
L_grow = MSE(output_rgb, target_rgb)
```

Optional:

```text
L_grow = L1(output_rgb, target_rgb) + MSE(output_rgb, target_rgb)
```

### 8.2 Persistence Objective

After reaching the target, the model should remain stable.

```text
target-like state -> more NCA steps -> still target-like
```

Loss:

```text
L_persist = MSE(output_after_extra_steps, target_rgb)
```

### 8.3 Transition Objective

The model starts from one painting condition and then switches to another.

```text
painting A state -> condition switch -> painting B target
```

Loss:

```text
L_transition = MSE(transformed_rgb, target_B_rgb)
```

### 8.4 Full Loss

Suggested full loss:

```text
L_total = 
    L_grow 
  + lambda_persist * L_persist
  + lambda_transition * L_transition
```

Start with:

```text
lambda_persist = 0.5
lambda_transition = 0.25
```

These values should be configurable.

---

## 9. Training Notebook

The main interactive training notebook must be:

```text
train_evaluate.ipynb
```

This notebook is used to explore hyperparameters and decide the final model configuration.

This notebook must allow the user to choose:

- Image resolution:
  - `64`
- Paintings to include.
- Neighborhood size:
  - `3`
  - `5`
- Number of hidden channels.
- Condition embedding size.
- Number of training iterations.
- Batch size.
- Number of NCA steps.
- Learning rate.
- Device:
  - CPU
  - CUDA if available

### 9.1 Notebook Outputs

The notebook must show:

- Training loss curves.
- Current generated preview.
- Target image.
- Growth animation.
- Transition animation if enabled.
- Metrics table.
- Saved checkpoint path.
- Saved GIF path.

### 9.2 GIF Preview

The notebook must save animated previews to:

```text
experiments/gifs/
```

Examples:

```text
experiments/gifs/kiss_64_3x3_growth.gif
experiments/gifs/kiss_to_tree_64_3x3_transition.gif
```

### 9.3 Final Configuration Export

After finding good hyperparameters, the user manually writes or updates:

```text
webapp/model_config.yaml
```

This file defines the only model that will be trained for the web app.

---

## 10. Scripted Training for the Final Web App Model

After choosing the best parameters using the notebook, the project must include a script that trains and saves the single model needed by the web app:

```text
train_script.py
```

This script must use the configuration stored in:

```text
webapp/model_config.yaml
```

Suggested CLI:

```bash
python train_script.py --config webapp/model_config.yaml
```

The script should save:

```text
webapp/saved_model/
  kiss_nca.pt
  metadata.json
```

The metadata must include:

```json
{
  "model_name": "kiss_nca",
  "resolution": 64,
  "neighborhood_size": 3,
  "state_channels": 16,
  "hidden_channels": 16,
  "condition_dim": 16,
  "paintings": [
    "the_kiss",
    "adele_bloch_bauer",
    "tree_of_life"
  ],
  "training_objectives": {
    "growth": true,
    "persistence": true,
    "transition": true
  }
}
```

### 10.1 Example `webapp/model_config.yaml`

```yaml
model_name: kiss_nca
resolution: 64
neighborhood_size: 3
state_channels: 16
hidden_channels: 16
condition_dim: 16
update_rate: 0.5

paintings:
  - the_kiss
  - adele_bloch_bauer
  - tree_of_life
  - judith
  - danae

training:
  iterations: 5000
  batch_size: 8
  steps_min: 48
  steps_max: 96
  learning_rate: 0.001
  lambda_persist: 0.5
  lambda_transition: 0.25
  use_growth: true
  use_persistence: true
  use_transition: true

device: auto

output:
  model_path: webapp/saved_model/kiss_nca.pt
  metadata_path: webapp/saved_model/metadata.json
```

---

## 11. Web App

All web app files must be placed in:

```text
webapp/
```

Recommended first implementation:

```text
Gradio
```

Alternative advanced implementation:

```text
FastAPI backend + HTML/Canvas or React frontend
```

Start with Gradio unless the interface becomes too limited.

The web app loads only one model:

```text
webapp/saved_model/kiss_nca.pt
```

The web app reads model metadata/configuration from:

```text
webapp/model_config.yaml
webapp/saved_model/metadata.json
```

The web app must not expose a model selector.

---

## 12. Web App Modes

The web app must support two main modes.

---

# Mode 1 — Living Canvas

### 12.1 User Flow

The user must be able to:

1. Load the single trained model defined by `webapp/model_config.yaml`.
2. Select the initial painting.
3. Press Play.
4. Watch the painting grow step by step.
5. Watch real-time metrics.
6. Pause at any moment.
7. Change the target painting.
8. Continue simulation and observe transition.
9. Reset to initial seed state.

### 12.2 Required Controls

```text
Current target painting:
  - The Kiss
  - Adele Bloch-Bauer
  - Tree of Life
  - Judith
  - Danae

Buttons:
  - Play
  - Pause
  - Step
  - Reset
```



### 12.3 Metrics in Living Canvas Mode

Since the target painting is known, metrics can be computed directly against the current selected target.

Real-time metrics:

- MSE to current target.
- L1 to current target.
- SSIM to current target.
- Alpha/liveness mean.
- Stability delta:
  - difference between current frame and previous frame.
- FPS.

For transition:

- MSE to source painting.
- MSE to target painting.
- Transition progress:

```text
transition_progress = mse_to_source / (mse_to_source + mse_to_target)
```

Interpretation:

- Close to 0: still similar to source.
- Close to 1: closer to target.

---

# Mode 2 — Paint the Genome

## **Paint the Genome**

This mode allows the user to paint local conditions on the grid.

Different areas of the canvas can be assigned to different Klimt paintings.

### 12.4 User Flow

The user must be able to:

1. Use the single trained model defined by `webapp/model_config.yaml`.
2. Choose brush condition or Random:
   - The Kiss
   - Adele Bloch-Bauer
   - Tree of Life
   - Judith
   - Danae
3. If Random is selected, choose a percentage for each painting.
4. Paint on a blank condition map.
5. See a legend:
   - color -> painting
6. Press Play.
7. Watch the cellular system grow according to the local condition map.
8. Pause, repaint conditions, and continue.
9. Reset state.
10. Save output image or GIF.

### 12.5 Condition Map Implementation

The app should maintain:

```text
condition_map: H x W integer tensor
```

Each cell stores a painting ID.

The NCA model receives:

```text
state
condition_map
```

At each step, each cell receives the embedding corresponding to its local painting ID.

### 12.6 Metrics in Paint the Genome Mode

There is no single target image, because the canvas is locally conditioned.

Use a synthetic local target:

For each pixel, the target color is taken from the painting assigned to that pixel in the condition map.

Example:

```text
if condition_map[y, x] == the_kiss:
    local_target[y, x] = the_kiss_target[y, x]

if condition_map[y, x] == tree_of_life:
    local_target[y, x] = tree_of_life_target[y, x]
```

Then compute:

- Local MSE against synthetic target.
- Local SSIM against synthetic target.
- Region-wise MSE per painting.
- Boundary instability:
  - average difference across borders between different conditions.
- Diversity/entropy of condition map.
- FPS.

### 12.7 Region-wise Metrics

For each painting condition:

```text
MSE_region[p] = MSE(output pixels where condition_map == p, target_p pixels)
```

Display:

```text
The Kiss region MSE: ...
Tree of Life region MSE: ...
Adele region MSE: ...
```

### 12.8 Boundary Metric

The boundary metric measures instability or conflict between adjacent regions with different conditions.

Simple version:

```text
boundary_score = average RGB difference across neighboring cells with different condition IDs
```

This can be interpreted carefully:

- High boundary score may indicate strong visual separation.
- Low boundary score may indicate smoother blending.
- It is not automatically good or bad; it describes behavior.

---

## 13. Suggested Hyperparameters

Start with:

```text
resolution = 64
neighborhood_size = 3
state_channels = 16
hidden_channels = 64
condition_dim = 16
update_rate = 0.5
steps_min = 48
steps_max = 96
batch_size = 8
learning_rate = 1e-3
optimizer = Adam
iterations = 5000
```

For 5x5 notebook experiments:

```text
neighborhood_size = 5
hidden_channels = 64
batch_size = 4 or 8
```

The selected final values must be written into:

```text
webapp/model_config.yaml
```

