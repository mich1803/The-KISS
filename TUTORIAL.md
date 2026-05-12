# The KISS Deep-Learning Tutorial

This guide covers the deep-learning workflow for the Conditional Neural Cellular Automata (NCA) side of the project. The web app will be documented later.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local package imports without installation, run scripts with:

```bash
PYTHONPATH=src python train_script.py --help
```

## 2. Validate the painting dataset

The training code expects manually prepared `64x64` PNG files in `paintings/64`. Logical painting names are:

- `the_kiss`
- `adele_bloch_bauer`
- `tree_of_life`
- `judith`
- `danae`

The loader accepts the checked-in aliases `the_kiss.png` and `adele_bloch_bauer.png`, while also supporting the shorter names described in the project brief.

```bash
PYTHONPATH=src python - <<'PY'
from kiss_nca.dataset import validate_dataset
print(validate_dataset('paintings/64'))
PY
```

## 3. Explore hyperparameters in the notebook

Open:

```text
train_evaluate.ipynb
```

The notebook lets you change:

- resolution (`64` only)
- included paintings
- neighborhood size (`3` or `5`)
- state/update hidden channels
- condition embedding dimension
- iterations, batch size, NCA rollout steps, learning rate
- CPU or CUDA device
- growth, persistence, and transition objectives

Notebook outputs include loss curves, generated-versus-target previews, metrics tables, checkpoints, growth GIFs, and transition GIFs under `experiments/gifs/`.

## 4. Choose final web-app model settings

After experiments, write the chosen settings into:

```text
webapp/model_config.yaml
```

The future web app will not expose model selection; it will load the single model defined by this file.

## 5. Train the final model

Run:

```bash
PYTHONPATH=src python train_script.py --config webapp/model_config.yaml
```

For a quick smoke test, override the training length:

```bash
PYTHONPATH=src python train_script.py --config webapp/model_config.yaml --iterations 1
```

The final script saves:

```text
webapp/saved_model/kiss_nca.pt
webapp/saved_model/metadata.json
```

The metadata records model architecture, resolution, selected paintings, source files, and enabled objectives.
