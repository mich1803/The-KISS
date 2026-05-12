"""Training, evaluation, checkpoint, and preview utilities for The KISS."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .dataset import PaintingRecord
from .model import ConditionalNCA, make_seed


@dataclass
class LossBreakdown:
    """Scalar loss terms for logging."""

    total: float
    growth: float
    persistence: float
    transition: float


def sample_steps(min_steps: int, max_steps: int) -> int:
    """Sample an inclusive random NCA rollout length."""

    return random.randint(min_steps, max_steps)


def clamp_rgb(state: torch.Tensor) -> torch.Tensor:
    """Return visible RGB channels clamped to image range."""

    return state[:, :3].clamp(0.0, 1.0)


def training_step(
    model: ConditionalNCA,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    steps_min: int,
    steps_max: int,
    lambda_persist: float = 0.5,
    lambda_transition: float = 0.25,
    use_growth: bool = True,
    use_persistence: bool = True,
    use_transition: bool = True,
) -> LossBreakdown:
    """Run one optimization step across growth, persistence, and transition losses."""

    device = targets.device
    resolution = targets.shape[-1]
    num_paintings = targets.shape[0]
    condition = torch.randint(0, num_paintings, (batch_size,), device=device)
    target = targets[condition]
    state = make_seed(batch_size, model.state_channels, resolution=resolution, device=device)

    total = torch.zeros((), device=device)
    growth = torch.zeros((), device=device)
    persistence = torch.zeros((), device=device)
    transition = torch.zeros((), device=device)

    if use_growth or use_persistence or use_transition:
        grown = model(state, condition, steps=sample_steps(steps_min, steps_max))
    else:
        raise ValueError("At least one training objective must be enabled.")

    if use_growth:
        rgb = clamp_rgb(grown)
        growth = F.l1_loss(rgb, target) + F.mse_loss(rgb, target)
        total = total + growth

    if use_persistence:
        persisted = model(grown.detach(), condition, steps=max(1, steps_min // 4))
        persistence = F.mse_loss(clamp_rgb(persisted), target)
        total = total + lambda_persist * persistence

    if use_transition and num_paintings > 1:
        new_condition = (condition + torch.randint(1, num_paintings, (batch_size,), device=device)) % num_paintings
        new_target = targets[new_condition]
        transitioned = model(grown.detach(), new_condition, steps=sample_steps(steps_min, steps_max))
        transition = F.mse_loss(clamp_rgb(transitioned), new_target)
        total = total + lambda_transition * transition

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return LossBreakdown(
        total=float(total.detach().cpu()),
        growth=float(growth.detach().cpu()),
        persistence=float(persistence.detach().cpu()),
        transition=float(transition.detach().cpu()),
    )


def train_model(
    model: ConditionalNCA,
    targets: torch.Tensor,
    training_config: dict[str, Any],
) -> list[dict[str, float]]:
    """Train a conditional NCA and return per-iteration scalar history."""

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    iterations = int(training_config["iterations"])
    history: list[dict[str, float]] = []
    progress = tqdm(range(iterations), desc="Training KISS NCA")
    for iteration in progress:
        losses = training_step(
            model=model,
            targets=targets,
            optimizer=optimizer,
            batch_size=int(training_config["batch_size"]),
            steps_min=int(training_config["steps_min"]),
            steps_max=int(training_config["steps_max"]),
            lambda_persist=float(training_config.get("lambda_persist", 0.5)),
            lambda_transition=float(training_config.get("lambda_transition", 0.25)),
            use_growth=bool(training_config.get("use_growth", True)),
            use_persistence=bool(training_config.get("use_persistence", True)),
            use_transition=bool(training_config.get("use_transition", True)),
        )
        row = {"iteration": float(iteration + 1), **losses.__dict__}
        history.append(row)
        progress.set_postfix(loss=f"{losses.total:.4f}")
    return history


@torch.no_grad()
def rollout_frames(
    model: ConditionalNCA,
    condition_id: int,
    resolution: int = 64,
    total_steps: int = 96,
    every: int = 4,
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Generate RGB frames from a centered seed for GIF previews."""

    model.eval()
    state = make_seed(1, model.state_channels, resolution=resolution, device=device)
    condition = torch.tensor([condition_id], device=device)
    frames: list[torch.Tensor] = []
    for step in range(total_steps + 1):
        if step % every == 0:
            frames.append(clamp_rgb(state)[0].detach().cpu())
        if step < total_steps:
            state = model(state, condition, steps=1)
    return frames


@torch.no_grad()
def transition_frames(
    model: ConditionalNCA,
    source_id: int,
    target_id: int,
    resolution: int = 64,
    grow_steps: int = 64,
    transition_steps: int = 64,
    every: int = 4,
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Generate frames that grow one condition and then switch to another."""

    model.eval()
    state = make_seed(1, model.state_channels, resolution=resolution, device=device)
    source = torch.tensor([source_id], device=device)
    target = torch.tensor([target_id], device=device)
    frames: list[torch.Tensor] = []
    for step in range(grow_steps):
        if step % every == 0:
            frames.append(clamp_rgb(state)[0].detach().cpu())
        state = model(state, source, steps=1)
    for step in range(transition_steps + 1):
        if step % every == 0:
            frames.append(clamp_rgb(state)[0].detach().cpu())
        if step < transition_steps:
            state = model(state, target, steps=1)
    return frames


def save_gif(frames: list[torch.Tensor], path: str | Path, duration: float = 0.08) -> Path:
    """Save ``[3, H, W]`` float tensors as an animated GIF."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frames:
        array = (frame.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255).astype("uint8")
        images.append(array)
    imageio.mimsave(output_path, images, duration=duration)
    return output_path


def build_metadata(config: dict[str, Any], records: list[PaintingRecord]) -> dict[str, Any]:
    """Build metadata saved next to the final checkpoint."""

    training = config["training"]
    return {
        "model_name": config["model_name"],
        "resolution": config["resolution"],
        "neighborhood_size": config["neighborhood_size"],
        "state_channels": config["state_channels"],
        "hidden_channels": config["hidden_channels"],
        "condition_dim": config["condition_dim"],
        "update_rate": config["update_rate"],
        "paintings": [record.name for record in records],
        "painting_files": {record.name: str(record.path) for record in records},
        "training_objectives": {
            "growth": bool(training.get("use_growth", True)),
            "persistence": bool(training.get("use_persistence", True)),
            "transition": bool(training.get("use_transition", True)),
        },
    }


def save_checkpoint(
    model: ConditionalNCA,
    config: dict[str, Any],
    records: list[PaintingRecord],
    model_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Save model weights/config and JSON metadata."""

    model_output = Path(model_path)
    metadata_output = Path(metadata_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, model_output)
    metadata_output.write_text(json.dumps(build_metadata(config, records), indent=2), encoding="utf-8")
