"""Dataset loading and validation helpers for manually prepared paintings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .constants import ImageSpec, PAINTING_FILE_ALIASES, PAINTING_IDS


@dataclass(frozen=True)
class PaintingRecord:
    """Loaded painting metadata."""

    name: str
    painting_id: int
    path: Path


def resolve_painting_path(dataset_dir: str | Path, painting_name: str) -> Path:
    """Return the first existing PNG path for a logical painting name."""

    root = Path(dataset_dir)
    aliases = PAINTING_FILE_ALIASES.get(painting_name, (f"{painting_name}.png",))
    for filename in aliases:
        path = root / filename
        if path.exists():
            return path
    alias_text = ", ".join(aliases)
    raise FileNotFoundError(f"Could not find painting '{painting_name}' in {root} (tried {alias_text}).")


def validate_painting(path: str | Path, spec: ImageSpec = ImageSpec()) -> None:
    """Validate format, mode, and dimensions of a single painting PNG."""

    image_path = Path(path)
    if image_path.suffix.lower() != spec.extension:
        raise ValueError(f"{image_path} must be a {spec.extension} file.")
    with Image.open(image_path) as image:
        if image.mode not in spec.channels:
            raise ValueError(f"{image_path} must be RGB or RGBA, got {image.mode}.")
        expected_size = (spec.resolution, spec.resolution)
        if image.size != expected_size:
            raise ValueError(f"{image_path} must be {expected_size}, got {image.size}.")


def load_painting_tensor(path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load a painting as an RGB tensor with shape ``[3, H, W]`` in ``[0, 1]``."""

    validate_painting(path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = list(rgb.getdata())
        tensor = torch.tensor(pixels, dtype=torch.float32).view(rgb.height, rgb.width, 3)
        tensor = tensor.permute(2, 0, 1).div(255.0)
    return tensor.to(device)


def load_paintings(
    dataset_dir: str | Path = "paintings/64",
    painting_names: list[str] | tuple[str, ...] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, list[PaintingRecord]]:
    """Load selected paintings into a target tensor and deterministic records list.

    Returns:
        targets: Tensor of shape ``[num_paintings, 3, 64, 64]``.
        records: Logical names, stable IDs, and file paths for each painting.
    """

    names = list(painting_names or PAINTING_IDS.keys())
    targets: list[torch.Tensor] = []
    records: list[PaintingRecord] = []
    for name in names:
        if name not in PAINTING_IDS:
            raise KeyError(f"Unknown painting '{name}'. Valid names: {sorted(PAINTING_IDS)}")
        path = resolve_painting_path(dataset_dir, name)
        targets.append(load_painting_tensor(path, device=device))
        records.append(PaintingRecord(name=name, painting_id=PAINTING_IDS[name], path=path))
    return torch.stack(targets, dim=0), records


def validate_dataset(dataset_dir: str | Path = "paintings/64", painting_names: list[str] | None = None) -> list[PaintingRecord]:
    """Validate all selected images and return their records."""

    _, records = load_paintings(dataset_dir=dataset_dir, painting_names=painting_names, device="cpu")
    return records
