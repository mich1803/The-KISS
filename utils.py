"""PyTorch Lightning utilities for training Growing Neural Cellular Automata on Klimt paintings.

The original reference notebook in this repository follows the Distill
"Growing Neural Cellular Automata" TensorFlow implementation.  This module keeps
that experiment structure, but exposes reusable PyTorch/PyTorch-Lightning
building blocks so the main notebook can stay short and focused on the
painting-training workflow.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


# -----------------------------------------------------------------------------
# Image and display helpers
# -----------------------------------------------------------------------------


def np2pil(array: np.ndarray) -> Image.Image:
    """Convert a float image in [0, 1] or uint8 image into a PIL image."""
    array = np.asarray(array)
    if array.dtype in (np.float32, np.float64):
        array = np.uint8(np.clip(array, 0.0, 1.0) * 255)
    return Image.fromarray(array)


def imwrite(path: str | Path | io.BytesIO, array: np.ndarray, fmt: Optional[str] = None) -> None:
    """Write an image array to disk or to an in-memory file object."""
    array = np.asarray(array)
    if isinstance(path, (str, Path)):
        suffix = Path(path).suffix.lower().lstrip(".")
        fmt = "jpeg" if suffix == "jpg" else suffix
    np2pil(array).save(path, fmt, quality=95)


def tile2d(array: np.ndarray, width: Optional[int] = None) -> np.ndarray:
    """Tile a batch of images into a single grid image for quick inspection."""
    array = np.asarray(array)
    if width is None:
        width = int(np.ceil(np.sqrt(len(array))))
    tile_h, tile_w = array.shape[1:3]
    pad = (width - len(array)) % width
    array = np.pad(array, [(0, pad)] + [(0, 0)] * (array.ndim - 1), mode="constant")
    height = len(array) // width
    array = array.reshape([height, width] + list(array.shape[1:]))
    return np.rollaxis(array, 2, 1).reshape([tile_h * height, tile_w * width] + list(array.shape[4:]))


def zoom(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """Nearest-neighbor zoom for pixel-art-like NCA visualizations."""
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def show_image(image: np.ndarray, scale: int = 1, title: Optional[str] = None) -> None:
    """Display an RGB/RGBA image in a notebook with optional nearest-neighbor scaling."""
    image = zoom(image, scale) if scale != 1 else image
    plt.figure(figsize=(4, 4))
    plt.imshow(np.clip(image, 0.0, 1.0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


# -----------------------------------------------------------------------------
# Target painting loading
# -----------------------------------------------------------------------------


def list_paintings(painting_dir: str | Path = "paintings/64") -> list[Path]:
    """Return available Klimt target images sorted by file name."""
    painting_dir = Path(painting_dir)
    return sorted(path for path in painting_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def load_painting(path: str | Path, max_size: int = 64) -> np.ndarray:
    """Load a painting as a premultiplied RGBA float image.

    Neural cellular automata in the reference notebook optimize the first four
    channels as premultiplied RGBA.  The Klimt files are normalized to that same
    representation here; RGB-only images receive an opaque alpha channel.
    """
    image = Image.open(path).convert("RGBA")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array[..., :3] *= array[..., 3:4]
    return array


def pad_target(target_rgba: np.ndarray, padding: int) -> np.ndarray:
    """Pad a target RGBA image with transparent pixels, matching the reference setup."""
    return np.pad(target_rgba, [(padding, padding), (padding, padding), (0, 0)], mode="constant")


# -----------------------------------------------------------------------------
# Tensor layout helpers
# -----------------------------------------------------------------------------


def numpy_rgba_to_tensor(image: np.ndarray, device: Optional[torch.device] = None) -> Tensor:
    """Convert HWC RGBA numpy data to a CHW torch tensor."""
    return torch.as_tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()


def tensor_to_numpy_image(x: Tensor) -> np.ndarray:
    """Convert CHW or NCHW torch image tensors to HWC/NHWC numpy arrays."""
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.ndim == 3:
        return x.permute(1, 2, 0).numpy()
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).numpy()
    raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")


def to_rgba(x: Tensor) -> Tensor:
    """Select the RGBA channels from an NCA state tensor in NCHW layout."""
    return x[:, :4]


def to_alpha(x: Tensor) -> Tensor:
    """Return the clipped alpha channel used to decide which cells are alive."""
    return x[:, 3:4].clamp(0.0, 1.0)


def to_rgb(x: Tensor) -> Tensor:
    """Composite premultiplied RGBA over a white background for visualization."""
    rgb, alpha = x[:, :3], to_alpha(x)
    return 1.0 - alpha + rgb


def get_living_mask(x: Tensor) -> Tensor:
    """Cells are alive when their 3x3 neighborhood contains visible alpha."""
    return F.max_pool2d(to_alpha(x), kernel_size=3, stride=1, padding=1) > 0.1


def make_seed(height: int, width: int, channel_n: int = 16, batch_size: int = 1, device: Optional[torch.device] = None) -> Tensor:
    """Create the single live-cell seed state used by Growing NCA."""
    x = torch.zeros(batch_size, channel_n, height, width, dtype=torch.float32, device=device)
    x[:, 3:, height // 2, width // 2] = 1.0
    return x


def make_circle_masks(n: int, height: int, width: int, device: Optional[torch.device] = None) -> Tensor:
    """Generate random circular masks for the regeneration experiment."""
    xs = torch.linspace(-1.0, 1.0, width, device=device)[None, None, :]
    ys = torch.linspace(-1.0, 1.0, height, device=device)[None, :, None]
    center = torch.empty(2, n, 1, 1, device=device).uniform_(-0.5, 0.5)
    radius = torch.empty(n, 1, 1, device=device).uniform_(0.1, 0.4)
    xs = (xs - center[0]) / radius
    ys = (ys - center[1]) / radius
    return ((xs * xs + ys * ys) < 1.0).float()


# -----------------------------------------------------------------------------
# Neural cellular automata model
# -----------------------------------------------------------------------------


class NeuralCA(nn.Module):
    """Growing Neural Cellular Automata core model in PyTorch.

    Each step perceives every cell with identity/Sobel filters, predicts a
    channel update with two 1x1 convolutions, applies stochastic cell firing, and
    masks out cells that were not alive before and after the update.
    """

    def __init__(self, channel_n: int = 16, hidden_n: int = 128, fire_rate: float = 0.5):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_n, channel_n, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, x: Tensor, angle: float | Tensor = 0.0) -> Tensor:
        """Apply identity and rotated Sobel filters depthwise to every channel."""
        device, dtype = x.device, x.dtype
        identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)
        dx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype) / 8.0
        dy = dx.t()
        angle = torch.as_tensor(angle, device=device, dtype=dtype)
        c, s = torch.cos(angle), torch.sin(angle)
        kernels = torch.stack([identity, c * dx - s * dy, s * dx + c * dy])
        kernels = kernels[:, None, :, :].repeat(self.channel_n, 1, 1, 1)
        y = F.conv2d(x, kernels, padding=1, groups=self.channel_n)
        b, _, h, w = y.shape
        return y.reshape(b, self.channel_n, 3, h, w).reshape(b, self.channel_n * 3, h, w)

    def forward(self, x: Tensor, fire_rate: Optional[float] = None, angle: float = 0.0, step_size: float = 1.0) -> Tensor:
        """Advance the CA state by one asynchronous update step."""
        pre_life_mask = get_living_mask(x)
        dx = self.update_net(self.perceive(x, angle)) * step_size
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        update_mask = (torch.rand_like(x[:, :1]) <= fire_rate).float()
        x = x + dx * update_mask
        post_life_mask = get_living_mask(x)
        return x * (pre_life_mask & post_life_mask).float()


class DummyStepDataset(Dataset):
    """Tiny dataset whose only job is to give Lightning a requested number of steps."""

    def __init__(self, steps: int):
        self.steps = steps

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> int:
        return index


class NCALightningModel(pl.LightningModule):
    """Lightning module that reproduces the reference NCA training loop.

    The module owns the target, seed, and optional pattern pool.  A Lightning
    ``Trainer`` can then run the experiment while callbacks/loggers/checkpoints
    remain available for longer training runs.
    """

    def __init__(
        self,
        target_rgba: np.ndarray,
        channel_n: int = 16,
        batch_size: int = 8,
        pool_size: int = 1024,
        target_padding: int = 16,
        lr: float = 2e-3,
        fire_rate: float = 0.5,
        use_pattern_pool: bool = True,
        damage_n: int = 3,
        min_steps: int = 64,
        max_steps: int = 96,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["target_rgba"])
        self.ca = NeuralCA(channel_n=channel_n, fire_rate=fire_rate)
        target = numpy_rgba_to_tensor(pad_target(target_rgba, target_padding))
        _, height, width = target.shape
        seed = make_seed(height, width, channel_n=channel_n, batch_size=1)[0]
        self.register_buffer("target", target)
        self.register_buffer("seed", seed)
        self.register_buffer("pool", seed.unsqueeze(0).repeat(pool_size, 1, 1, 1))
        self.loss_history: list[float] = []
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def loss_per_sample(self, x: Tensor) -> Tensor:
        """Mean squared RGBA loss against the padded target for each batch item."""
        return (to_rgba(x) - self.target.unsqueeze(0)).pow(2).mean(dim=(1, 2, 3))

    def _sample_batch(self) -> tuple[Tensor, Optional[Tensor]]:
        """Sample the pattern pool, reset the worst item to the seed, and add damage."""
        batch_size = self.hparams.batch_size
        if not self.hparams.use_pattern_pool:
            return self.seed.unsqueeze(0).repeat(batch_size, 1, 1, 1), None

        indices = torch.randperm(self.hparams.pool_size, device=self.device)[:batch_size]
        x0 = self.pool[indices].clone()
        ranks = torch.argsort(self.loss_per_sample(x0), descending=True)
        x0 = x0[ranks]
        indices = indices[ranks]
        x0[:1] = self.seed

        damage_n = min(int(self.hparams.damage_n), batch_size)
        if damage_n > 0:
            _, height, width = self.target.shape
            keep = 1.0 - make_circle_masks(damage_n, height, width, device=self.device)[:, None]
            x0[-damage_n:] *= keep
        return x0, indices

    def training_step(self, batch, batch_idx):  # noqa: D401 - Lightning hook name is self-explanatory.
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        x0, pool_indices = self._sample_batch()
        x = x0
        iter_n = int(torch.randint(self.hparams.min_steps, self.hparams.max_steps + 1, (), device=self.device).item())
        for _ in range(iter_n):
            x = self.ca(x)
        loss = self.loss_per_sample(x).mean()

        self.manual_backward(loss)
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad /= parameter.grad.norm() + 1e-8
        optimizer.step()
        scheduler.step()

        if pool_indices is not None:
            self.pool[pool_indices] = x.detach()

        loss_value = float(loss.detach().cpu())
        self.loss_history.append(loss_value)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return {"loss": loss.detach(), "x0": x0.detach(), "x": x.detach()}

    @torch.no_grad()
    def grow(self, steps: int = 128, batch_size: int = 1) -> Tensor:
        """Grow one or more paintings from the learned seed for visualization."""
        x = self.seed.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
        for _ in range(steps):
            x = self.ca(x)
        return x


# -----------------------------------------------------------------------------
# Notebook convenience functions
# -----------------------------------------------------------------------------


def make_trainer(max_steps: int, accelerator: str = "auto", log_every_n_steps: int = 10, **kwargs) -> pl.Trainer:
    """Create a compact Lightning trainer suitable for notebook experiments."""
    return pl.Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=log_every_n_steps,
        **kwargs,
    )


def make_step_loader(max_steps: int) -> DataLoader:
    """Return a deterministic dummy DataLoader with one item per optimization step."""
    return DataLoader(DummyStepDataset(max_steps), batch_size=1, shuffle=False)


def plot_loss(loss_history: Sequence[float]) -> None:
    """Plot the log10 training loss curve used in the reference notebook."""
    plt.figure(figsize=(10, 4))
    plt.title("Loss history (log10)")
    plt.plot(np.log10(np.maximum(loss_history, 1e-12)), ".", alpha=0.25)
    plt.xlabel("training step")
    plt.ylabel("log10 loss")
    plt.show()


def visualize_states(states: Tensor, title: Optional[str] = None, scale: int = 2) -> None:
    """Tile NCA states as RGB images on a white background."""
    rgb = tensor_to_numpy_image(to_rgb(states))
    show_image(tile2d(rgb), scale=scale, title=title)


def save_checkpoint(model: NCALightningModel, path: str | Path) -> None:
    """Save a lightweight PyTorch checkpoint containing model weights and settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}, path)
