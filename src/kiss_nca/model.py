"""Conditional Neural Cellular Automata model for Klimt painting growth."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalNCA(nn.Module):
    """A single shared update rule conditioned on a painting identity.

    The visible state is ``RGB + alpha`` and the remaining channels are hidden
    memory. A local condition map can be supplied at inference time even though
    training uses global conditions.
    """

    def __init__(
        self,
        state_channels: int,
        hidden_channels: int,
        num_paintings: int,
        condition_dim: int,
        neighborhood_size: int = 3,
        update_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if neighborhood_size not in {3, 5}:
            raise ValueError("neighborhood_size must be 3 or 5.")
        if state_channels < 4:
            raise ValueError("state_channels must include RGB, alpha, and optional hidden channels.")
        if not 0.0 < update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1].")

        self.state_channels = state_channels
        self.hidden_channels = hidden_channels
        self.num_paintings = num_paintings
        self.condition_dim = condition_dim
        self.neighborhood_size = neighborhood_size
        self.update_rate = update_rate

        self.condition_embedding = nn.Embedding(num_paintings, condition_dim)
        perceived_channels = state_channels * (neighborhood_size**2)
        self.update_net = nn.Sequential(
            nn.Conv2d(perceived_channels + condition_dim, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Collect flattened local neighborhoods for every cell."""

        padding = self.neighborhood_size // 2
        patches = F.unfold(state, kernel_size=self.neighborhood_size, padding=padding)
        batch, channels_times_area, height_times_width = patches.shape
        height, width = state.shape[-2:]
        return patches.view(batch, channels_times_area, height, width)

    def condition_to_map(
        self,
        condition: torch.Tensor,
        height: int,
        width: int,
        local_condition_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create a dense condition embedding map from global IDs or a local ID map."""

        if local_condition_map is not None:
            if local_condition_map.ndim == 2:
                local_condition_map = local_condition_map.unsqueeze(0)
            embedded = self.condition_embedding(local_condition_map.long())
            return embedded.permute(0, 3, 1, 2).contiguous()

        if condition.ndim == 0:
            condition = condition.unsqueeze(0)
        if condition.ndim != 1:
            raise ValueError("Global condition must be a scalar or [batch] integer tensor.")
        embedded = self.condition_embedding(condition.long())
        return embedded[:, :, None, None].expand(-1, -1, height, width)

    @staticmethod
    def living_mask(state: torch.Tensor) -> torch.Tensor:
        """Cells with enough alpha in their 3x3 neighborhood are considered alive."""

        alpha = state[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

    def step(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        local_condition_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one stochastic NCA update."""

        height, width = state.shape[-2:]
        condition_map = self.condition_to_map(condition, height, width, local_condition_map)
        if condition_map.shape[0] == 1 and state.shape[0] > 1:
            condition_map = condition_map.expand(state.shape[0], -1, -1, -1)
        perceived = self.perceive(state)
        delta = self.update_net(torch.cat([perceived, condition_map], dim=1))
        if self.training and self.update_rate < 1.0:
            update_mask = (torch.rand_like(state[:, :1]) <= self.update_rate).float()
            delta = delta * update_mask
        next_state = state + delta
        alive = self.living_mask(state) | self.living_mask(next_state)
        return next_state * alive.float()

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        steps: int = 1,
        local_condition_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the automata for ``steps`` iterations."""

        for _ in range(steps):
            state = self.step(state, condition, local_condition_map=local_condition_map)
        return state


def make_seed(
    batch_size: int,
    state_channels: int,
    resolution: int = 64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create a centered living seed state."""

    seed = torch.zeros(batch_size, state_channels, resolution, resolution, device=device)
    center = resolution // 2
    seed[:, 3:, center, center] = 1.0
    return seed
