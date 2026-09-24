"""Observation adapters and alternative BW networks; imported only with Torch."""

import torch
from torch import nn


class SpatialObservation(nn.Module):
    """Map the shared replay encoding to goal-relative height-by-column planes.

    Other block identities are irrelevant to a single on-goal. Occupancy and
    the two named blocks preserve its geometry without an arbitrary ID axis.
    Arm position and held-block role are broadcast over the height dimension.
    """

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return seven binary planes with shape (batch, 7, 30, 10)."""
        cells = values[:, 2:32]
        source = values[:, 32].unsqueeze(1)
        destination = values[:, 33].unsqueeze(1)
        arm = values[:, 0].amax(dim=-1).unsqueeze(1).expand(-1, 30, -1)
        held = values[:, 1]
        held_source = (held * values[:, 32]).sum(dim=-1)
        held_destination = (held * values[:, 33]).sum(dim=-1)
        held_other = held.sum(dim=-1) - held_source - held_destination
        return torch.stack([
            cells.sum(dim=-1), (cells * source).sum(dim=-1),
            (cells * destination).sum(dim=-1), arm,
            held_other.unsqueeze(1).expand(-1, 30, -1),
            held_source.unsqueeze(1).expand(-1, 30, -1),
            held_destination.unsqueeze(1).expand(-1, 30, -1),
        ], dim=1)


class VolumeObservation(nn.Module):
    """Keep the original three observation axes spatial for Conv3d.

    The original volume includes the arm and held-block slices. Two separate
    channels broadcast the requested block identities across those slices.
    """

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return shape (batch, 3, 32, 10, 30) without collapsing height."""
        return torch.stack([
            values[:, :32],
            values[:, 32].unsqueeze(1).expand(-1, 32, -1, -1),
            values[:, 33].unsqueeze(1).expand(-1, 32, -1, -1),
        ], dim=1)


def spatial_network() -> nn.Sequential:
    """Use physical height and table column as the two convolution axes."""
    return nn.Sequential(
        SpatialObservation(),
        nn.Conv2d(7, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(128 * 8 * 3, 512), nn.ReLU(), nn.Linear(512, 4),
    )


def volume_network() -> nn.Sequential:
    """Convolve over slice, column and block identity in the original volume."""
    return nn.Sequential(
        VolumeObservation(),
        nn.Conv3d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(64 * 8 * 3 * 8, 128), nn.ReLU(), nn.Linear(128, 4),
    )
