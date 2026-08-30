"""Models (plan par.4). M0: monolithic MLP on padded board planes + rule vector.

Every variant is embedded in a fixed PAD x PAD frame, bottom-left anchored
(PAD = 6, the grid bound). The action space is the PAD*PAD cell grid with
illegal/nonexistent cells masked at loss/eval time. Rule conditioning is the
normalized [m, n, k, gravity, misere, torus] vector concatenated to the
flattened planes -- the concat baseline mirroring DMA*-SH's control.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .rules import Ruleset

PAD = 6
N_CELLS = PAD * PAD
N_PLANES = 3
RULE_DIM = 6
INPUT_DIM = N_PLANES * N_CELLS + RULE_DIM


def pad_planes(planes: np.ndarray) -> np.ndarray:
    """(3, n, m) planes -> (3, PAD, PAD), bottom-left anchored."""
    _, n, m = planes.shape
    out = np.zeros((N_PLANES, PAD, PAD), dtype=np.float32)
    out[:, :n, :m] = planes
    return out


def pad_cells(vec: np.ndarray, n: int, m: int) -> np.ndarray:
    """(n*m,) native cell-indexed vector -> (N_CELLS,) frame vector."""
    out = np.zeros(N_CELLS, dtype=np.float32)
    out.reshape(PAD, PAD)[:n, :m] = np.asarray(vec, dtype=np.float32).reshape(n, m)
    return out


def native_to_frame(cell: int, m: int) -> int:
    r, c = divmod(cell, m)
    return r * PAD + c


def frame_to_native(cell: int, m: int) -> int:
    r, c = divmod(cell, PAD)
    return r * m + c


def norm_rule_vector(rules: Ruleset) -> np.ndarray:
    return np.array(
        [
            rules.m / PAD,
            rules.n / PAD,
            rules.k / 4.0,
            float(rules.gravity),
            float(rules.misere),
            float(rules.torus),
        ],
        dtype=np.float32,
    )


class M0(nn.Module):
    """Monolithic MLP baseline. `hidden`/`depth` are the parameter-matching knobs."""

    def __init__(self, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = INPUT_DIM
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(d, N_CELLS)
        self.value_head = nn.Linear(d, 3)  # WDL classes (loss/draw/win for player to move)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
