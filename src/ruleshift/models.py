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
RULE_DIM = 8  # conditioning mode 1: explicit knob vector
DESCRIPTOR_DIM = N_PLANES * N_CELLS + 7  # mode 2: padded planes + behavioural signature
INPUT_DIM = N_PLANES * N_CELLS + RULE_DIM  # default (knob mode)

# Amendment A3: the two rule-conditioning modes compared in E2b.
KNOB, DESCRIPTOR = "knob", "descriptor"


def conditioning_dim(mode: str = KNOB) -> int:
    if mode == KNOB:
        return RULE_DIM
    if mode == DESCRIPTOR:
        return DESCRIPTOR_DIM
    raise ValueError(f"unknown conditioning mode {mode!r}")


def input_dim(mode: str = KNOB) -> int:
    return N_PLANES * N_CELLS + conditioning_dim(mode)


def conditioning_vector(engine, mode: str = KNOB) -> np.ndarray:
    """The rule-conditioning input for one variant, in the requested mode.

    KNOB hands the model our factorization (one slot per knob); DESCRIPTOR
    gives a knob-free description derived from the game's own semantics
    (`ruleshift.descriptor`). E2b reports the gap between the two.
    """
    if mode == KNOB:
        return norm_rule_vector(engine.rules)
    if mode == DESCRIPTOR:
        d = engine.rule_descriptor()
        return np.concatenate([pad_planes(d["planes"]).reshape(-1), d["signature"]]).astype(
            np.float32
        )
    raise ValueError(f"unknown conditioning mode {mode!r}")


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
    """Conditioning MODE 1 (A3): the explicit knob vector, normalized."""
    return np.array(
        [
            rules.m / PAD,
            rules.n / PAD,
            rules.k / 4.0,
            float(rules.gravity),
            float(rules.misere),
            float(rules.torus),
            float(rules.capture),
            float(rules.scoring),
        ],
        dtype=np.float32,
    )


class M0(nn.Module):
    """Monolithic MLP baseline. `hidden`/`depth` are the parameter-matching knobs;
    `conditioning` selects the A3 mode (knob vector vs. knob-free description)."""

    def __init__(self, hidden: int = 256, depth: int = 3, conditioning: str = KNOB):
        super().__init__()
        self.conditioning = conditioning
        layers: list[nn.Module] = []
        d = input_dim(conditioning)
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
