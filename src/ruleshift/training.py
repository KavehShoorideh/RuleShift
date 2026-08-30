"""Training and evaluation for the model track (plan par.4-5).

Supervised distillation from exact solutions: policy target = uniform over the
optimal-move set (KL to a masked softmax); value target = exact WDL class.
Few-shot adaptation = fine-tune on n samples of the target variant.
Regret evaluation bridges back to the exact solver via metrics.evaluate_policy.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .dataset import encode_state
from .engine import Engine, State
from .metrics import RegretReport, evaluate_policy
from .models import (
    M0,
    PAD,
    native_to_frame,
    norm_rule_vector,
    pad_cells,
    pad_planes,
)
from .rules import Ruleset
from .solver import Solver


@dataclass
class Tensors:
    """A tensorized labeled dataset for one variant."""

    x: torch.Tensor       # (N, INPUT_DIM)
    policy: torch.Tensor  # (N, N_CELLS) target distribution (sums to 1)
    legal: torch.Tensor   # (N, N_CELLS) 0/1 mask
    value: torch.Tensor   # (N,) long class index (WDL value + 1)
    rules: Ruleset

    def __len__(self) -> int:
        return len(self.x)

    def subsample(self, n: int, seed: int) -> "Tensors":
        idx = torch.randperm(len(self.x), generator=torch.Generator().manual_seed(seed))[:n]
        return Tensors(self.x[idx], self.policy[idx], self.legal[idx], self.value[idx], self.rules)


def tensorize(data: dict, rules: Ruleset) -> Tensors:
    n, m = rules.n, rules.m
    boards = np.stack([pad_planes(b) for b in data["boards"]])
    rule = norm_rule_vector(rules)
    x = np.concatenate(
        [boards.reshape(len(boards), -1), np.tile(rule, (len(boards), 1))], axis=1
    ).astype(np.float32)
    pol = np.stack([pad_cells(p, n, m) for p in data["policy_mask"]])
    pol /= pol.sum(axis=1, keepdims=True)
    legal = np.stack([pad_cells(l, n, m) for l in data["legal_mask"]])
    value = data["values"].astype(np.int64) + 1
    return Tensors(
        torch.from_numpy(x),
        torch.from_numpy(pol),
        torch.from_numpy(legal),
        torch.from_numpy(value),
        rules,
    )


def stack(datasets: Sequence[Tensors]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.cat([d.x for d in datasets]),
        torch.cat([d.policy for d in datasets]),
        torch.cat([d.legal for d in datasets]),
        torch.cat([d.value for d in datasets]),
    )


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2000
    batch: int = 256
    lr: float = 1e-3
    seed: int = 0
    value_weight: float = 1.0
    device: str = "cpu"


def distill_loss(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    policy_t: torch.Tensor,
    legal: torch.Tensor,
    value_t: torch.Tensor,
    value_weight: float,
) -> tuple[torch.Tensor, float, float]:
    masked = policy_logits.masked_fill(legal == 0, -1e9)
    logp = torch.log_softmax(masked, dim=1)
    pol_loss = -(policy_t * logp).sum(dim=1).mean()
    val_loss = nn.functional.cross_entropy(value_logits, value_t)
    total = pol_loss + value_weight * val_loss
    return total, float(pol_loss.detach()), float(val_loss.detach())


def train(
    model: M0,
    datasets: Sequence[Tensors] | Tensors,
    cfg: TrainConfig,
    log: Callable[..., None] | None = None,
) -> M0:
    """In-place SGD training over the union of the given datasets."""
    if isinstance(datasets, Tensors):
        datasets = [datasets]
    x, pol, legal, val = stack(datasets)
    device = torch.device(cfg.device)
    model.to(device).train()
    x, pol, legal, val = (t.to(device) for t in (x, pol, legal, val))
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(x)
    for step in range(cfg.steps):
        idx = torch.randint(0, n, (min(cfg.batch, n),), generator=g).to(device)
        pl, vl = model(x[idx])
        loss, pol_l, val_l = distill_loss(pl, vl, pol[idx], legal[idx], val[idx], cfg.value_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log is not None and (step % 200 == 0 or step == cfg.steps - 1):
            log(step=step, loss=float(loss), policy_loss=pol_l, value_loss=val_l)
    model.eval()
    return model


def adapt(model: M0, target: Tensors, n_samples: int, cfg: TrainConfig) -> M0:
    """Few-shot adaptation: fine-tune a COPY of model on n_samples of the target."""
    adapted = copy.deepcopy(model)
    if n_samples == 0:
        return adapted.eval()
    sub = target.subsample(n_samples, seed=cfg.seed)
    return train(adapted, sub, cfg)


def model_policy_fn(model: M0, engine: Engine, device: str = "cpu") -> Callable[[State], int]:
    """Greedy move chooser over legal moves (native indices) for metrics.evaluate_policy."""
    rules = engine.rules
    rv = norm_rule_vector(rules)
    dev = torch.device(device)
    model.to(dev).eval()

    def fn(state: State) -> int:
        planes = pad_planes(encode_state(engine, state))
        x = np.concatenate([planes.reshape(-1), rv]).astype(np.float32)[None]
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(x).to(dev))
        logits = logits[0].cpu().numpy()
        legal = engine.legal_moves(state)
        return max(legal, key=lambda mv: logits[native_to_frame(mv, rules.m)])

    return fn


def eval_model_regret(
    model: M0, engine: Engine, solver: Solver, positions: Sequence[State], device: str = "cpu"
) -> RegretReport:
    return evaluate_policy(solver, model_policy_fn(model, engine, device), positions)
