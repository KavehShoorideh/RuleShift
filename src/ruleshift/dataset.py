"""Supervised distillation datasets (plan par.4): exact WDL values and
optimal-move sets from the solver. v1 sampling per docs/scope-freeze.md:
uniform-random playouts, all path prefixes, deduped, fixed seed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .engine import Engine, State
from .rules import Ruleset
from .solver import Solver


def sample_positions(
    engine: Engine, n: int, seed: int = 0, max_playouts_factor: int = 50, strict: bool = True
) -> list[State]:
    """Distinct non-terminal states from uniform-random playout prefixes."""
    rng = random.Random(seed)
    seen: set[State] = set()
    out: list[State] = []
    max_playouts = max_playouts_factor * max(n, 1)
    playouts = 0
    stall = 0  # consecutive playouts adding no new state (small variants saturate)
    while len(out) < n and playouts < max_playouts and stall < 1000:
        playouts += 1
        before = len(out)
        s = engine.initial()
        while True:
            if s not in seen:
                seen.add(s)
                out.append(s)
            mv = rng.choice(engine.legal_moves(s))
            s2, status = engine.step(s, mv)
            if status is not None:
                break
            s = s2
        stall = stall + 1 if len(out) == before else 0
    if strict and len(out) < n:
        raise ValueError(
            f"{engine.rules.variant_id}: only {len(out)} distinct positions found "
            f"in {playouts} playouts (variant may be smaller than n={n})"
        )
    return out[:n]


def encode_state(engine: Engine, state: State) -> np.ndarray:
    """(3, n, m) float32 planes: player-to-move stones, opponent stones, playable mask."""
    rules = engine.rules
    arr = np.zeros((3, rules.n, rules.m), dtype=np.float32)
    cur, opp = state
    for r in range(rules.n):
        for c in range(rules.m):
            i = rules.cell_index(r, c)
            if (cur >> i) & 1:
                arr[0, r, c] = 1.0
            elif (opp >> i) & 1:
                arr[1, r, c] = 1.0
            if (r, c) not in rules.forbidden:
                arr[2, r, c] = 1.0
    return arr


def build_dataset(
    engine: Engine,
    solver: Solver,
    n: int,
    seed: int = 0,
    strict: bool = True,
    states: list[State] | None = None,
) -> dict[str, np.ndarray]:
    """Arrays: boards (N,3,n,m), values (N,), policy_mask/legal_mask (N, n*m)
    multi-hot, states (N,2) uint64, rule_vector (6,). Policy target =
    uniform over the optimal-move set (normalize model-side from the mask).
    Pass `states` to label an explicit position list instead of sampling."""
    if states is None:
        states = sample_positions(engine, n, seed=seed, strict=strict)
    ncells = engine.rules.num_cells
    boards = np.stack([encode_state(engine, s) for s in states])
    values = np.zeros(len(states), dtype=np.int8)
    policy_mask = np.zeros((len(states), ncells), dtype=np.float32)
    legal_mask = np.zeros((len(states), ncells), dtype=np.float32)
    for i, s in enumerate(states):
        v, moves = solver.policy(s)
        values[i] = v
        policy_mask[i, list(moves)] = 1.0
        legal_mask[i, engine.legal_moves(s)] = 1.0
    return {
        "boards": boards,
        "values": values,
        "policy_mask": policy_mask,
        "legal_mask": legal_mask,
        "states": np.array(states, dtype=np.uint64),
        "rule_vector": np.array(engine.rules.rule_vector(), dtype=np.float32),
    }


def save_dataset(path: str | Path, data: dict[str, np.ndarray], rules: Ruleset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, variant_json=json.dumps(rules.to_dict()), **data)


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], Ruleset]:
    with np.load(path) as z:
        rules = Ruleset.from_dict(json.loads(str(z["variant_json"])))
        data = {k: z[k] for k in z.files if k != "variant_json"}
    return data, rules
