"""Knob-free rule description (amendment A3, conditioning MODE 2).

Mode 1 (`Ruleset.rule_vector`) hands a model our own factorization: one slot
per knob, so "the goal changed" arrives pre-separated from "the dynamics
changed". If a factorized architecture wins on that input, the win may belong
to the knob design rather than the architecture.

Mode 2 describes the same game WITHOUT knob structure, deriving everything by
probing the game's own semantics: what the board looks like, where you may
place, and how games actually end. Nothing here reads a knob field, so it
generalizes to any family implementing `ruleshift.interface.Game` (each family
supplies its own `rule_descriptor`).

The E2b ablation reports the gap between modes on M0 and M2, which *quantifies*
how much the hand-designed parameterization is doing.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from .engine import Engine

N_PLANES = 3
SIGNATURE_DIM = 7
_PROBE_PLAYOUTS = 24
_PROBE_SEED = 12345


def descriptor_planes(engine: Engine) -> np.ndarray:
    """(3, n, m) per-cell features derived from the game's own structure:

    0. playable mask
    1. line incidence: share of winning lines through each cell (geometry of
       the goal, without naming k, torus, or the forbidden set)
    2. opening legality: cells you may actually place on from the initial
       state (exposes gravity as a placement restriction, not as a flag)
    """
    rules = engine.rules
    planes = np.zeros((N_PLANES, rules.n, rules.m), dtype=np.float32)
    for r, c in rules.playable_cells():
        planes[0, r, c] = 1.0
    n_lines = max(len(engine.lines), 1)
    for cell in engine.playable:
        r, c = divmod(cell, rules.m)
        planes[1, r, c] = len(engine.lines_through[cell]) / n_lines
    for mv in engine.legal_moves(engine.initial()):
        r, c = divmod(mv, rules.m)
        planes[2, r, c] = 1.0
    return planes


def behavioral_signature(
    engine: Engine, n_playouts: int = _PROBE_PLAYOUTS, seed: int = _PROBE_SEED
) -> np.ndarray:
    """Global rule features measured by PLAYING the game, never by reading knobs.

    0. mean game length / cells        (how long games run)
    1. fraction ending on a full board (vs. an early decisive event -> scoring)
    2. mean terminal value to the last mover (goal polarity -> misere)
    3. mean ownership churn per move   (stones changing hands -> capture)
    4. mean branching / cells          (placement freedom -> gravity)
    5. line density: lines / cells     (how much structure the goal has)
    6. mean line size / cells          (how large a goal pattern is -> k)
    """
    rng = random.Random(seed)
    cells = max(engine.rules.num_cells, 1)
    lengths, full_ends, last_values, churns, branchings = [], [], [], [], []
    for _ in range(n_playouts):
        state = engine.initial()
        plies = 0
        while True:
            moves = engine.legal_moves(state)
            if not moves:
                break
            branchings.append(len(moves) / cells)
            before_other = state[1]
            nxt, status = engine.step(state, rng.choice(moves))
            plies += 1
            # stones that changed hands: the mover's own stones only ever grow,
            # so churn shows up as the OPPONENT losing stones (capture flips)
            churns.append((before_other & ~nxt[0]).bit_count() / cells)
            if status is not None:
                lengths.append(plies / cells)
                full_ends.append(1.0 if (nxt[0] | nxt[1]) == engine.full else 0.0)
                last_values.append(float(-status))  # value to the player who just moved
                break
            state = nxt
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    line_sizes = [bin(m).count("1") for m in engine.lines] or [0]
    return np.array(
        [
            mean(lengths),
            mean(full_ends),
            mean(last_values),
            mean(churns),
            mean(branchings),
            len(engine.lines) / cells,
            float(np.mean(line_sizes)) / cells,
        ],
        dtype=np.float32,
    )


def rule_description(engine: Engine) -> dict[str, Any]:
    """Structured knob-free description (the `Game.rule_descriptor` payload)."""
    return {
        "planes": descriptor_planes(engine),
        "signature": behavioral_signature(engine),
        "n_actions": engine.n_actions(),
    }
