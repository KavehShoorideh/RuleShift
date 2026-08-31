"""Solver-grounded rule distance (amendment A1) — the PRIMARY distance axis.

Distance between two variants is the divergence of *optimal play* on a shared
sample of positions, not the number of knobs that differ. This makes the
headline x-axis ground truth rather than an artifact of our parameterization,
which is the direct answer to the construct-validity objection: a factorized
model cannot win by construction on an axis defined by the exact solver.

Two components, both in [0, 1]:
- `policy_disagreement`: Jaccard distance between the exact optimal-move sets.
- `value_divergence`:    |v_a - v_b| / 2 over exact WDL values.

Knob edit distance (`Ruleset.distance`) is retained as a descriptive label only.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dataset import sample_positions
from .engine import Engine, State
from .rules import Ruleset
from .solver import Solver


@dataclass(frozen=True)
class BehavioralDistance:
    policy_disagreement: float
    value_divergence: float
    n_positions: int
    knob_distance: int

    @property
    def distance(self) -> float:
        """Headline scalar: mean of the two divergence components."""
        return (self.policy_disagreement + self.value_divergence) / 2


def cells_of(bits: int, m: int) -> set[tuple[int, int]]:
    out = set()
    i = 0
    while bits:
        if bits & 1:
            out.add(divmod(i, m))
        bits >>= 1
        i += 1
    return out


def remap(bits: int, m_from: int, m_to: int) -> int:
    """Re-index a bitboard from one board width to another (same (r, c) cells)."""
    out = 0
    for r, c in cells_of(bits, m_from):
        out |= 1 << (r * m_to + c)
    return out


def transfer(state: State, m_from: int, engine_to: Engine) -> State | None:
    """Embed a position into another variant, or None if it has no counterpart.

    A position transfers when every occupied cell exists and is playable in the
    target, the move parity is consistent, and the position is non-terminal
    there (so both variants have a well-defined optimal-move set).
    """
    rules = engine_to.rules
    cur, opp = state
    out = []
    for bits in (cur, opp):
        for r, c in cells_of(bits, m_from):
            if not (0 <= r < rules.n and 0 <= c < rules.m) or (r, c) in rules.forbidden:
                return None
        out.append(remap(bits, m_from, rules.m))
    moved = (out[0], out[1])
    if moved[1].bit_count() - moved[0].bit_count() not in (0, 1):
        return None
    try:
        if engine_to.full_status(moved) is not None:
            return None
    except ValueError:
        return None  # unreachable in the target variant
    return moved


def _jaccard_distance(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def behavioral_distance(
    rules_a: Ruleset,
    rules_b: Ruleset,
    n_positions: int = 200,
    seed: int = 0,
    solver_a: Solver | None = None,
    solver_b: Solver | None = None,
) -> BehavioralDistance:
    """Divergence of optimal play between two variants.

    Positions are drawn from BOTH variants and pooled, so the measure is
    symmetric and not biased toward either variant's reachable set.
    """
    engine_a, engine_b = Engine(rules_a), Engine(rules_b)
    solver_a = solver_a or Solver(engine_a)
    solver_b = solver_b or Solver(engine_b)

    pool: list[State] = []
    seen: set[State] = set()
    half = max(1, n_positions // 2)
    for src_engine, dst_engine in ((engine_a, engine_b), (engine_b, engine_a)):
        for s in sample_positions(src_engine, half, seed=seed, strict=False):
            in_a = s if src_engine is engine_a else transfer(s, src_engine.rules.m, engine_a)
            if in_a is None or in_a in seen:
                continue
            if transfer(in_a, engine_a.rules.m, engine_b) is None:
                continue
            try:
                if engine_a.full_status(in_a) is not None:
                    continue
            except ValueError:
                continue
            seen.add(in_a)
            pool.append(in_a)

    if not pool:
        raise ValueError(
            f"{rules_a.variant_id} and {rules_b.variant_id} share no comparable positions"
        )

    pol, val = 0.0, 0.0
    for s_a in pool:
        s_b = transfer(s_a, engine_a.rules.m, engine_b)
        v_a, moves_a = solver_a.policy(s_a)
        v_b, moves_b = solver_b.policy(s_b)
        set_a = {divmod(mv, engine_a.rules.m) for mv in moves_a}
        set_b = {divmod(mv, engine_b.rules.m) for mv in moves_b}
        pol += _jaccard_distance(set_a, set_b)
        val += abs(v_a - v_b) / 2
    n = len(pool)
    return BehavioralDistance(
        policy_disagreement=pol / n,
        value_divergence=val / n,
        n_positions=n,
        knob_distance=rules_a.distance(rules_b),
    )


def distance_to_set(
    rules: Ruleset, training: list[Ruleset], n_positions: int = 200, seed: int = 0
) -> BehavioralDistance:
    """Distance from a held-out variant to its NEAREST training variant."""
    best = None
    for t in training:
        try:
            d = behavioral_distance(rules, t, n_positions=n_positions, seed=seed)
        except ValueError:
            continue
        if best is None or d.distance < best.distance:
            best = d
    if best is None:
        raise ValueError(f"{rules.variant_id} shares no positions with any training variant")
    return best
