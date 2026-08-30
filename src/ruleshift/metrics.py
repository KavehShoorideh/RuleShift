"""Exact-regret evaluation suite (plan par.3).

Per-move regret vs. the optimal WDL value, from the mover's perspective:
regret = value(best move) - value(played move), in {0, 1, 2}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .engine import State
from .solver import Solver


def move_regret(solver: Solver, state: State, move: int) -> int:
    best, _ = solver.policy(state)
    return best - solver.child_value(state, move)


@dataclass(frozen=True)
class RegretReport:
    per_position: tuple[int, ...]

    @property
    def n_positions(self) -> int:
        return len(self.per_position)

    @property
    def total_regret(self) -> int:
        return sum(self.per_position)

    @property
    def mean_regret(self) -> float:
        return self.total_regret / len(self.per_position)

    @property
    def frac_optimal(self) -> float:
        return sum(r == 0 for r in self.per_position) / len(self.per_position)


def evaluate_policy(
    solver: Solver,
    policy_fn: Callable[[State], int],
    positions: Sequence[State],
) -> RegretReport:
    """Exact per-move regret of policy_fn's chosen move on each position."""
    if not positions:
        raise ValueError("no positions to evaluate")
    return RegretReport(
        per_position=tuple(move_regret(solver, s, policy_fn(s)) for s in positions)
    )


def samples_to_epsilon(
    sample_sizes: Sequence[int], regrets: Sequence[float], eps: float
) -> int | None:
    """Smallest sample size whose measured regret is <= eps (adaptation curves);
    None if never reached."""
    for size, regret in sorted(zip(sample_sizes, regrets, strict=True)):
        if regret <= eps:
            return size
    return None
