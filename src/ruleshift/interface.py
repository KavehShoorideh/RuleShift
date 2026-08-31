"""Formalism-agnostic game interface (amendment A5).

The harness is written against this minimal protocol; the m,n,k family in
`ruleshift.engine` is ONE plug-in implementation, not the architecture. Any
game family that implements it — Fairy-Stockfish variants with Betza piece
definitions, a Ludii game-description family, a gridworld — can be labeled,
solved, scored, and compared by every downstream module (solver, datasets,
metrics, behavioral distance, rule descriptors) without changes.

Contract:
- states are hashable and self-contained (no history), so a transposition
  table is sound;
- `step` returns the terminal value FOR THE PLAYER TO MOVE IN THE RESULT
  (the negamax convention used throughout), or None if the game continues;
- every game terminates in a bounded number of plies from any state.
"""
from __future__ import annotations

from typing import Any, Hashable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Game(Protocol):
    """Minimal game interface: legal moves, terminal test, exact value, descriptor."""

    def initial(self) -> Hashable:
        """The starting state."""

    def legal_moves(self, state: Hashable) -> Sequence[int]:
        """Legal actions in `state`, as indices into a fixed action space."""

    def step(self, state: Hashable, move: int) -> tuple[Hashable, int | None]:
        """(next_state, terminal value for the player to move in next_state or None)."""

    def full_status(self, state: Hashable) -> int | None:
        """Terminal value for the player to move, without last-move knowledge."""

    def n_actions(self) -> int:
        """Size of the action space."""

    def rule_descriptor(self) -> dict[str, Any]:
        """Knob-free description of THIS game's rules (amendment A3, mode 2).

        Must be derivable from the game's own semantics, not from the
        parameterization used to construct it.
        """


def check_game(game: Game, max_plies: int = 10_000) -> None:
    """Smoke-check that an implementation satisfies the contract (one playout)."""
    import random

    rng = random.Random(0)
    state = game.initial()
    if game.full_status(state) is not None:
        raise ValueError("initial state must be non-terminal")
    for ply in range(max_plies):
        moves = list(game.legal_moves(state))
        if not moves:
            raise ValueError(f"no legal moves at ply {ply} but game was not terminal")
        nxt, status = game.step(state, rng.choice(moves))
        if not isinstance(nxt, Hashable):
            raise TypeError("states must be hashable")
        if status is not None:
            if status not in (-1, 0, 1):
                raise ValueError(f"terminal value {status} outside {{-1, 0, 1}}")
            return
        state = nxt
    raise ValueError(f"game did not terminate within {max_plies} plies")
