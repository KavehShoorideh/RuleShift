"""Exact game solver: negamax, fail-soft alpha-beta over {-1, 0, +1}, TT.

Correctness is certified in tests against a pruning-free reference solver
across knob combinations (tests/reference.py). With the full window
(LOSS, WIN), returned values are exact: the value domain is confined to
[-1, +1], so a bound at the boundary is the value itself.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from .engine import DRAW, LOSS, WIN, Engine, State

EXACT, LOWER, UPPER = 0, 1, -1


class Solver:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.tt: dict[State, tuple[int, int, int]] = {}  # (value, flag, best_move)
        self.nodes = 0

    # ------------------------------------------------------------ public API
    def value(self, state: State) -> int:
        """Exact WDL value for the player to move (terminal-aware)."""
        status = self.engine.full_status(state)
        if status is not None:
            return status
        return self._search(state[0], state[1], LOSS, WIN)

    def policy(self, state: State) -> tuple[int, tuple[int, ...]]:
        """(exact value, sorted tuple of ALL optimal moves) for a non-terminal state."""
        if self.engine.full_status(state) is not None:
            raise ValueError("policy of a terminal state")
        vals = [(mv, self.child_value(state, mv)) for mv in self.engine.legal_moves(state)]
        best = max(v for _, v in vals)
        return best, tuple(sorted(mv for mv, v in vals if v == best))

    def child_value(self, state: State, mv: int) -> int:
        """Exact value of playing mv in state, from the mover's perspective."""
        cur, opp = state
        nxt = cur | (1 << mv)
        if self.engine.completes_line(nxt, mv):
            return LOSS if self.engine.rules.misere else WIN
        if (nxt | opp) == self.engine.full:
            return DRAW
        return -self._search(opp, nxt, LOSS, WIN)

    # ----------------------------------------------------------------- core
    def _search(self, cur: int, opp: int, alpha: int, beta: int) -> int:
        """Fail-soft negamax with TT; position must be non-terminal."""
        self.nodes += 1
        key = (cur, opp)
        entry = self.tt.get(key)
        tt_move = -1
        if entry is not None:
            v, flag, tt_move = entry
            if flag == EXACT:
                return v
            if flag == LOWER:
                if v >= beta:
                    return v
                if v > alpha:
                    alpha = v
            else:  # UPPER
                if v <= alpha:
                    return v
                if v < beta:
                    beta = v
        alpha_orig = alpha
        engine = self.engine
        occ = cur | opp
        misere = engine.rules.misere
        full = engine.full
        completes = engine.completes_line

        # Dead-position rule: if every line already contains stones of both
        # players, no line can ever be completed, so the game is an exact draw
        # (sound in misere too: nobody can ever be made to complete a line).
        # Certified against the rule-free reference solver in tests.
        dead = True
        for mask in engine.lines:
            if not (mask & cur) or not (mask & opp):
                dead = False
                break
        if dead:
            self.tt[key] = (DRAW, EXACT, -1)
            return DRAW

        if engine.rules.gravity:
            moves = engine.legal_moves((cur, opp))
            moves.sort(key=lambda i: -len(engine.lines_through[i]))
        else:
            moves = [i for i in engine.move_order if not (occ >> i) & 1]
        if tt_move >= 0 and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)

        best = -2
        best_move = -1
        forced_loss_move = -1
        for mv in moves:
            nxt = cur | (1 << mv)
            if completes(nxt, mv):
                if misere:
                    # value LOSS: dominated whenever any non-completing move exists
                    forced_loss_move = mv
                    continue
                v = WIN
            elif (nxt | opp) == full:
                v = DRAW
            else:
                v = -self._search(opp, nxt, -beta, -alpha)
            if v > best:
                best = v
                best_move = mv
            if best > alpha:
                alpha = best
            if alpha >= beta or best == WIN:
                break
        if best == -2:
            # misere: every legal move completes a line -> forced loss
            best = LOSS
            best_move = forced_loss_move

        if best <= alpha_orig:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[key] = (best, flag, best_move)
        return best

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"variant": self.engine.rules.to_dict(), "tt": self.tt},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["variant"] != self.engine.rules.to_dict():
            raise ValueError(
                f"cache variant mismatch: file is {d['variant']}, "
                f"solver is {self.engine.rules.to_dict()}"
            )
        self.tt = d["tt"]
