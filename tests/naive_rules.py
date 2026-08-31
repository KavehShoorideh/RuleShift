"""Deliberately naive, independent reimplementation of the rule semantics.

Written FROM THE SPEC (docs/scope-freeze.md), not from engine.py, and in a
different style on purpose: a 2-D grid of ints with explicit (r, c) loops,
no bitboards, no precomputation, no optimizations. Slow and obvious.

Its only job is to disagree with `ruleshift.engine` if either is wrong. The
existing reference solver shares the engine, so it certifies the SEARCH; this
certifies the RULES.
"""
from __future__ import annotations

EMPTY, P1, P2 = 0, 1, 2
DIRS4 = [(0, 1), (1, 0), (1, 1), (1, -1)]
DIRS8 = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

# state = (grid, turn); grid is a tuple of row-tuples, row 0 = bottom
NaiveState = tuple


class NaiveGame:
    def __init__(self, rules):
        self.r = rules
        self._lines = None

    # ---------------------------------------------------------------- setup
    def initial(self) -> NaiveState:
        grid = tuple(tuple(EMPTY for _ in range(self.r.m)) for _ in range(self.r.n))
        return (grid, P1)

    def _wrap(self, rr, cc):
        if self.r.torus:
            return rr % self.r.n, cc % self.r.m
        if 0 <= rr < self.r.n and 0 <= cc < self.r.m:
            return rr, cc
        return None

    def lines(self) -> list[frozenset]:
        if self._lines is not None:
            return self._lines
        out = set()
        for r0 in range(self.r.n):
            for c0 in range(self.r.m):
                for dr, dc in DIRS4:
                    cells, ok = [], True
                    for i in range(self.r.k):
                        pos = self._wrap(r0 + i * dr, c0 + i * dc)
                        if pos is None or pos in self.r.forbidden:
                            ok = False
                            break
                        cells.append(pos)
                    if ok and len(set(cells)) == self.r.k:
                        out.add(frozenset(cells))
        self._lines = sorted(out, key=lambda f: sorted(f))
        return self._lines

    # ----------------------------------------------------------------- play
    def legal_moves(self, state: NaiveState) -> list[tuple[int, int]]:
        grid, _ = state
        out = []
        for c in range(self.r.m):
            for r in range(self.r.n):
                if (r, c) in self.r.forbidden or grid[r][c] != EMPTY:
                    continue
                out.append((r, c))
                if self.r.gravity:
                    break  # only the lowest open cell of each column
        return sorted(out)

    def _set(self, grid, cells, who):
        rows = [list(row) for row in grid]
        for r, c in cells:
            rows[r][c] = who
        return tuple(tuple(row) for row in rows)

    def flipped_by(self, state: NaiveState, move) -> list[tuple[int, int]]:
        """Cells captured by playing `move` (Othello sandwich, all 8 directions)."""
        if not self.r.capture:
            return []
        grid, turn = state
        other = P2 if turn == P1 else P1
        out = []
        for dr, dc in DIRS8:
            run, r, c = [], move[0], move[1]
            for _ in range(max(self.r.m, self.r.n)):
                pos = self._wrap(r + dr, c + dc)
                if pos is None:
                    break
                r, c = pos
                if grid[r][c] == other:
                    run.append((r, c))
                    continue
                if grid[r][c] == turn and run:
                    out.extend(run)
                break
        return out

    def apply(self, state: NaiveState, move) -> NaiveState:
        grid, turn = state
        grid = self._set(grid, [move] + self.flipped_by(state, move), turn)
        return (grid, P2 if turn == P1 else P1)

    # ------------------------------------------------------------- terminal
    def owner_cells(self, grid, who):
        return {(r, c) for r in range(self.r.n) for c in range(self.r.m) if grid[r][c] == who}

    def count_lines(self, grid, who) -> int:
        own = self.owner_cells(grid, who)
        return sum(1 for line in self.lines() if line <= own)

    def status(self, state: NaiveState) -> int | None:
        """Value for the player to move in `state`, or None if not terminal."""
        grid, turn = state
        other = P2 if turn == P1 else P1
        if self.r.scoring:
            if self.legal_moves(state):
                return None
            diff = self.count_lines(grid, turn) - self.count_lines(grid, other)
            v = (diff > 0) - (diff < 0)
            return -v if self.r.misere else v
        if self.count_lines(grid, other) > 0:
            return 1 if self.r.misere else -1
        if self.count_lines(grid, turn) > 0:
            return -1 if self.r.misere else 1  # only reachable via capture
        if not self.legal_moves(state):
            return 0
        return None

    def value(self, state: NaiveState, memo=None) -> int:
        """Plain memoized negamax over the naive rules."""
        if memo is None:
            memo = {}
        st = self.status(state)
        if st is not None:
            return st
        if state in memo:
            return memo[state]
        best = -2
        for mv in self.legal_moves(state):
            best = max(best, -self.value(self.apply(state, mv), memo))
        memo[state] = best
        return best


# ------------------------------------------------------ representation bridge
def to_naive(engine, state) -> NaiveState:
    """(cur, opp) bitboards -> (grid, turn). The mover is P1 iff they moved first."""
    cur, opp = state
    m = engine.rules.m
    mover_first = engine.first_player_to_move(state)
    turn = P1 if mover_first else P2
    mover, other = (P1, P2) if mover_first else (P2, P1)
    rows = [[EMPTY] * m for _ in range(engine.rules.n)]
    for bits, who in ((cur, mover), (opp, other)):
        i = 0
        while bits:
            if bits & 1:
                r, c = divmod(i, m)
                rows[r][c] = who
            bits >>= 1
            i += 1
    return (tuple(tuple(row) for row in rows), turn)


def naive_move(engine, move: int) -> tuple[int, int]:
    return divmod(move, engine.rules.m)
