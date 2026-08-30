"""Bitboard game engine for a Ruleset.

State = (cur, opp): bitboards from the perspective of the player to move.
apply() returns (opp, cur | bit) -- perspectives swap every ply.
All values are from the perspective of the player to move: +1 win, 0 draw, -1 loss.
"""
from __future__ import annotations

from .rules import Ruleset

State = tuple[int, int]

WIN, DRAW, LOSS = 1, 0, -1
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))  # (dr, dc): E, N, NE, NW


class Engine:
    def __init__(self, rules: Ruleset):
        self.rules = rules
        m, n = rules.m, rules.n
        self.playable = tuple(rules.cell_index(r, c) for (r, c) in rules.playable_cells())
        full = 0
        for i in self.playable:
            full |= 1 << i
        self.full = full
        self.lines = self._gen_lines()
        lt: list[list[int]] = [[] for _ in range(m * n)]
        for mask in self.lines:
            mm = mask
            while mm:
                b = mm & -mm
                lt[b.bit_length() - 1].append(mask)
                mm ^= b
        self.lines_through = tuple(tuple(x) for x in lt)
        # static move order: cells on more lines first (center-out heuristic)
        self.move_order = tuple(sorted(self.playable, key=lambda i: -len(self.lines_through[i])))
        cols: list[list[int]] = [[] for _ in range(m)]
        for r in range(n):
            for c in range(m):
                if (r, c) not in rules.forbidden:
                    cols[c].append(rules.cell_index(r, c))
        self.col_cells = tuple(tuple(x) for x in cols)

    def _gen_lines(self) -> tuple[int, ...]:
        """All winning lines as bitmasks: k distinct cells in one of 4 directions.

        Torus: stepping wraps; a wrapped line revisiting a cell is discarded.
        Lines containing forbidden cells are discarded (never completable).
        Duplicate cell sets are deduped (winning is set ownership).
        """
        rules = self.rules
        m, n, k = rules.m, rules.n, rules.k
        masks = set()
        for r in range(n):
            for c in range(m):
                for dr, dc in _DIRS:
                    cells = []
                    ok = True
                    for i in range(k):
                        rr, cc = r + i * dr, c + i * dc
                        if rules.torus:
                            rr, cc = rr % n, cc % m
                        elif not (0 <= rr < n and 0 <= cc < m):
                            ok = False
                            break
                        cells.append((rr, cc))
                    if not ok or len(set(cells)) < k:
                        continue
                    if any(cell in rules.forbidden for cell in cells):
                        continue
                    mask = 0
                    for rr, cc in cells:
                        mask |= 1 << rules.cell_index(rr, cc)
                    masks.add(mask)
        return tuple(sorted(masks))

    # ------------------------------------------------------------------ play
    def initial(self) -> State:
        return (0, 0)

    def legal_moves(self, state: State) -> list[int]:
        """Legal actions as cell indices. Under gravity: landing cells of
        non-full columns (forbidden cells act as solid blocks)."""
        occ = state[0] | state[1]
        if self.rules.gravity:
            out = []
            for col in self.col_cells:
                for i in col:
                    if not (occ >> i) & 1:
                        out.append(i)
                        break
            return out
        return [i for i in self.playable if not (occ >> i) & 1]

    def apply(self, state: State, move: int) -> State:
        cur, opp = state
        return (opp, cur | (1 << move))

    def completes_line(self, bits: int, move: int) -> bool:
        for mask in self.lines_through[move]:
            if bits & mask == mask:
                return True
        return False

    def status_after(self, state: State, last_move: int) -> int | None:
        """Terminal value of `state` for its player to move, given the opponent
        just played last_move; None if non-terminal. Line completion takes
        precedence over the board-filling draw."""
        cur, opp = state
        if self.completes_line(opp, last_move):
            return WIN if self.rules.misere else LOSS
        if (cur | opp) == self.full:
            return DRAW
        return None

    def full_status(self, state: State) -> int | None:
        """Terminal check without last-move knowledge (entry points only).

        Raises on unreachable states (player to move already holds a line --
        impossible, since line completion ends the game immediately)."""
        cur, opp = state
        for mask in self.lines:
            if cur & mask == mask:
                raise ValueError("unreachable state: player to move holds a completed line")
        for mask in self.lines:
            if opp & mask == mask:
                return WIN if self.rules.misere else LOSS
        if (cur | opp) == self.full:
            return DRAW
        return None

    # --------------------------------------------------------------- helpers
    def first_player_to_move(self, state: State) -> bool:
        cur, opp = state
        return cur.bit_count() == opp.bit_count()

    def play(self, moves: list[int]) -> tuple[State, int | None]:
        """Apply a move sequence from the initial state; returns (state, status).
        status is the terminal value for the player to move after the last move,
        or None. Raises if a move is illegal or made after the game ended."""
        s: State = self.initial()
        status: int | None = None
        for mv in moves:
            if status is not None:
                raise ValueError("move after game end")
            if mv not in self.legal_moves(s):
                raise ValueError(f"illegal move {mv}")
            s = self.apply(s, mv)
            status = self.status_after(s, mv)
        return s, status

    def render(self, state: State) -> str:
        """ASCII board: X first player, O second, # forbidden, . empty. Top row first."""
        cur, opp = state
        x_bits, o_bits = (cur, opp) if self.first_player_to_move(state) else (opp, cur)
        rows = []
        for r in range(self.rules.n - 1, -1, -1):
            row = []
            for c in range(self.rules.m):
                i = self.rules.cell_index(r, c)
                if (r, c) in self.rules.forbidden:
                    row.append("#")
                elif (x_bits >> i) & 1:
                    row.append("X")
                elif (o_bits >> i) & 1:
                    row.append("O")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)
