"""Bitboard game engine for a Ruleset — one plug-in implementation of
`ruleshift.interface.Game`.

State = (cur, opp): bitboards from the perspective of the player to move.
All values are from the perspective of the player to move: +1 win, 0 draw, -1 loss.

Knob semantics (docs/scope-freeze.md). The two adversarial knobs (A2) change
the game's shape enough to disable the solver's fast paths:

- `capture` (ENTANGLED): placing a stone flips sandwiched opponent runs
  (Othello-style) in all 8 directions. This changes the transition function
  AND the terminal test together — a line can now be created remotely by a
  flip, and an opponent's line can be destroyed — so stone ownership is no
  longer monotonic and threat/dead-position reasoning is unsound.
- `scoring` (DIFFUSE): completing a line no longer ends the game; play runs
  to a full board and the winner is whoever owns more completed lines. Value
  becomes an accumulated differential rather than a localized event.

Occupancy still grows by exactly one cell per move under every knob, so every
game terminates within `num_cells` plies and states stay history-free.
"""
from __future__ import annotations

from typing import Any

from .rules import Ruleset

State = tuple[int, int]

WIN, DRAW, LOSS = 1, 0, -1
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))  # line directions: E, N, NE, NW
_DIRS8 = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


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
        # Fast path = the v1 knob set, where line completion is an immediate,
        # local, monotonic event. The A2 knobs turn it off (see module docstring).
        self.fast_path = not (rules.capture or rules.scoring)

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

    def n_actions(self) -> int:
        return self.rules.num_cells

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

    def flips(self, cur: int, opp: int, move: int) -> int:
        """Opponent stones flipped by playing `move` (capture knob, else 0)."""
        if not self.rules.capture:
            return 0
        rules = self.rules
        m, n = rules.m, rules.n
        r0, c0 = divmod(move, m)
        flipped = 0
        span = max(m, n)
        for dr, dc in _DIRS8:
            run = 0
            r, c = r0, c0
            for _ in range(span):
                r += dr
                c += dc
                if rules.torus:
                    r %= n
                    c %= m
                elif not (0 <= r < n and 0 <= c < m):
                    break
                i = r * m + c
                if (opp >> i) & 1:
                    run |= 1 << i
                    continue
                if run and (cur >> i) & 1:
                    flipped |= run
                break
        return flipped

    def apply(self, state: State, move: int) -> State:
        cur, opp = state
        placed = cur | (1 << move)
        if self.rules.capture:
            f = self.flips(cur, opp, move)
            return (opp & ~f, placed | f)
        return (opp, placed)

    def owns_line(self, bits: int) -> bool:
        for mask in self.lines:
            if bits & mask == mask:
                return True
        return False

    def count_lines(self, bits: int) -> int:
        return sum(1 for mask in self.lines if bits & mask == mask)

    def completes_line(self, bits: int, move: int) -> bool:
        """Fast local check: does `move` complete a line for the owner of `bits`?

        Valid only when ownership is monotonic (no capture) — under capture use
        `owns_line` on the post-flip board.
        """
        for mask in self.lines_through[move]:
            if bits & mask == mask:
                return True
        return False

    def _score_value(self, mover_bits: int, other_bits: int) -> int:
        """Terminal value for the mover under the scoring knob."""
        diff = self.count_lines(mover_bits) - self.count_lines(other_bits)
        v = (diff > 0) - (diff < 0)
        return -v if self.rules.misere else v

    def step(self, state: State, move: int) -> tuple[State, int | None]:
        """(next_state, terminal value for the player to move in next_state, or None).

        The single source of truth for transitions and terminality under every
        knob; `ruleshift.interface.Game.step`.
        """
        nxt = self.apply(state, move)
        opp_after, mover_after = nxt  # mover_after = stones of the player who just moved
        occupied_full = (opp_after | mover_after) == self.full
        if self.rules.scoring:
            if occupied_full:
                return nxt, -self._score_value(mover_after, opp_after)
            return nxt, None
        completed = (
            self.owns_line(mover_after)
            if self.rules.capture
            else self.completes_line(mover_after, move)
        )
        if completed:
            return nxt, (WIN if self.rules.misere else LOSS)
        if occupied_full:
            return nxt, DRAW
        return nxt, None

    def status_after(self, state: State, last_move: int) -> int | None:
        """Terminal value of `state` for its player to move, given the opponent
        just played last_move; None if non-terminal. Fast-path knobs only —
        general callers should use `step` (which needs the pre-move state)."""
        cur, opp = state
        if not self.fast_path:
            raise ValueError("status_after is fast-path only; use step() for A2 knobs")
        if self.completes_line(opp, last_move):
            return WIN if self.rules.misere else LOSS
        if (cur | opp) == self.full:
            return DRAW
        return None

    def full_status(self, state: State) -> int | None:
        """Terminal value for the player to move, without last-move knowledge."""
        cur, opp = state
        if self.rules.scoring:
            if (cur | opp) == self.full:
                return self._score_value(cur, opp)
            return None
        if self.owns_line(opp):
            return WIN if self.rules.misere else LOSS
        if self.owns_line(cur):
            if self.rules.capture:
                # reachable only as a just-decided position; value for the mover
                return LOSS if self.rules.misere else WIN
            raise ValueError("unreachable state: player to move holds a completed line")
        if (cur | opp) == self.full:
            return DRAW
        return None

    # --------------------------------------------------------------- helpers
    def first_player_to_move(self, state: State) -> bool:
        """True if the player to move opened the game.

        Derived from OCCUPANCY parity, not from stone counts: every move fills
        exactly one empty cell under every knob, whereas `capture` moves stones
        between owners, so `popcount(cur) == popcount(opp)` is false there.
        """
        cur, opp = state
        return (cur | opp).bit_count() % 2 == 0

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
            s, status = self.step(s, mv)
        return s, status

    def rule_descriptor(self) -> dict[str, Any]:
        """Knob-free description of this game's rules (A3 mode 2).

        Delegates to `ruleshift.descriptor`, which derives everything from the
        engine's own semantics rather than from the Ruleset's knobs.
        """
        from .descriptor import rule_description

        return rule_description(self)

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
