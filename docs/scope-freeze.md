# Scope freeze — harness v1 (frozen 2026-08-30)

Per the project plan (docs/plan.md §6, Track A, Sep w1-2). Changes to this document
require a failed experiment, never a new idea (risk register: scope creep).

## Rule knobs in v1

| Knob | Range | Status |
|---|---|---|
| board m (width, columns) | 3..6 | in |
| board n (height, rows) | 3..6 | in |
| k (in-a-row) | 3, 4 | in |
| gravity | on/off | in |
| misère (goal inversion) | on/off | in |
| torus wrap | on/off | in |
| forbidden-cell masks | any subset of cells | in |
| pie rule | — | **out** (parking lot) |
| double-move | — | **out** (parking lot) |

Grid exclusion: variants with k > max(m, n) are excluded from the standard grid
(no line of length k fits even with wrapping discarded — degenerate all-draw games).

## Frozen conventions

- **m = width (number of columns), n = height (number of rows).**
  Connect Four is m=7, n=6, k=4, gravity.
- Cells are (r, c); **r = 0 is the BOTTOM row** (gravity pulls toward r = 0).
- Cell index = r * m + c. Bitboards use bit i for cell index i.
- **Action space = cell index** in every variant. Under gravity, the legal actions
  are the landing cells of the non-full columns (keeps policy heads uniform across knobs).
- State is (cur, opp) bitboards from the perspective of the player to move.
  Player identity is derivable: popcount(cur) == popcount(opp) iff cur is the first player.
- Value scale is WDL {-1, 0, +1} from the perspective of the player to move.
- Per-move regret = value(best move) - value(move), in {0, 1, 2}.

## Frozen rule semantics

- A winning line is a set of **k distinct cells** in one of 4 directions
  (E, N, NE, NW). Under torus, stepping wraps; a wrapped line that revisits
  a cell is discarded. Without torus, lines leaving the board are discarded.
- Lines containing a forbidden cell are discarded (they can never be completed).
- Two lines with the same cell set are the same line (winning is set ownership).
- **Misère**: the player who completes a k-line LOSES. Line completion takes
  precedence over the board-filling draw when both happen on the same move.
- **Forbidden cells under gravity are solid blocks**: a stone falls to the lowest
  empty playable cell of the column, resting on occupied or forbidden cells.
- Draw = no legal moves (equivalently: all playable cells filled).
- Rule distance = knob edit distance:
  |Δm| + |Δn| + |Δk| + (#boolean flag flips) + |forbidden symmetric difference|.
  A learned/behavioral distance is analysis-time, not part of the harness.

## Solver scope (v1)

- Exact negamax, fail-soft alpha-beta over {-1, 0, +1}, transposition table with
  EXACT/LOWER/UPPER flags and best-move ordering; correctness certified against a
  pruning-free reference solver on small variants across all knob combinations.
- Threat-forced move restriction in normal play (win now, else block a lone
  immediate threat; two open threats are a loss). Not applied under misere.
- Dead-position rule: a position where every line contains stones of both
  players is an exact draw (holds in misere too) and is cut off immediately.
- No symmetry reduction in v1 (parking lot). Pure Python; numba/bitboard
  optimizations only if a needed variant is out of reach (measure first).
- Full-enumeration solution tables only below a configurable reachable-state cap;
  above it, on-demand solving with a persistent per-variant cache.

## Training data scope (v1)

- Supervised distillation targets only: value = exact WDL, policy = uniform over
  the exact optimal-move set. No RL loop.
- Position sampling v1: uniform-random playouts, all path prefixes collected,
  deduped, subsampled with a fixed seed. Optimal/mixed sampling is a later flag.
