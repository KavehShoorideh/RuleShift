# Technical reference

The scientific/engineering companion to the README. Conventions here are
frozen in [scope-freeze.md](scope-freeze.md); change them only per the plan's
scope rules.

## Domain

m,n,k-games (tic-tac-toe = 3,3,3) with parameterized rule knobs, each a
coordinate in rule space: board size (m width x n height), k-in-a-row,
gravity, misère (goal inversion), torus wrap, forbidden-cell masks.
Rule distance = knob edit distance. Ground truth is an exact negamax solver;
the metric is exact per-move regret (WDL-aware, in {0, 1, 2}).

## Layout

```
src/ruleshift/
  rules.py     rule space: Ruleset, variant ids, knob edit distance, standard grid
  engine.py    bitboard engine, all knobs (lines precomputed; uniform cell-index actions)
  solver.py    exact negamax + alpha-beta + TT + dead-position rule
               + threat-forced move restriction; disk cache
  tables.py    full-enumeration solution tables below a state cap
  metrics.py   exact per-move regret, policy evaluation, samples-to-epsilon
  dataset.py   distillation datasets: exact WDL values + optimal-move sets (npz)
  tracker.py   file-based experiment tracker (config/metrics.jsonl/summary per run)
scripts/
  solve_grid.py    solve + cache a variant grid; feasibility benchmark
  gen_dataset.py   generate a labeled dataset for one variant
```

## Key conventions (full list in scope-freeze.md)

- m = width (columns), n = height (rows); r = 0 is the bottom row.
- Action space = cell index in every variant; under gravity the legal actions
  are the landing cells of non-full columns (uniform policy heads across knobs).
- State = (cur, opp) bitboards from the player to move's perspective.
- Values are WDL {-1, 0, +1} from the player to move's perspective.

## Validation (Gate G1)

- Solver certified against a pruning-free reference solver across knob
  combinations (misère x torus x gravity x forbidden), from-initial and
  midgame (tests/reference.py, tests/test_solver.py).
- Known-result checks against sourced values ([known-results.md](known-results.md)):
  the m,n,k grid (Uiterwijk 2019; Wikipedia), Tromp's Connect Four size table
  (including second-player wins at 6x4 and 6x6), misère tic-tac-toe (draw,
  center-only opening), torus tic-tac-toe (first-player win). 37 cells
  validated, zero discrepancies; feasibility map in [benchmarks.md](benchmarks.md).
- Regret metric unit-tested on hand-verified forced-block / blunder positions.

## Test tiers

```bash
.venv/bin/pytest                # fast suite (~5s): certification + quick known results
.venv/bin/pytest -m slow        # heavier known-result checks (~95s total)
.venv/bin/pytest -m frontier    # boards beyond pure-Python reach today (will time out
                                # until the parked solver optimizations land)
```
