# RuleShift

**Concept survival and localized adaptation across game rule spaces.**
Do factorized, rule-conditioned game models adapt to rule changes faster and
more locally than monolithic ones — measured against exact solvers, with
per-concept survival maps showing which learned knowledge transfers and which
dies? Full plan: [docs/plan.md](docs/plan.md).

## Domain

m,n,k-games (tic-tac-toe = 3,3,3) with parameterized rule knobs, each a
coordinate in rule space: board size (m width x n height), k, gravity,
misère (goal inversion), torus wrap, forbidden-cell masks. Ground truth is an
exact negamax solver; the metric is exact per-move regret (WDL-aware).
Conventions and v1 semantics are frozen in
[docs/scope-freeze.md](docs/scope-freeze.md).

## Layout

```
src/ruleshift/
  rules.py     rule space: Ruleset, variant ids, knob edit distance, standard grid
  engine.py    bitboard engine, all knobs (lines precomputed; uniform cell-index actions)
  solver.py    exact negamax + alpha-beta + TT (+ dead-position rule); disk cache
  tables.py    full-enumeration solution tables below a state cap
  metrics.py   exact per-move regret, policy evaluation, samples-to-epsilon
  dataset.py   distillation datasets: exact WDL values + optimal-move sets (npz)
  tracker.py   file-based experiment tracker (config/metrics.jsonl/summary per run)
scripts/
  solve_grid.py    solve + cache a variant grid; feasibility benchmark
  gen_dataset.py   generate a labeled dataset for one variant
docs/
  plan.md            project plan (gates G1-G5)
  scope-freeze.md    frozen v1 conventions and semantics
  parking-lot.md     deferred ideas (reviewed only at gates)
  known-results.md   sourced ground-truth values used by tests
```

## Quickstart

```bash
uv venv .venv && uv pip install -e ".[dev]" -p .venv/bin/python
.venv/bin/pytest                # fast suite
.venv/bin/pytest -m slow        # heavy known-result checks (minutes)
.venv/bin/python scripts/solve_grid.py --gravity --misere --torus
.venv/bin/python scripts/gen_dataset.py --m 4 --n 4 --k 3 --n-positions 1000
```

## Validation (Gate G1)

- Solver certified against a pruning-free reference solver across knob
  combinations (misère x torus x gravity x forbidden), from-initial and midgame.
- Known-result checks against sourced values ([docs/known-results.md](docs/known-results.md)):
  the m,n,k grid (Uiterwijk 2019; Wikipedia), Tromp's Connect Four size table
  (including second-player wins at 6x4 and 6x6), misère tic-tac-toe (draw,
  center-only opening), torus tic-tac-toe (first-player win).
- Regret metric unit-tested on hand-verified forced-block / blunder positions.
