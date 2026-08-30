# Solver feasibility benchmarks (2026-08-30)

Machine: Apple Silicon laptop, single core, pure Python 3.13. Solver v1 =
negamax + alpha-beta + TT + dead-position rule; v2 adds threat-forced move
restriction in normal play (win now, else block a lone immediate threat; two
open threats = loss). Every completed solve matched its sourced expectation
(docs/known-results.md); no discrepancies.

Plan reality check (docs/plan.md par.3 "all chosen sizes solvable in
minutes"): true for 37 of the checked cells, false in pure Python for the
frontier list below — the risk-register "compute overrun" row, with
mitigations parked (docs/parking-lot.md) and the test suite split into
fast / slow / frontier markers to match.

## Solved (all values correct vs. source)

| variant | value | v1 time | v2 time |
|---|---|---|---|
| m4n4k4 | draw | 0.6s | <0.1s |
| m5n4k4 | draw | 72s | 3.1s |
| m4n5k4 | draw | 85s | 3.0s |
| m6n4k4 | draw | >900s (timeout) | 64s |
| m4n6k4 | draw | — | 50s |
| m4n4k4_grav | draw | 0.02s | — |
| m5n4k4_grav | draw | 0.4s | — |
| m4n5k4_grav | draw | 0.2s | — |
| m4n6k4_grav | draw | 2.5s | — |
| m5n5k4_grav | draw | 5.9s | 2.6s |
| m6n4k4_grav | **second-player win** | 10.4s | 4.2s |
| m6n5k4_grav | draw | >30s | 49s |
| m5n6k4_grav | draw | >30s | 52s |

Plus, in the default suite in seconds: the whole k=3 grid (16 cells), k=4
pairing draws (6), misere 3,3,3 (draw; center the only non-losing opening),
torus 3,3,3 (first-player win; all openings optimal). The v2 threat
restriction also cut the fast test suite from ~12s to ~3.4s and the slow
tier to ~95s total.

## Frontier (timeout under v2; `pytest -m frontier` once the solver improves)

| variant | expected value | best attempt |
|---|---|---|
| m5n5k4 | draw | >120s, 29M nodes |
| m6n5k4 / m5n6k4 | first win | >180s / >90s, 20-39M nodes |
| m6n6k4 | first win | >90s, 19M nodes |
| m6n6k4_grav | second-player win | >400s, 34M nodes |
| m7n6k4_grav (± misere) | first win / second win | not attempted (larger) |

Notes: draw-valued big boards are the structurally hard case (no early
cutoffs); TT memory is the second constraint (~13M entries per ~85s of v1
search). Deep-position labeling for datasets is much cheaper than
initial-position proofs, so experiment grids can include frontier-adjacent
variants for training data even where the initial value is not yet
solver-proved; E1-E5 grid selection should still respect this feasibility map.
