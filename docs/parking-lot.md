# Parking lot

New ideas land here, not in the plan (docs/plan.md). Reviewed only at gates.

## Harness / performance
- Symmetry reduction in the solver TT (dihedral; torus adds translations — big win there).
- Numba/C bitboard solver if a grid variant is out of pure-Python reach (measure first).
- TT memory cap + replacement policy for the largest variants.
- Mixed optimal/ε-random position sampling for datasets (v1 is uniform-random playouts).

## Rule knobs deferred from v1
- Pie rule.
- Double-move (breaks the (cur, opp)-alternation state convention — needs explicit to-move).

## Plan-level (already deferred by the plan itself)
- M3 sparse gating — cut by default, only if G3 lands ≥1 week early with H1 positive.
- Chess-variant chapter (Fairy-Stockfish, engine-eval regret proxy) — December, optional.
- Exhaustive causal audits of concepts — paper two.
- E6 adversarial-policy exploitability audit — successor paper (or pivot if H1 null).

## Next-paper backlog (plan §8)
- Fog-of-war belief states.
- Chess concept engine.
- LLM-composed components.
