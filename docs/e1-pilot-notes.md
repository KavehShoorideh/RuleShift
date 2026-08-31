# E1 pilot findings (2026-08-30)

Pilot run: M0 (MLP 256x3, concat rule vector), 20 training variants (k=3,
feasibility-gated), 10 held-out (6 interp / 4 extrap), 2 seeds, exact-regret
eval on positions disjoint from every adaptation pool. Artifacts:
figures/e1_pilot_*.png, figures/e1_pilot_results.json; full protocol in
scripts/e1_baseline.py. Numbers below are means over seeds.

1. **Zero-shot transfer is real and knob-dependent.** Gravity held-outs
   transfer almost perfectly zero-shot (regret 0.07-0.08 vs random ~0.75);
   torus sits in the middle (0.45-0.52 vs random ~1.0); misere barely beats
   random (0.45 vs 0.5). Dynamics-type knobs transfer, the goal-inversion
   knob kills value knowledge -- the E5 dissociation is already visible in
   the monolithic baseline.

2. **A negative-transfer case, on schedule.** m5n4k3_grav_mis (distance 2,
   unseen gravity+misere combo): zero-shot regret 0.47 vs random 0.37 --
   *worse than random play*. The pretrained model confidently plays
   line-completing moves that now lose. This mirrors the Hex<->Misere-Hex
   negative transfer the novelty table cites (arXiv:2107.01078) and is
   exactly the phenomenon H1-H3 are built to dissect.

3. **The 10-sample dip.** Naive fine-tuning (500 steps) on only 10 positions
   makes regret *worse* than zero-shot (interp 0.34 -> 0.49) before 100+
   samples recover it. Protocol decision for full E1 at G2: scale adaptation
   steps with budget and/or early-stop; alternatively keep the dip -- it is a
   real property of naive monolithic fine-tuning and a natural foil for the
   factorized models (H2 predicts localized updates should not suffer it).
   Either way, decide once and apply to every model identically.

4. **Pretraining beats scratch at low budgets, converges by 1000.**
   Fine-tune@100 = 0.31-0.34 vs scratch@100 = 0.51; by 1000 samples both
   reach ~0.13-0.16. The sample-efficiency gap H1 cares about lives in the
   10-1000 range on these variants -- the budget grid {10, 100, 1000} brackets
   it well; consider adding 30 and 300.

Pilot caveats: 2 seeds (plan requires >=5 for reported numbers); k=3 only;
misere/torus capped at 16-cell boards pending solver speedups; distances only
span 1-2 (k and board-size axes will widen the range in the full grid).

## Decisions taken for full E1 (2026-08-30, applied identically to all models)

- Budget grid: {0, 10, 30, 100, 300, 1000}; scratch controls at {30, 100, 300, 1000}.
- Adaptation: steps = min(25 x n, 2500) at lr 3e-4 (dip mitigation, item 3).
- Scratch: steps = min(50 x n, 4000).
- Grid extended with feasible k=4 cells (plain, gravity, one misere) in train
  and 4 more held-outs, widening the rule-distance axis to the k knob.
- 5 seeds (plan minimum for reported numbers).
