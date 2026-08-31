# Limitations and scope statement

Carried into the paper's limitations section (amendment A6).

## Scope of the claim

**The claim holds for rule changes expressible in the tested rule space. No
claim is made for arbitrary rule modifications.** Concretely: the results
characterize adaptation across the knob space of §3 (board geometry, k,
gravity, misère, torus wrap, forbidden masks, capture, scoring) on
two-player, perfect-information, deterministic games with exact solutions.
They do not establish anything about rule changes outside that space —
stochastic transitions, hidden information, continuous action spaces, or
rule edits that change the number of players or the objective type.

## Known threats to validity, and what we do about them

1. **The rule formalism is ours** (objection: results may only apply to models
   trained in our parameterization). Mitigations: the harness is written
   against a formalism-agnostic game interface (A5, `ruleshift.interface`),
   with m,n,k as one plug-in implementation; an out-of-formalism family
   (Fairy-Stockfish / Ludii) is an optional external-validity check.

2. **Knob design may align with the architecture's factorization** — if the
   goal knob maps onto the value head and dynamics knobs onto the transition
   module, a factorized model wins by construction, and the result would be
   about knob design rather than factorization. Mitigations:
   - **A1**: the headline distance axis is *solver-grounded* (divergence of
     optimal play), not knob-edit count, so the x-axis is ground truth rather
     than our parameterization.
   - **A2**: two adversarial knobs are included by design — one entangled
     (dynamics *and* win condition change together: `capture`) and one whose
     effect on optimal play is diffuse rather than localized (`scoring`).
     These cut across the architecture's seams. If factorization still helps
     there, the claim is stronger; if it fails, that is the honest boundary
     and is reported as such.
   - **A3 (E2b)**: the conditioning ablation quantifies how much the
     hand-designed parameterization contributes, by comparing an explicit
     knob vector against a general, knob-free rule description.

3. **Solver feasibility bounds the grid** (docs/benchmarks.md): variants whose
   exact labeling is intractable are excluded, so the grid is feasibility-gated
   rather than uniformly sampled.
