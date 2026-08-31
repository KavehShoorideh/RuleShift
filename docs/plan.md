# Project Plan — Concept Survival and Localized Adaptation Across Game Rule Spaces

**One-line claim:** Factorized, rule-conditioned game models adapt to rule changes faster and more locally than monolithic ones — measured against exact solvers, with per-concept survival maps showing which learned knowledge transfers and which dies.

**Target:** arXiv preprint by end of January 2027; ICLR 2027 workshop submissions (~early Feb); backups: RL Journal (rolling), CoRL 2027, AAMAS. Robotics-legible framing throughout (world-model adaptation under environment change).

---

## 1. Hypotheses (primary → secondary)

- **H1 (primary, few-shot):** Rule-conditioned factorized models (GNN + rule embeddings, optionally sparse-gated) reach a given regret threshold on a modified-rule variant in fewer adaptation samples than parameter-matched monolithic baselines, and the gap grows with rule distance.
- **H2 (localization):** The adaptation delta in factorized models is localized (few components/low-rank change, e.g. value head flips under goal inversion) while monolithic models change diffusely. Measured via per-module weight/representation drift and freeze-ablation adaptation curves.
- **H3 (concept survival):** Learned concepts (slot supports / probes / SAE features) can be individually scored for survival across each rule knob; survival predicts transfer (concepts that survive → components that need no retraining). Misère flips value-linked concepts while structural concepts survive.
- **Secondary (reported either way, no dependence):** zero-shot regret vs. rule distance.

Any H1 outcome is publishable: positive → factorization thesis confirmed in adversarial games; negative → contradicts Schema-Networks/DMA*-SH intuition in a new setting, explained via H3 maps.

**Framing decision (locked at G3, Oct 31):** the same results support two headlines — (a) world-model transfer across rule changes, or (b) reliability/interpretability: "when does learned knowledge stop being trustworthy after the world changes, and can we tell from the inside" (concept faithfulness under shift, audited against exact ground truth). Default to (b) unless G3 results favor (a); the robotics-bridge section is written under either headline.

**Designated successor (paper two, reuses the entire harness):** E6 — exact-exploitability audit of adversarial policies. Train attacker policies against each victim model (per Gleave/Wang adversarial-policies line, arXiv:2211.00241; defenses evaluated in low-dim envs per arXiv:2208.05083); measure exact exploitability via the solver instead of proxy win rates; use concept supports to characterize victim blindspots; test whether factorized/concept-structured victims are less exploitable. **Pivot right:** if H1 is null at G3, E6 is promoted to the primary paper — it stands alone regardless of the transfer result.

## 2. Novelty audit (checked 2026-08-30; re-verify before submission)

| Prior art | What they did | Our delta |
|---|---|---|
| Soemers et al. 2021 (arXiv:2102.12375), Polygames/Ludii | Transfer conv policy-value nets across games/variants (mostly board sizes); win-rate metric; notes negative transfer on goal inversion (Hex↔Misère Hex, in arXiv:2107.01078) | Rule-conditioned + factorized architectures; exact-solver regret; adaptation-delta localization; goal-inversion treated as probe, not anecdote |
| Schema Networks, Kansky et al. ICML 2017 (arXiv:1706.04317) | Hand-structured object-oriented causal model; zero-shot on Breakout variations vs A3C | Learned (not hand-built) concepts; adversarial 2-player games; parameterized rule grid; concept-survival measurement |
| DMA*-SH / Actuator Inversion Benchmark, 2026 (arXiv:2602.06550) | Hypernetwork-factorized context conditioning beats concat under discontinuous dynamics flips (continuous control) | Adversarial board games; rule-space (not actuator) shifts; exact ground truth; concept-level interpretability; cite as strongest adjacent work |
| GateL0RD (arXiv:2110.15949), VSG (NeurIPS 2022) | Sparse latent gating → generalization + interpretable latents | Gating linked to *rule-change transfer* and per-concept survival; games not control |
| Probing Transfer without Task Engineering (arXiv:2210.12448) | ANOVA of zero-shot transfer across human game variants (Atari) | Controlled rule knobs; exact regret; architecture comparison (they study one net); adaptation localization |
| Rusu/Braylan module reuse; Compete-and-Compose; RIMs/NPS | Module recombination across environments | Board-game rule spaces; solver ground truth; concept maps |
| GGP value-fn transfer (Banerjee & Stone 2006 etc.) | Symbolic feature-based transfer | Learned neural concepts; systematic rule grid |
| Contextual RL (HyperZero arXiv:2211.15457; Beukman 2023; CARL/Kirk et al. protocol) | Context-conditioned zero-shot in control | Games with exact solvers; discrete combinatorial rules; concepts |
| Concept-probe caveats (CAV faithfulness, 2025-26) | Probe accuracy ≠ faithfulness; spurious correlations | Dictates method: concept survival must pair probes with causal checks (occlusion/patching), not probe accuracy alone |

**Re-check gate (Dec):** one fresh sweep for 2026 papers on rule-space transfer in games before writing the related-work section.

## 3. Domain and rule space

> **Amended by `docs/plan-amendments.md` (A1, A2, A4, A5):** rule distance is
> now solver-grounded (divergence of optimal play), two adversarial knobs
> (`capture`, `scoring`) are in scope, the largest board tier is dropped, and
> the harness targets a formalism-agnostic game interface.

- Base family: m,n,k-games (tic-tac-toe = 3,3,3). Rule knobs (each a coordinate in rule space):
  - board (m,n) ∈ {3..6}², k ∈ {3,4}; gravity on/off (Connect-Four-ification); misère on/off (goal inversion); torus wrap on/off; forbidden-cell masks; optional: pie rule, double-move.
- Rule distance: edit distance in knob space (and a learned/behavioral distance as analysis).
- Ground truth: exact negamax with transposition tables, cached per-variant to disk. Feasibility measured 2026-08-30 (docs/benchmarks.md): most of the grid solves in seconds-to-minutes; frontier boards (5,5,4; the k=4 win boards; 6x6 gravity; 7x6) exceed laptop budgets in pure Python. **The experiment grid is feasibility-gated: only variants with measured-tractable exact labeling enter training/eval.**
- Metric: exact per-move regret vs. optimal value (win/draw/loss-aware), plus sample-efficiency curves (adaptation samples to reach regret ≤ ε).
- Ground-truth provenance (for the methods section): 37 grid cells validated against published results (docs/known-results.md); gravity k=3 and knob combinations (e.g. misère+torus) have no published values anywhere — there the solver is the first word, self-certified against a pruning-free reference solver.
- Scale-up chapter (only if core done, December): 2-3 chess variants via Fairy-Stockfish, engine-eval regret proxy.

## 4. Models (all parameter-matched, 2-3 sizes each)

1. **M0 monolithic:** MLP and/or small transformer on board tensor + rule vector concatenated (concat baseline mirrors DMA*-SH's control).
2. **M1 rule-conditioned GNN:** cells/stones as nodes; rule knobs enter via edge construction (movement/adjacency from rules) + FiLM/context conditioning; pointer-style action head.
3. **M2 = M1 + factorized heads:** separate dynamics / value / policy modules with a rule-conditioning pathway per module (hypernetwork or FiLM), so goal knobs can only enter through the value path — the architectural bet H2 tests.
4. **M3 = M2 + sparse gating (stretch, only if Oct milestone met):** architectural touch-set gate on per-cell/per-concept updates.

Training: supervised distillation from exact solutions (policy = optimal-move set, value = exact WDL) — no RL loop needed; cheapest, lowest-variance path. Multi-variant batches over a training subset of the rule grid; held-out variants split by knob (interpolation) and by knob combination (extrapolation).

## 5. Experiment matrix

> **Amended by `docs/plan-amendments.md` (A3):** adds E2b, the rule-conditioning
> ablation (explicit knob vector vs. knob-free rule description) on M0 and M2 only.

- E1: baseline transfer curves (M0): zero-shot / few-shot (10, 100, 1k, 10k positions) / from-scratch, per held-out variant.
- E2: same for M1, M2 (M3 stretch). Primary plot: adaptation-samples-to-ε vs. rule distance, per model.
- E3: misère probe: adaptation curves under goal inversion only; freeze-ablations (value head only vs. full fine-tune vs. LoRA-rank sweep) to localize the delta.
- E4: concept survival: train probes for exact, solver-derived concepts (threats-in-1, open-line counts, fork existence, tempo) + slot/SAE features; measure firing consistency and value-predictiveness per variant. **Scope cap:** causal checks (occlusion/patching) on the top ~5 concepts per variant only — exhaustive causal auditing is paper-two material.
- E5: dynamics-shift vs. goal-shift dissociation (gravity/wrap vs. misère). **No new training runs** — this is an analysis slice of E2/E3 data by knob type.
- E6 (successor paper, not in this timeline): adversarial-policy exploitability audit — see §1.
- Stats: ≥5 seeds per cell; report CIs; ANOVA across knobs (mirrors arXiv:2210.12448 methodology).

## 6. Timeline — three parallel tracks (gates in bold)

**Principle: no track gates another.** The research track runs on its own clock; the job track runs continuously from September; the research-support track removes the "solo publication" risk on a schedule. An offer landing early doesn't pause the paper; the paper landing doesn't pause applications.

### Track A — Research

- **Sep w1-2:** harness — game engine + knobs, exact solver + cached tables, dataset generator, eval suite, experiment tracker. Scope-freeze doc + parking-lot file. **Gate G1 (Sep 14): solver validated against known m,n,k results; regret metric unit-tested.** *(Done 2026-08-30 — G1 met early: 37 sourced cells validated, zero discrepancies. M0/E1 starts immediately; the banked ~2 weeks cushion November, the flagged collision month.)*
- **Sep w3-4:** M0 trained across grid; E1 complete. **Gate G2 (Sep 30): baseline transfer curves plotted. No new architecture before G2.** Also at G2: keep-or-drop decision on the frontier boards (default: drop; spend 1-2 days on symmetry reduction only if E1 design needs the big boards — docs/benchmarks.md, docs/parking-lot.md).
- **Oct:** M1, M2; E2. **Gate G3 (Oct 31): H1 answerable from data in hand (either direction); framing headline locked (§1); pivot decision if H1 null (promote E6).** M3 (sparse gating) is **cut by default** — build only if G3 lands ≥1 week early with H1 positive.
- **Nov (priority order — lower items are the cut line, not the top):**
  1. w1: paper skeleton drafted (intro, related work from §2 table, method) — writing starts now, not December.
  2. w1-2: E3 misère freeze-ablations (small runs, biggest headline value).
  3. w2-3: E4 concept survival at capped scope (§5).
  4. w3-4: E5 analysis slice; figures assembled as results land, not batched at month end.
  5. Track C collaborator emails go out w2 (needs only G3 plots, not E3/E4).
  **Gate G4 (Nov 30): figures for E1-E3 final; E4 at minimum viable (top-5 concepts); anything unfinished below the line moves to paper-two, not to December.**
- **Dec:** paper draft; robotics-bridge section (sparse/local adaptation ↔ object-centric world models; cite Dreamer/VSG/DMA*-SH); fresh prior-art re-sweep; optional chess-variant chapter. Red-team pass incorporating external feedback (Track C).
- **Jan:** polish; arXiv preprint; repo + 5-minute blog post. **Gate G5 (Jan 31): preprint live.**
- **Feb:** ICLR 2027 workshop submissions (verify exact deadlines when workshops announced ~Dec).

Weekly cadence: one experiment goal per week; results logged even when negative; any new architecture idea → parking lot, not the plan.

### Track B — Job search (RE/MTS at world-model / robotics / AV orgs; MLOps roles as floor)

- **Sep w1:** resume updated with the project as work-in-progress — one crisp line: claim, method, "results expected Jan 2027." LinkedIn/profile match. Target list drafted: Seattle-area first (AV research orgs, NVIDIA robotics/sim, Amazon robotics, embodied-AI startups), remote-friendly research labs second. Distinguish two role tiers per company: research-adjacent RE/MTS (primary) vs. platform MLOps (floor).
- **Sep onward, standing cadence:** 3-5 tailored applications/week to primary-tier roles; referral hunt through LeRobot and past colleagues before every cold application. Interview prep kit: the 90-second project pitch (chess-free version), the robotics-bridge story, infra war stories.
- **Oct 15 checkpoint:** if response rate < ~10%, revise materials (likely the pitch, not the resume) rather than raising volume.
- **Nov onward:** interviews carry preliminary plots (E1/E2 curves) — in-progress results are demo material. If an offer lands: negotiate start date that protects Dec-Jan writing where possible; do NOT drop the paper — finish it with new colleagues as feedback channel.
- **Jan 31+:** preprint replaces "in progress" everywhere; re-ping recruiters/hiring managers from stalled Fall threads with the link (legitimate re-engagement trigger).

### Track C — Research support (de-risking solo publication)

- **Sep:** join ML Collective (or equivalent open research community); lurk → present the project plan in their session format when comfortable. Post the one-paragraph project pitch in the LeRobot community channel — flag it as adjacent (transfer/world-models), ask who else cares.
- **Oct:** identify 3 named potential feedback-givers/collaborators from the prior-art table: Ludii transfer authors (Soemers et al.), GateL0RD/DMA*-SH group (Butz/Martius orbit), one Ludii-concepts author. Draft (don't send) the cold emails.
- **Nov w2 (after G3, with plots in hand):** send the cold emails — concrete results attached, one specific question each, explicit offer: feedback, co-authorship if they engage deeply, or nothing. Concrete-results-first is what makes small academic groups respond.
- **Dec:** at least two external readers on the draft (community members count; collaborators better). Their calibration feedback = the red-team input for the Dec pass.
- **Feb:** whoever gave substantive input → acknowledged or co-author per contribution; workshop poster/talk becomes the in-person networking artifact.

### Cross-track failure handling

- Paper slips past Jan: job track unaffected (it never depended on the preprint); submit to the next workshop cycle (ICML ~May) — deadlines recur, the artifact doesn't expire.
- Offer lands before paper: take it if it's the right seat; timeline compresses but gates G4-G5 survive on evenings/weekends because the compute is laptop-scale by design.
- No collaborator bites: proceed solo — the plan was built to survive that (exact ground truth, workshops, pre-tabled deltas); ML Collective feedback alone is sufficient external calibration.

## 7. Risk register (impossible-problem guards)

| Risk | Signal | Mitigation / pivot |
|---|---|---|
| Zero-shot ≈ chance everywhere | E1/E2 zero-shot flat | Expected-possible; primary claim is few-shot H1. Zero-shot reported as secondary finding. |
| No H1 gap (factorization doesn't help) | E2 curves overlap | Publishable negative result vs. Schema-Nets/DMA*-SH intuition; H3 maps explain; reframe paper around the dissociation. |
| All models adapt instantly (tasks too easy) | curves saturate at ~0 samples | Grow boards/k; add knob combinations; extrapolation split is the hard split by design. |
| Concept probes unfaithful | probe-vs-occlusion disagreement | Already planned: causal checks are part of E4; disagreement is itself a finding. |
| Matched-params disputed | reviewer concern | 2-3 sizes per model; report params + FLOPs; freeze-ablation controls. |
| Scope creep | parking-lot file growing into the plan | Gates G1-G4 immovable; plan changes only on failed experiment, never on new idea. |
| Compute overrun | runs > minutes | Confirmed 2026-08-30 for the feasibility-gated grid (docs/benchmarks.md); frontier boards excluded per §3; chess chapter is the only cloud item and is optional. |
| Scooped | Dec re-sweep finds overlap | Deltas vs. each neighbor already tabled (§2); pivot emphasis to whichever delta survives. |
| Solo calibration failure (framing/novelty misjudged) | external readers confused; workshop reviews cite framing | Track C exists for this: community feedback Sep+, collaborator outreach Nov, two external readers before Dec red-team; workshops chosen as the calibration-forgiving venue. |
| November overload (ablations + outreach + interviews collide) | w2 slippage on E3/E4 | Nov is priority-ordered with an explicit cut line (§6); E4 pre-capped; E5 needs no runs; paper skeleton pulled into Nov w1 so Dec absorbs slack; below-the-line items default to paper-two. |
| Job offer disrupts research timeline | start date lands mid-Dec/Jan | Negotiate start date; gates survive on reduced hours (laptop-scale compute by design); worst case slip to ICML cycle (~May) — see cross-track failure handling. |

## 8. Deliverables

1. arXiv paper (title/abstract chess-free; robotics-legible).
2. Clean repo: harness, solvers, configs, one-command reproduction per figure.
3. Blog post (5-minute read) + resume line: "object-centric world models with rule-conditioned factorization; measured concept transfer across environment changes against exact ground truth."
4. Parking-lot file → next-paper backlog (fog-of-war belief states, chess concept engine, LLM-composed components).

## 9. References to carry into the paper

arXiv:2102.12375 (variant transfer), arXiv:2107.01078 (board-game concepts; Hex/Misère Hex negative transfer), arXiv:1706.04317 (Schema Networks), arXiv:2602.06550 (DMA*-SH/AIB), arXiv:2110.15949 (GateL0RD), VSG NeurIPS 2022, THICK ICLR 2024, arXiv:2210.12448 (transfer probing/ANOVA), arXiv:2211.15457 (HyperZero), RIMs arXiv:1909.10893, NPS, C-SWMs arXiv:1911.12247, EMPA/theory-based RL, CARL/Kirk et al. zero-shot-generalization protocol, CAV-faithfulness critique (2025-26), Ludii, Fairy-Stockfish. Verify all IDs at writing time.
