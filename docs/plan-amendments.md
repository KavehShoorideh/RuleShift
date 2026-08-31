# Plan Amendments — post-review corrections

Amendments to `plan.md` decided after that document was written. Each states the change, the reason, where it lands, and its cost. **No gate dates change (G1–G5 stand);** A1–A4 are funded by trimming the variant grid, not by adding time.

Origin of these corrections: two objections raised against the plan — (1) the rule formalism is ours, so results may only apply to models trained in it; (2) the knob design encodes assumptions about what stays constant, so others must share those assumptions for the work to be useful. The second is the construct-validity objection a reviewer will actually write down: if the knobs align with the architecture's factorization (goal knob → value head, dynamics knob → transition module), the factorized model wins by construction and the result is about knob design, not factorization.

---

## A1 — Rule distance grounded in the solver, not in knob count *(primary metric change)*

- **Change:** define distance between variants as divergence of *optimal play* — disagreement rate between optimal move sets, and/or exact value divergence, over a shared sample of positions. Knob-edit-distance demoted to a descriptive label.
- **Why:** makes the headline x-axis ground truth rather than our parameterization; directly answers the circularity objection. This is the plot used in the blog post and interviews, so it must not rest on our own design choices.
- **Lands in:** §3 (metrics), §5 E2, headline figure. Promotes what was previously "behavioral distance as secondary analysis" to primary.
- **Cost:** one metric implementation in the Sep harness; solutions are already cached.

## A2 — Two adversarial knobs, capped at two

- **Change:** add exactly two knobs designed to cut across the architecture's seams: (i) one entangled change altering dynamics *and* win condition together; (ii) one whose effect on optimal play is diffuse rather than localized.
- **Why:** tests factorization where it should be weakest. If it still helps, the claim is much stronger; if it fails, that is the honest boundary of the result and reporting it is what makes the rest credible.
- **Lands in:** §3 rule space; implemented in Sep w1–2 harness so no October rework.
- **Cost:** two variants' worth of runs. **Hard cap at two** — more multiplies the October grid.

## A3 — Conditioning ablation, scoped to M0 and M2 only

- **Change:** run each of the two rule-conditioning modes on the monolithic baseline (M0) and the factorized model (M2) only, not across the whole grid:
  1. explicit knob vector (hands the model our factorization);
  2. general rule description (movement/terminal predicates, no knob structure).
- **Why:** the gap between modes *quantifies* how much the hand-designed parameterization is doing — converting the weakness into a measurement. Highest career-signal item in the plan: "I noticed my benchmark could hand the model the answer key, so I measured what that was worth" is exactly the research-taste evidence an RE hiring panel screens for, and it is one paragraph in an interview.
- **Lands in:** §5 as E2b.
- **Cost:** 2 extra model-variant training sweeps; fits October because no new model classes are introduced.

## A4 — Trim the variant grid to fund A1–A3

- **Change:** fewer board sizes (drop the largest (m,n) tier); keep knob coverage, cut redundant size combinations.
- **Why:** variety of variants is not what makes this paper land — the distance axis (A1) and the ablation (A3) are. Keeps October within budget without touching gates.
- **Lands in:** §3 rule space.

## A5 — Formalism-agnostic game interface; external formalism stays optional

- **Change:** the harness is written against a minimal game interface (legal moves, terminal test, exact value, rule descriptor); the m,n,k family is one plug-in implementation, not the architecture. One out-of-formalism family (Fairy-Stockfish chess variants with Betza-style piece definitions, or a Ludii game-description-language family) is an **optional** external-validity check.
- **Why:** the reusable-benchmark value depends on others not having to adopt our rule formalism; if the ordering of adaptation curves survives in a formalism we did not design, objection (1) is answered empirically rather than rhetorically. Optional because it buys reviewer comfort, not a new result — and December is the writing month.
- **Lands in:** §6 Sep w1–2 (interface); external family → December chess chapter or paper two.
- **Cost:** near-zero in September if decided up front; expensive if retrofitted later.

## A6 — Explicit scope statement in limitations

- **Change:** state plainly: the claim holds for rule changes expressible in the tested space; no claim is made for arbitrary rule modifications.
- **Why:** pre-empting the reviewer's objection tends to disarm it.
- **Cost:** free.

## A7 — Demo deliverable = concept-survival explorer

- **Change:** the demo for this project is a static concept-survival explorer built from E4 output — pick a rule change, see which concepts survive and which die, with exact-solver ground truth behind it. **Not** gameplay.
- **Why:** near-free once figures exist, screenshot-able, and it makes the abstract concrete for anyone who will not read the paper. The narrating/trash-talking chess engine remains the eventual crowd-pleaser but belongs to the chess repo and a later cycle.
- **Lands in:** §8 deliverables; built in January alongside the blog post.

## A8 — Perturbation-search outcomes (decided, do not revisit)

| Option considered | Decision |
|---|---|
| Reliability/interpretability headline vs. world-model-transfer headline | **Both live; choose at G3** (already in plan §1). Default: reliability framing. |
| Adversarial-policy / AI-exploits-AI direction | **Paper two (E6)**, already in plan §1 with pivot rights if H1 is null. |
| M3 sparse gating | **Cut by default**; build only if G3 lands ≥1 week early with H1 positive. |
| Full pivot to LLM interpretability/safety | **Rejected** — discards the games harness, solver ground truth, infra and robotics arc; most crowded lane in ML. |
| Agentic-AI security (LLM agents exploiting systems) | **Rejected** — furthest from existing assets, no reusable infrastructure, hard to publish or wield solo. |

## A9 — Repository decision

- New repo (**not** a fork or conversion of the chess repo). Suggested name `ruleshift` — verify availability on GitHub/PyPI before committing; fallbacks `rule-shift`, `ruleshift-bench`.
- Chess repo stays as-is: it is paper two and the eventual demo. Carry over utilities by cherry-pick (training loop, plotting, config handling), never by inheriting the tree.
- The sparse-concept-rollforward spec belongs to the **chess** repo; this plan and these amendments belong to `ruleshift`.
- Root `CLAUDE.md`: compact — one-line claim, scope-freeze rule (new architecture ideas → `docs/parking-lot.md`; plan changes only on failed experiments, never on new ideas), and an import pointing at `docs/plan.md`.

---

**Net effect on the timeline:** none. A1, A2 and A5 are September harness decisions; A3 is an October sweep; A4 pays for all of them; A6–A9 are free.

---

## Implementation status (updated as amendments land)

| Amendment | Status | Where |
|---|---|---|
| A1 solver-grounded distance | implemented | `ruleshift.distance`, headline figure axis |
| A2 two adversarial knobs | implemented | `capture` (entangled), `scoring` (diffuse) in `ruleshift.rules` |
| A3 conditioning ablation (E2b) | descriptor implemented; sweep is October | `ruleshift.descriptor` |
| A4 grid trim | implemented | `standard_grid` default 3..5; `docs/scope-freeze.md` |
| A5 formalism-agnostic interface | implemented | `ruleshift.interface` |
| A6 scope statement | written | `docs/limitations.md` |
| A7 concept-survival explorer | January deliverable | `docs/plan.md` §8 |
| A8 decided items | recorded | this file |
| A9 repo decision | done | repo `ruleshift`; name free on PyPI; `CLAUDE.md` written |
