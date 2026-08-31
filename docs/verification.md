# How the ground truth is verified

The whole project rests on the solver being right: it produces the training
labels *and* the regret metric, so a rules bug would corrupt every downstream
result silently. This document states exactly what is established, by what
evidence, and what is not.

## The layers, and what covers each

| Layer | Risk | Evidence |
|---|---|---|
| Search (negamax, alpha-beta, TT flags, pruning) | subtle unsoundness in my own optimizations | `tests/reference.py`: a pruning-free memoized negamax with no TT flags, no move ordering, no shortcuts, compared on initial and midgame positions across knob combinations |
| Rules, v1 knobs | wrong semantics | 37 externally sourced published values (`known-results.md`) — an oracle nobody in this project wrote |
| Rules, A2 knobs (`capture`, `scoring`) | wrong semantics, **no published oracle exists** | differential + metamorphic + reduction testing, below (`tests/test_ground_truth.py`) |

**The gap this closes:** `tests/reference.py` shares `Engine.step`, so it
certifies the *search* and says nothing about the *rules*. For `capture` and
`scoring` there is no literature to check against, so correlated error — the
same wrong mental model writing both the code and its unit tests — was the
live risk.

## 1. Differential testing against an independent implementation

`tests/naive_rules.py` re-implements the rule semantics from the spec in a
deliberately different style: a 2-D grid of ints with explicit (r, c) loops,
no bitboards, no precomputation, a different state representation (grid +
explicit turn, rather than perspective-relative bitboards), and a plain
memoized negamax of its own. It is slow and obvious; its only job is to
disagree if either implementation is wrong.

Compared across 14 knob combinations: every legal-move set, every board state,
every terminal status, at every ply of random playouts; full game-tree values
from the initial position; and midgame values from random prefixes.

**This found a real bug on first run.** `Engine.first_player_to_move` derived
player identity from stone counts (`popcount(cur) == popcount(opp)`), which is
false under `capture`, because flips move stones between owners without
changing move parity. Fixed to use occupancy parity (every move fills exactly
one cell under every knob). It had also propagated into the A1 distance
validity filter. Game *values* were unaffected — the value comparisons passed —
but board rendering and position-comparability filtering were wrong, and the
frozen convention in `scope-freeze.md` asserted something untrue.

## 2. Metamorphic testing (oracle-free)

Exact values must be invariant under the board symmetries a ruleset admits:
mirrors and rotations without gravity, horizontal mirror only with gravity
(gravity fixes the vertical direction), plus translations under torus. A
position and its image must solve to the same value. This needs no external
answer key and catches direction, wrap, and indexing errors in the rules as
well as asymmetric bugs in the search.

## 3. Reduction testing

Knob settings with a known degenerate equivalent must reproduce the simpler
game exactly: gravity on a single-row board is a no-op; `capture` on a 2x1
board (no room to enclose a stone) reduces to the base game; `scoring` with
k larger than the board admits no lines and is always a draw; misère scoring
inverts a decisive scoring payoff.

## What is still NOT established

- **The prose-to-code gap.** These techniques show two implementations agree
  and that values respect the rules' symmetries. They cannot show that
  `capture` is what a reader pictures when reading "Othello-style flips" — only
  that it is a well-defined, deterministic, two-player game with the stated
  invariants. Mitigation: the paper defines the knobs operationally, and the
  repo is the definition of record.
- **Large-board search optimizations.** The dead-position and threat-forced
  rules are my own arguments, certified against the reference solver only on
  boards small enough to brute-force. They are disabled for the A2 knobs.
- Solver timings in `benchmarks.md` were measured under heavy machine load.

## Why not a proof assistant

Considered and rejected as disproportionate. Formalizing the harness in Lean or
Dafny would take weeks, and the residual risk it addresses is not the risk we
have: negamax is textbook and the search is already differentially certified,
while the real hazard is *semantic* — whether the implemented rule is the rule
we meant. A proof assistant would verify implementation-against-specification,
and a specification written by the same person, from the same mental model,
fails the same way the code does. Differential testing against an
independently written implementation, plus symmetry invariants the rules must
satisfy, attacks the correlated-error problem directly and at a fraction of the
cost — and demonstrably caught a live bug within an hour.

Revisit only if a result ever hinges on a single hard-to-test invariant.
