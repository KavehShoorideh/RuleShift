# RuleShift

**What happens to everything an AI has learned about a game when you change the rules?**

Take a neural network that has gotten good at tic-tac-toe. Now flip one rule —
say, three-in-a-row now *loses* instead of wins. Most of what the network
knows ("two of mine in a row is great news") is suddenly wrong. But some of it
("the center square touches the most lines") is still true. Which knowledge
survives? Which dies? And are some network designs better at keeping the good
parts and relearning only what actually changed?

That's the question this project measures. The full research plan lives in
[docs/plan.md](docs/plan.md); the short version:

1. **Build a family of small games** — tic-tac-toe and its many cousins.
   Bigger boards, longer lines to win, Connect-Four-style falling pieces,
   boards that wrap around at the edges, blocked-off squares, and goal-flipped
   "misère" versions where completing a line loses.
2. **Solve those games perfectly** with a classical algorithm — an answer key
   that knows the objectively best move in every position.
3. **Train neural networks to play, then change the rules on them** — and
   measure against the answer key, not vibes, how fast they adapt and which
   internal pieces of their knowledge survived the change.

Why bother? Because "the world changed and my model's knowledge is now partly
stale" is a very general problem — a robot's gripper wears down, a road rule
differs across a border — and tiny board games are one of the few places you
can study it with *exact* ground truth instead of approximations.

## What's here so far

The measurement equipment, built and tested:

- **The game box** (`rules.py`, `engine.py`) — build any variant by turning
  knobs, and play it.
- **The answer key** (`solver.py`) — perfect play for every game small enough
  to solve on a laptop, saved to disk so it's only computed once.
- **The report card** (`metrics.py`) — for any move, exactly what it cost
  compared to perfect play: nothing, a draw, or the whole game.
- **Flashcards** (`dataset.py`) — training sets of positions labeled with the
  true value and every perfect move.
- **The lab notebook** (`tracker.py`) — every experiment writes its settings
  and results to a folder, so everything is reproducible.

Trust, but verify: the answer key is checked two ways. Against a
slow-but-obviously-correct twin solver on thousands of positions, and against
37 published results from the games literature — every one matched, including
the delightfully weird fact that on a 6-wide, 4-tall Connect Four board, the
player who goes *second* wins with perfect play.

One honest limitation: a handful of the biggest boards (like 5×5 with
4-in-a-row) are still too slow to solve perfectly in pure Python.
[docs/benchmarks.md](docs/benchmarks.md) records exactly which ones, and
[docs/parking-lot.md](docs/parking-lot.md) queues the speed-up ideas for when
they're needed.

## Try it

```bash
uv venv .venv && uv pip install -e ".[dev]" -p .venv/bin/python
.venv/bin/pytest                                   # fast test suite (~5 seconds)
.venv/bin/python scripts/solve_grid.py             # solve a grid of game variants
.venv/bin/python scripts/gen_dataset.py --m 4 --n 4 --k 3 --n-positions 1000
```

## The details

- [docs/technical.md](docs/technical.md) — conventions, architecture, and how
  the validation works (the scientific version of this page)
- [docs/plan.md](docs/plan.md) — research plan, hypotheses, timeline
- [docs/known-results.md](docs/known-results.md) — the published game values we
  validate against, with sources
- [docs/benchmarks.md](docs/benchmarks.md) — what's solvable, how fast
