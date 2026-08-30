# Known results used as solver ground truth

Compiled 2026-08-30 from sources fetched and read directly (research pass for
Gate G1). Encoded as tests in `tests/test_known_results.py`. Boards are
**width x height** (our m x n). Re-verify at writing time (plan par.9).

## Sources

- **[W]** Wikipedia "m,n,k-game": "(3,3,3) is a draw ... (m,n,3) is a win if
  m >= 3 and n >= 4 or m >= 4 and n >= 3"; "(5,5,4) is a draw"; "(6,5,4) is a
  win"; "(m,n,4) is a win for m >= 6 and n >= 5 or m >= 5 and n >= 6".
  https://en.wikipedia.org/wiki/M,n,k-game (k=3 sentence has no inline citation;
  corroborated by [U19].)
- **[U19]** Uiterwijk, "Solving Strong and Weak 4-in-a-Row", IEEE CoG 2019.
  Convention: m = rows, n = columns (transposition-invariant without gravity).
  Theorem 2 (pairing): any mnk-game with m < k and/or n < k is a draw.
  Table I: 4x4, 4x5, 4x6, 5x5 draw; 5x6 win ("the 5x6 board is the smallest
  board on which the first player wins"). Footnote 2: all boards
  5<=m<=10, 6<=n<=10 are wins. https://ieee-cog.org/2019/papers/paper_115.pdf
- **[T]** John Tromp, "John's Connect Four Playground" — solved size table
  (width 4-11 x height 4-11 partial), transcribed cell-by-cell from raw HTML.
  Legend: "+ first player win, = tie, - second player win".
  https://tromp.github.io/c4/c4.html
- **[C4W]** Wikipedia "Connect Four": 7x6 is a first-player win (Allen, Oct 1
  1988; Allis, Oct 16 1988, independently).
- **[WV]** Wikipedia "Tic-tac-toe variants": misere 3x3 "A 3x3 game is a draw";
  first player draws by center-then-mirror (Golomb & Hales).
- **[MYD]** MindYourDecisions (2016): misere 3x3 strongly solved, draw; center
  is the only non-losing first move.
- **[CN]** Concrete Nonsense blog (2008), "Topological Tic Tac Toe 1: The
  Torus": torus 3x3 is a first-player win in 4 moves; all openings equivalent;
  12 winning lines. **Weakly sourced** (math blog + hedged forum corroboration;
  no peer-reviewed source found). Our exact solver corroborates.
- **[SL24]** arXiv:2410.05551 (Steele & Larremore): misere Connect Four 7x6 is
  a second-player win.

## Values encoded in tests

- **k=3, no gravity, m,n in 3..6:** 3x3 draw; every other board first-player win.
- **k=4, no gravity:** pairing draws (m<4 or n<4): 4x3, 3x4, 5x3, 3x5, 6x3, 3x6.
  Computer-solved draws: 4x4, 5x4, 4x5, 6x4, 4x6, 5x5. Wins: 6x5, 5x6, 6x6.
- **k=4 gravity (w x h)** [T]: 4x4 =, 5x4 =, 6x4 **-**, 4x5 =, 5x5 =, 6x5 =,
  4x6 =, 5x6 =, 6x6 **-** (second-player wins on 6x4 and 6x6).
- **Misere 3,3,3:** draw; unique non-losing opening = center.
- **Torus 3,3,3:** first-player win; all 9 openings optimal.

## Frontier targets (not in tests; checked via scripts/solve_grid.py if feasible)

- Connect Four 7x6 = first-player win [T, C4W].
- Misere Connect Four 7x6 = second-player win [SL24].

## Caveats

- Tromp's non-standard sizes are single-source computer solutions (his page
  invites independent verification — our solver is one).
- Torus value is the one weakly sourced entry; treat our solve as the primary
  evidence and the blog as corroboration.
- Gravity with k=3: no sourced values found anywhere — our solver's outputs
  for those variants are self-certified only (reference-solver cross-checks).
- An automated summary of Tromp's page misread one cell during research;
  values here come from a direct cell-by-cell read of the raw HTML table.
