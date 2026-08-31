"""Ground-truth verification for the rule semantics themselves.

tests/reference.py certifies the SEARCH (it shares Engine.step). These tests
certify the RULES, three ways that do not share the engine's code:

1. DIFFERENTIAL: an independent naive reimplementation (tests/naive_rules.py,
   2-D grids, no bitboards) must agree on legal moves, transitions, terminal
   status and full game values.
2. METAMORPHIC: exact values must be invariant under the board symmetries the
   rules admit (an oracle-free check of rules AND solver together).
3. REDUCTION: knob settings with a known degenerate equivalent must reproduce
   the simpler game exactly.
"""
import random

import pytest

from naive_rules import NaiveGame, naive_move, to_naive
from ruleshift.engine import Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver

ALL_KNOBS = [
    dict(m=3, n=3, k=3),
    dict(m=3, n=3, k=3, misere=True),
    dict(m=3, n=3, k=3, torus=True),
    dict(m=4, n=3, k=3, gravity=True),
    dict(m=3, n=3, k=3, capture=True),
    dict(m=3, n=3, k=3, capture=True, misere=True),
    dict(m=3, n=3, k=3, capture=True, torus=True),
    dict(m=4, n=3, k=4, capture=True, gravity=True),
    dict(m=3, n=3, k=3, scoring=True),
    dict(m=3, n=3, k=3, scoring=True, misere=True),
    dict(m=4, n=3, k=3, scoring=True, gravity=True),
    dict(m=3, n=3, k=3, capture=True, scoring=True),
    dict(m=4, n=3, k=3, forbidden=frozenset({(1, 1)})),
    dict(m=3, n=3, k=3, capture=True, forbidden=frozenset({(1, 1)})),
]
IDS = [Ruleset(**kw).variant_id for kw in ALL_KNOBS]


# ------------------------------------------------------------ 1. DIFFERENTIAL
@pytest.mark.parametrize("kw", ALL_KNOBS, ids=IDS)
def test_transitions_match_independent_implementation(kw):
    """Every ply of many random playouts: legal moves, board state, terminality."""
    rules = Ruleset(**kw)
    engine, naive = Engine(rules), NaiveGame(rules)
    rng = random.Random(0)
    plies_checked = 0
    for _ in range(40):
        state, nstate = engine.initial(), naive.initial()
        while True:
            eng_moves = sorted(naive_move(engine, mv) for mv in engine.legal_moves(state))
            nai_moves = naive.legal_moves(nstate)
            assert eng_moves == nai_moves, f"legal moves differ at {state}"
            assert to_naive(engine, state) == nstate, f"board state differs at {state}"
            if not eng_moves:
                break
            mv = rng.choice(engine.legal_moves(state))
            state, status = engine.step(state, mv)
            nstate = naive.apply(nstate, naive_move(engine, mv))
            assert status == naive.status(nstate), f"terminal status differs after {mv}"
            plies_checked += 1
            if status is not None:
                break
    assert plies_checked > 100


@pytest.mark.parametrize("kw", ALL_KNOBS, ids=IDS)
def test_exact_values_match_independent_implementation(kw):
    """Full game-tree value from the initial position, computed twice over
    two independent rule implementations and two different search programs."""
    rules = Ruleset(**kw)
    engine = Engine(rules)
    naive = NaiveGame(rules)
    assert Solver(engine).value(engine.initial()) == naive.value(naive.initial())


@pytest.mark.parametrize("kw", ALL_KNOBS[:8], ids=IDS[:8])
def test_midgame_values_match_independent_implementation(kw):
    """Values agree away from the opening, where knob interactions bite."""
    rules = Ruleset(**kw)
    engine, naive = Engine(rules), NaiveGame(rules)
    solver = Solver(engine)
    rng = random.Random(7)
    checked = 0
    for _ in range(30):
        state = engine.initial()
        for _ply in range(rng.randint(1, 4)):
            moves = engine.legal_moves(state)
            if not moves:
                break
            nxt, status = engine.step(state, rng.choice(moves))
            if status is not None:
                break
            state = nxt
        else:
            assert solver.value(state) == naive.value(to_naive(engine, state))
            checked += 1
    assert checked >= 10


# -------------------------------------------------------------- 2. METAMORPHIC
def cell_maps(rules):
    """Board symmetries the ruleset admits, as (r, c) -> (r, c) maps."""
    n, m = rules.n, rules.m
    maps = [lambda r, c: (r, c)]
    if rules.forbidden:
        return maps  # a forbidden mask generally breaks symmetry
    maps.append(lambda r, c: (r, m - 1 - c) if m == m else None)  # horizontal mirror
    if not rules.gravity:  # gravity fixes the vertical direction
        maps.append(lambda r, c: (n - 1 - r, c))
        maps.append(lambda r, c: (n - 1 - r, m - 1 - c))
        if n == m:
            maps.append(lambda r, c: (c, r))  # transpose
            maps.append(lambda r, c: (m - 1 - c, n - 1 - r))
    if rules.torus:  # wrapping adds translations
        maps.append(lambda r, c: ((r + 1) % n, c))
        maps.append(lambda r, c: (r, (c + 1) % m))
    return maps


def permute(engine, state, fn):
    m = engine.rules.m
    out = []
    for bits in state:
        acc = 0
        i = 0
        while bits:
            if bits & 1:
                r, c = divmod(i, m)
                rr, cc = fn(r, c)
                acc |= 1 << (rr * m + cc)
            bits >>= 1
            i += 1
        out.append(acc)
    return (out[0], out[1])


SYM_VARIANTS = [
    dict(m=3, n=3, k=3),
    dict(m=3, n=3, k=3, misere=True),
    dict(m=3, n=3, k=3, torus=True),
    dict(m=3, n=3, k=3, capture=True),
    dict(m=3, n=3, k=3, scoring=True),
    dict(m=4, n=3, k=3, gravity=True),
    dict(m=4, n=4, k=4, capture=True),
]


@pytest.mark.parametrize(
    "kw", SYM_VARIANTS, ids=[Ruleset(**kw).variant_id for kw in SYM_VARIANTS]
)
def test_values_are_invariant_under_board_symmetry(kw):
    """Oracle-free check: a position and its mirror/rotation must have equal
    exact value. Catches direction, wrap and indexing errors in the rules and
    asymmetric bugs in the search."""
    rules = Ruleset(**kw)
    engine = Engine(rules)
    solver = Solver(engine)
    rng = random.Random(3)
    maps = cell_maps(rules)
    assert len(maps) > 1
    checked = 0
    for _ in range(25):
        state = engine.initial()
        for _ply in range(rng.randint(1, 4)):
            moves = engine.legal_moves(state)
            if not moves:
                break
            nxt, status = engine.step(state, rng.choice(moves))
            if status is not None:
                break
            state = nxt
        else:
            base = solver.value(state)
            for fn in maps[1:]:
                assert solver.value(permute(engine, state, fn)) == base
            checked += 1
    assert checked >= 8


# --------------------------------------------------------------- 3. REDUCTION
def test_gravity_is_a_no_op_on_a_single_row():
    """With one row, every cell is its own column's floor: gravity changes nothing."""
    a = Engine(Ruleset(m=4, n=1, k=3))
    b = Engine(Ruleset(m=4, n=1, k=3, gravity=True))
    assert Solver(a).value(a.initial()) == Solver(b).value(b.initial())
    assert sorted(a.legal_moves(a.initial())) == sorted(b.legal_moves(b.initial()))


def test_capture_is_a_no_op_when_no_sandwich_can_exist():
    """On a 2x1 board there is no room to enclose a stone, so capture reduces
    to the base game."""
    a = Engine(Ruleset(m=2, n=1, k=2))
    b = Engine(Ruleset(m=2, n=1, k=2, capture=True))
    assert Solver(a).value(a.initial()) == Solver(b).value(b.initial())


def test_scoring_without_any_line_is_always_a_draw():
    """k larger than the board admits no lines: every scoring game ends 0-0."""
    e = Engine(Ruleset(m=3, n=3, k=4, scoring=True))
    assert e.lines == ()
    assert Solver(e).value(e.initial()) == 0


def test_misere_scoring_inverts_a_decisive_scoring_game():
    """Sanity on the polarity wiring: if plain scoring is decisive for one side
    on a fixed line of play, misere scoring flips that terminal payoff."""
    plain = Engine(Ruleset(m=3, n=3, k=3, scoring=True))
    mis = Engine(Ruleset(m=3, n=3, k=3, scoring=True, misere=True))
    seq = [0, 4, 1, 5, 2, 7, 3, 8, 6]
    assert plain.play(seq)[1] == -mis.play(seq)[1] != 0


def test_player_identity_survives_capture():
    """Regression: stone counts do not identify the opener once ownership can
    change hands; occupancy parity does. Caught by differential testing."""
    engine = Engine(Ruleset(m=3, n=3, k=3, capture=True))
    rng = random.Random(11)
    for _ in range(60):
        state, plies = engine.initial(), 0
        while True:
            assert engine.first_player_to_move(state) == (plies % 2 == 0)
            moves = engine.legal_moves(state)
            if not moves:
                break
            state, status = engine.step(state, rng.choice(moves))
            plies += 1
            if status is not None:
                break
