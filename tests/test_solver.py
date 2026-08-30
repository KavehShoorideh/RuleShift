import random

import pytest

from reference import plain_value
from ruleshift.engine import DRAW, LOSS, WIN, Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver

# Certification matrix: Solver (alpha-beta + TT) must match the pruning-free
# reference solver exactly, across knob combinations and interactions.
CERT_VARIANTS = [
    dict(m=3, n=3, k=3),
    dict(m=3, n=3, k=3, misere=True),
    dict(m=3, n=3, k=3, torus=True),
    dict(m=3, n=3, k=3, torus=True, misere=True),
    dict(m=3, n=3, k=3, gravity=True),
    dict(m=3, n=3, k=3, gravity=True, misere=True),
    dict(m=4, n=3, k=3),
    dict(m=4, n=3, k=3, misere=True),
    dict(m=4, n=3, k=3, forbidden=frozenset({(2, 3)})),
    dict(m=4, n=4, k=4, gravity=True),
    dict(m=4, n=4, k=4, gravity=True, misere=True),
    dict(m=4, n=4, k=3, gravity=True, torus=True),
]


@pytest.mark.parametrize("kw", CERT_VARIANTS, ids=lambda kw: Ruleset(**kw).variant_id)
def test_solver_matches_reference_from_initial(kw):
    engine = Engine(Ruleset(**kw))
    assert Solver(engine).value(engine.initial()) == plain_value(engine, engine.initial())


def test_solver_matches_reference_midgame_443():
    engine = Engine(Ruleset(m=4, n=4, k=3))
    rng = random.Random(0)
    solver = Solver(engine)
    memo: dict = {}
    checked = 0
    for _ in range(200):
        s = engine.initial()
        for _ply in range(6):
            mv = rng.choice(engine.legal_moves(s))
            s2 = engine.apply(s, mv)
            if engine.status_after(s2, mv) is not None:
                break
            s = s2
        else:
            assert solver.value(s) == plain_value(engine, s, memo)
            checked += 1
            if checked >= 25:
                break
    assert checked >= 25


# ------------------------------------------------- textbook known results
def test_tictactoe_draw_and_all_openings_draw():
    engine = Engine(Ruleset(m=3, n=3, k=3))
    solver = Solver(engine)
    assert solver.value(engine.initial()) == DRAW
    v, moves = solver.policy(engine.initial())
    assert v == DRAW
    assert moves == tuple(range(9))


def test_mnk_433_first_player_win_and_transpose_invariance():
    # (m,n,3) is a first-player win once the board reaches 3x4 / 4x3
    for m, n in ((4, 3), (3, 4)):
        engine = Engine(Ruleset(m=m, n=n, k=3))
        assert Solver(engine).value(engine.initial()) == WIN


def test_monotonicity_spot_check():
    # extra space never hurts the first player: 4,3,3 win => 5,5,3 win
    engine = Engine(Ruleset(m=5, n=5, k=3))
    assert Solver(engine).value(engine.initial()) == WIN


# ------------------------------------------------------- policy correctness
def test_forced_block_unique_optimal_move():
    engine = Engine(Ruleset(m=3, n=3, k=3))
    s, status = engine.play([0, 4, 1])  # X threatens 0-1-2; O to move
    assert status is None
    v, moves = Solver(engine).policy(s)
    assert (v, moves) == (DRAW, (2,))


def test_win_in_one_policy():
    engine = Engine(Ruleset(m=3, n=3, k=3))
    s, _ = engine.play([0, 6, 1, 7])  # X{0,1} O{6,7}, X to move
    v, moves = Solver(engine).policy(s)
    assert v == WIN
    assert moves == (2, 8)  # immediate win at 2; 8 blocks O and forces a win too
    # blundering instead: O wins at 8 -> LOSS for X
    assert Solver(engine).child_value(s, 4) == LOSS


def test_policy_terminal_raises():
    engine = Engine(Ruleset(m=3, n=3, k=3))
    s, _ = engine.play([0, 3, 1, 4, 2])
    with pytest.raises(ValueError):
        Solver(engine).policy(s)


# ----------------------------------------------------------- persistence
def test_save_load_roundtrip(tmp_path):
    engine = Engine(Ruleset(m=3, n=3, k=3))
    solver = Solver(engine)
    solver.value(engine.initial())
    p = tmp_path / "m3n3k3.pkl"
    solver.save(p)

    fresh = Solver(engine)
    fresh.load(p)
    assert fresh.value(engine.initial()) == DRAW
    assert fresh.nodes <= 1  # answered from the loaded TT

    other = Solver(Engine(Ruleset(m=3, n=3, k=3, misere=True)))
    with pytest.raises(ValueError):
        other.load(p)
