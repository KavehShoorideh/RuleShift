"""Gate G1: solver validated against externally sourced known results.

Expected values cite docs/known-results.md (sources fetched 2026-08-30).
Heavy boards are marked slow: run those with `pytest -m slow`.
"""
import pytest

from ruleshift.engine import DRAW, LOSS, WIN, Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver


def initial_value(**kw) -> int:
    engine = Engine(Ruleset(**kw))
    return Solver(engine).value(engine.initial())


def initial_policy(**kw):
    engine = Engine(Ruleset(**kw))
    return Solver(engine).policy(engine.initial())


# ------------------------------------------- k=3, no gravity [W, U19]
K3 = [
    pytest.param(m, n, DRAW if (m, n) == (3, 3) else WIN, id=f"m{m}n{n}k3")
    for m in range(3, 7)
    for n in range(3, 7)
]


@pytest.mark.parametrize("m,n,expected", K3)
def test_k3_grid(m, n, expected):
    assert initial_value(m=m, n=n, k=3) == expected


# ------------------------------------------- k=4, no gravity [U19, W]
K4_PAIRING_DRAWS = [(4, 3), (3, 4), (5, 3), (3, 5), (6, 3), (3, 6)]  # U19 Thm 2
K4_SOLVED_DRAWS = [(4, 4), (5, 4), (4, 5), (6, 4), (4, 6), (5, 5)]  # U19 Table I
K4_WINS = [(6, 5), (5, 6), (6, 6)]  # U19 Table I + fn. 2

K4 = (
    [pytest.param(m, n, DRAW, id=f"m{m}n{n}k4-pairing") for m, n in K4_PAIRING_DRAWS]
    + [
        pytest.param(m, n, DRAW, id=f"m{m}n{n}k4", marks=pytest.mark.slow)
        for m, n in K4_SOLVED_DRAWS
    ]
    + [
        pytest.param(m, n, WIN, id=f"m{m}n{n}k4", marks=pytest.mark.slow)
        for m, n in K4_WINS
    ]
)


@pytest.mark.parametrize("m,n,expected", K4)
def test_k4_grid(m, n, expected):
    assert initial_value(m=m, n=n, k=4) == expected


# ------------------------------------------- gravity, k=4 [T]
GRAV = [
    (4, 4, DRAW),
    (5, 4, DRAW),
    (6, 4, LOSS),  # second-player win
    (4, 5, DRAW),
    (5, 5, DRAW),
    (6, 5, DRAW),
    (4, 6, DRAW),
    (5, 6, DRAW),
    (6, 6, LOSS),  # second-player win
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "m,n,expected", [pytest.param(*g, id=f"m{g[0]}n{g[1]}k4grav") for g in GRAV]
)
def test_gravity_k4_tromp_table(m, n, expected):
    assert initial_value(m=m, n=n, k=4, gravity=True) == expected


# ------------------------------------------- misere and torus 3,3,3
def test_misere_tictactoe_draw_center_only():
    # [WV, MYD]: draw; center is the only non-losing first move
    v, moves = initial_policy(m=3, n=3, k=3, misere=True)
    assert v == DRAW
    assert moves == (4,)


def test_torus_tictactoe_first_player_wins_all_openings():
    # [CN] (weakly sourced; this solve is corroborating evidence)
    v, moves = initial_policy(m=3, n=3, k=3, torus=True)
    assert v == WIN
    assert moves == tuple(range(9))
