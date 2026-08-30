"""Gate G1: solver validated against externally sourced known results.

Expected values cite docs/known-results.md (sources fetched 2026-08-30).
Markers reflect measured solve times (docs/benchmarks.md):
  (default)  seconds     -- runs in the fast suite
  slow       ~5s-2min    -- run with: pytest -m slow
  frontier   beyond pure-Python v1 reach -- run with: pytest -m frontier
"""
import pytest

from ruleshift.engine import DRAW, LOSS, WIN, Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver

slow = pytest.mark.slow
frontier = pytest.mark.frontier


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
K4 = [
    # pairing draws, U19 Thm 2 (m < 4 or n < 4)
    *[pytest.param(m, n, DRAW, id=f"m{m}n{n}k4-pairing")
      for m, n in [(4, 3), (3, 4), (5, 3), (3, 5), (6, 3), (3, 6)]],
    # computer-solved draws, U19 Table I
    pytest.param(4, 4, DRAW, id="m4n4k4"),
    pytest.param(5, 4, DRAW, id="m5n4k4", marks=slow),
    pytest.param(4, 5, DRAW, id="m4n5k4", marks=slow),
    pytest.param(6, 4, DRAW, id="m6n4k4", marks=slow),
    pytest.param(4, 6, DRAW, id="m4n6k4", marks=slow),
    pytest.param(5, 5, DRAW, id="m5n5k4", marks=frontier),
    # wins, U19 Table I + fn. 2
    pytest.param(6, 5, WIN, id="m6n5k4", marks=frontier),
    pytest.param(5, 6, WIN, id="m5n6k4", marks=frontier),
    pytest.param(6, 6, WIN, id="m6n6k4", marks=frontier),
]


@pytest.mark.parametrize("m,n,expected", K4)
def test_k4_grid(m, n, expected):
    assert initial_value(m=m, n=n, k=4) == expected


# ------------------------------------------- gravity, k=4 [T]
GRAV = [
    pytest.param(4, 4, DRAW, id="m4n4k4grav"),
    pytest.param(5, 4, DRAW, id="m5n4k4grav"),
    pytest.param(6, 4, LOSS, id="m6n4k4grav", marks=slow),  # second-player win
    pytest.param(4, 5, DRAW, id="m4n5k4grav"),
    pytest.param(5, 5, DRAW, id="m5n5k4grav", marks=slow),
    pytest.param(6, 5, DRAW, id="m6n5k4grav", marks=slow),
    pytest.param(4, 6, DRAW, id="m4n6k4grav"),
    pytest.param(5, 6, DRAW, id="m5n6k4grav", marks=slow),
    pytest.param(6, 6, LOSS, id="m6n6k4grav", marks=frontier),  # second-player win
]


@pytest.mark.parametrize("m,n,expected", GRAV)
def test_gravity_k4_tromp_table(m, n, expected):
    assert initial_value(m=m, n=n, k=4, gravity=True) == expected


@frontier
def test_connect_four_classic():
    # 7x6 first-player win (Allen 1988; Allis 1988; Tromp) [T, C4W]
    assert initial_value(m=7, n=6, k=4, gravity=True) == WIN


@frontier
def test_misere_connect_four():
    # 7x6 misere: second-player win [SL24]
    assert initial_value(m=7, n=6, k=4, gravity=True, misere=True) == LOSS


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
