"""Amendment A2: the two adversarial knobs.

capture = ENTANGLED (dynamics + win condition change together)
scoring = DIFFUSE (value is an accumulated line-count differential)
"""
import pytest

from ruleshift.engine import DRAW, LOSS, WIN, Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver


def E(**kw):
    return Engine(Ruleset(**kw))


# ------------------------------------------------------------- capture: dynamics
def test_capture_flips_sandwiched_run():
    e = E(m=4, n=3, k=4, capture=True)  # k=4 so the flip does not end the game
    s, status = e.play([0, 1, 2])  # X at (0,0), O at (0,1), X at (0,2) sandwiches
    assert status is None
    x_bits = s[1]  # stones of the player who just moved
    assert x_bits == 0b111  # O's stone at cell 1 flipped to X
    assert s[0] == 0  # O has nothing left


def test_capture_needs_own_stone_to_close_the_sandwich():
    e = E(m=4, n=3, k=4, capture=True)
    s, _ = e.play([0, 1])  # X at 0, O at 1: no flip, nothing is sandwiched
    assert s[1] == 0b10 and s[0] == 0b1
    # an unterminated run does not flip
    e2 = E(m=4, n=3, k=4, capture=True)
    s2, _ = e2.play([0, 1, 8])  # X plays far away; O's stone survives
    assert s2[0] & 0b10


def test_capture_entangles_the_win_condition():
    # Without capture, X owning {0, 2} is nothing; with capture the flip of
    # cell 1 completes X's bottom row on the spot.
    seq = [0, 1, 2]
    assert E(m=3, n=3, k=3).play(seq)[1] is None
    assert E(m=3, n=3, k=3, capture=True).play(seq)[1] == LOSS  # mover (O) has lost


def test_capture_is_not_monotonic_so_fast_path_is_off():
    assert E(m=3, n=3, k=3, capture=True).fast_path is False
    assert E(m=3, n=3, k=3).fast_path is True
    with pytest.raises(ValueError):
        E(m=3, n=3, k=3, capture=True).status_after((0, 1), 0)


def test_capture_respects_forbidden_and_bounds():
    # a forbidden cell breaks the sandwich: no flip through a hole
    e = E(m=4, n=3, k=4, capture=True, forbidden=frozenset({(0, 1)}))
    s, _ = e.play([0, 5, 2])  # O plays elsewhere; X at (0,2) cannot flip across the hole
    assert s[0] & (1 << 5)  # O's stone untouched


# ------------------------------------------------------------- scoring: diffuse
def test_scoring_does_not_end_on_a_line():
    normal = E(m=3, n=3, k=3)
    scoring = E(m=3, n=3, k=3, scoring=True)
    seq = [0, 3, 1, 4, 2]  # X completes the bottom row
    assert normal.play(seq)[1] == LOSS  # normal play: game over
    assert scoring.play(seq)[1] is None  # scoring: play continues


def test_scoring_terminal_is_a_line_count_differential():
    e = E(m=3, n=3, k=3, scoring=True)
    # X: 0,1,2 (row0) and 6,8; O: 3,4,5 (row1) and 7  -> 1 line each -> draw
    s, status = e.play([0, 3, 1, 4, 2, 5, 6, 7, 8])
    assert (s[0] | s[1]) == e.full
    assert status == DRAW
    assert e.count_lines(s[1]) == e.count_lines(s[0]) == 1


def test_scoring_decisive_when_counts_differ():
    e = E(m=3, n=3, k=3, scoring=True)
    # X takes row0 and col0 (0,1,2,3,6); O gets 4,5,7,8 (col2 = 2,5,8 blocked by X's 2)
    s, status = e.play([0, 4, 1, 5, 2, 7, 3, 8, 6])
    x_lines, o_lines = e.count_lines(s[1]), e.count_lines(s[0])
    assert x_lines > o_lines
    assert status == LOSS  # player to move (O) lost the count


def test_scoring_misere_inverts_the_differential():
    plain = E(m=3, n=3, k=3, scoring=True)
    mis = E(m=3, n=3, k=3, scoring=True, misere=True)
    seq = [0, 4, 1, 5, 2, 7, 3, 8, 6]
    assert plain.play(seq)[1] == LOSS
    assert mis.play(seq)[1] == WIN


# ------------------------------------------------- solver correctness on A2 knobs
A2_VARIANTS = [
    dict(m=3, n=3, k=3, capture=True),
    dict(m=3, n=3, k=3, capture=True, misere=True),
    dict(m=3, n=3, k=3, capture=True, torus=True),
    dict(m=4, n=3, k=4, capture=True),
    dict(m=3, n=3, k=3, capture=True, gravity=True),
    dict(m=3, n=3, k=3, scoring=True),
    dict(m=3, n=3, k=3, scoring=True, misere=True),
    dict(m=4, n=3, k=4, scoring=True, gravity=True),
    dict(m=3, n=3, k=3, capture=True, scoring=True),
]


@pytest.mark.parametrize("kw", A2_VARIANTS, ids=lambda kw: Ruleset(**kw).variant_id)
def test_solver_matches_reference_on_a2_knobs(kw):
    from reference import plain_value

    engine = Engine(Ruleset(**kw))
    assert Solver(engine).value(engine.initial()) == plain_value(engine, engine.initial())


def test_a2_knobs_change_optimal_play():
    """The knobs must actually move the game, or they are not adversarial.

    Checked with the A1 solver-grounded measure, not the opening position: on
    3x3 every opening is symmetric, so the opening alone hides real divergence.
    """
    from ruleshift.distance import behavioral_distance

    base = Ruleset(m=3, n=3, k=3)
    for kw in (dict(capture=True), dict(scoring=True)):
        d = behavioral_distance(base, Ruleset(m=3, n=3, k=3, **kw), n_positions=60, seed=0)
        assert d.policy_disagreement > 0.1, f"{kw} barely moves optimal play ({d})"


def test_one_knob_spans_a_wide_range_of_real_change():
    """The A1 claim: knob count is not a proxy for behavioural change.

    Every variant here is exactly ONE knob from the base, yet the divergence of
    optimal play spans a wide range -- which is why the headline axis is
    solver-grounded rather than knob-counted.
    """
    from ruleshift.distance import behavioral_distance

    base = Ruleset(m=3, n=3, k=3)
    ds = {}
    for name, kw in (("misere", dict(misere=True)), ("scoring", dict(scoring=True)),
                     ("capture", dict(capture=True)), ("torus", dict(torus=True))):
        d = behavioral_distance(base, Ruleset(m=3, n=3, k=3, **kw), n_positions=120, seed=0)
        assert d.knob_distance == 1
        ds[name] = d.distance
    assert max(ds.values()) / min(ds.values()) > 1.5, ds
    assert ds["misere"] == max(ds.values())  # goal inversion is the farthest single knob
