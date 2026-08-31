"""Amendment A1: solver-grounded distance is the primary axis."""
import pytest

from ruleshift.distance import behavioral_distance, distance_to_set, remap, transfer
from ruleshift.engine import Engine
from ruleshift.rules import Ruleset


def test_remap_and_transfer_across_board_widths():
    # cell (1, 1) is index 4 on a 3-wide board, index 5 on a 4-wide board
    assert remap(1 << 4, 3, 4) == 1 << 5
    small = Engine(Ruleset(m=3, n=3, k=3))
    big = Engine(Ruleset(m=4, n=4, k=3))
    s, _ = small.play([4, 0])  # X centre (cell 4), O corner (cell 0); X to move
    moved = transfer(s, 3, big)
    assert moved is not None
    assert moved[0] == 1 << 5  # X's centre (1,1): index 4 on a 3-wide board, 5 on a 4-wide
    assert moved[1] == 1 << 0  # O's corner (0,0) is index 0 either way


def test_transfer_rejects_out_of_bounds_and_terminal():
    big = Engine(Ruleset(m=4, n=4, k=3))
    small = Engine(Ruleset(m=3, n=3, k=3))
    s, _ = big.play([3, 0])  # a stone in column 3 has no counterpart on a 3-wide board
    assert transfer(s, 4, small) is None
    won, _ = small.play([0, 3, 1, 4, 2])  # terminal: no optimal-move set to compare
    assert transfer(won, 3, small) is None


def test_identical_variants_have_zero_distance():
    r = Ruleset(m=3, n=3, k=3)
    d = behavioral_distance(r, r, n_positions=40, seed=0)
    assert d.distance == 0.0
    assert d.knob_distance == 0


def test_goal_inversion_is_behaviourally_far():
    normal = Ruleset(m=3, n=3, k=3)
    misere = Ruleset(m=3, n=3, k=3, misere=True)
    d = behavioral_distance(normal, misere, n_positions=60, seed=0)
    # one knob apart, but optimal play is almost entirely different
    assert d.knob_distance == 1
    assert d.policy_disagreement > 0.5
    assert d.distance > 0.3


def test_behavioral_distance_is_symmetric():
    a = Ruleset(m=3, n=3, k=3)
    b = Ruleset(m=3, n=3, k=3, torus=True)
    ab = behavioral_distance(a, b, n_positions=40, seed=1)
    ba = behavioral_distance(b, a, n_positions=40, seed=1)
    assert ab.distance == pytest.approx(ba.distance, abs=1e-9)


def test_distance_decouples_from_knob_count():
    """The point of A1: knob count is not a proxy for behavioural change."""
    base = Ruleset(m=3, n=3, k=3)
    misere = behavioral_distance(base, Ruleset(m=3, n=3, k=3, misere=True), n_positions=60)
    torus = behavioral_distance(base, Ruleset(m=3, n=3, k=3, torus=True), n_positions=60)
    assert misere.knob_distance == torus.knob_distance == 1
    assert misere.distance != torus.distance  # same knob distance, different reality


def test_distance_to_set_picks_the_nearest():
    held = Ruleset(m=3, n=3, k=3, misere=True)
    train = [Ruleset(m=3, n=3, k=3), Ruleset(m=3, n=3, k=3, misere=True, torus=True)]
    d = distance_to_set(held, train, n_positions=40)
    near = behavioral_distance(held, train[1], n_positions=40)
    assert d.distance == pytest.approx(near.distance, abs=1e-9)
