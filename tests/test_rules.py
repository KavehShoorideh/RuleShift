import pytest

from ruleshift.rules import Ruleset, standard_grid


def test_variant_id_stable_and_safe():
    r = Ruleset(m=4, n=3, k=3, gravity=True, misere=True, forbidden=frozenset({(0, 1), (2, 3)}))
    assert r.variant_id == "m4n3k3_grav_mis_f0.1-2.3"
    assert "/" not in r.variant_id and " " not in r.variant_id
    assert Ruleset(m=3, n=3, k=3).variant_id == "m3n3k3"


def test_dict_roundtrip():
    r = Ruleset(m=5, n=4, k=4, torus=True, forbidden=frozenset({(1, 1)}))
    assert Ruleset.from_dict(r.to_dict()) == r


def test_distance():
    a = Ruleset(m=3, n=3, k=3)
    assert a.distance(a) == 0
    b = Ruleset(m=4, n=3, k=3, misere=True)
    assert a.distance(b) == 2
    assert b.distance(a) == 2
    c = Ruleset(m=3, n=3, k=3, forbidden=frozenset({(0, 0)}))
    assert a.distance(c) == 1


def test_standard_grid():
    grid = standard_grid()
    # k=3: all 16 boards; k=4: all but 3x3 -> 15
    assert len(grid) == 31
    assert all(g.k <= max(g.m, g.n) for g in grid)
    assert len({g.variant_id for g in grid}) == 31


def test_validation():
    with pytest.raises(ValueError):
        Ruleset(m=3, n=3, k=1)
    with pytest.raises(ValueError):
        Ruleset(m=3, n=3, k=3, forbidden=frozenset({(3, 0)}))
    with pytest.raises(ValueError):
        Ruleset(m=0, n=3, k=3)


def test_rule_vector():
    r = Ruleset(m=4, n=5, k=4, gravity=True)
    assert r.rule_vector() == (4.0, 5.0, 4.0, 1.0, 0.0, 0.0)
