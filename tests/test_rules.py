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
    # amendment A4: default tier is 3..5, the largest board tier dropped
    grid = standard_grid()
    # k=3: all 9 boards; k=4: all but 3x3 -> 8
    assert len(grid) == 17
    assert all(g.k <= max(g.m, g.n) for g in grid)
    assert max(max(g.m, g.n) for g in grid) == 5
    assert len({g.variant_id for g in grid}) == 17
    assert len(standard_grid(ms=range(3, 7), ns=range(3, 7))) == 31  # opt back in


def test_validation():
    with pytest.raises(ValueError):
        Ruleset(m=3, n=3, k=1)
    with pytest.raises(ValueError):
        Ruleset(m=3, n=3, k=3, forbidden=frozenset({(3, 0)}))
    with pytest.raises(ValueError):
        Ruleset(m=0, n=3, k=3)


def test_rule_vector():
    r = Ruleset(m=4, n=5, k=4, gravity=True)
    assert r.rule_vector() == (4.0, 5.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert Ruleset(m=3, n=3, k=3, capture=True, scoring=True).rule_vector()[-2:] == (1.0, 1.0)


def test_a2_knobs_in_id_roundtrip_and_distance():
    r = Ruleset(m=3, n=3, k=3, capture=True, scoring=True)
    assert r.variant_id == "m3n3k3_cap_sco"
    assert Ruleset.from_dict(r.to_dict()) == r
    assert Ruleset(m=3, n=3, k=3).distance(r) == 2
