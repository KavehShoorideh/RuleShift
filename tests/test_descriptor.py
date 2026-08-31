"""Amendment A3 mode 2: knob-free rule description."""
import numpy as np

from ruleshift.descriptor import behavioral_signature, descriptor_planes, rule_description
from ruleshift.engine import Engine
from ruleshift.rules import Ruleset


def E(**kw):
    return Engine(Ruleset(**kw))


def test_planes_expose_structure_not_knobs():
    planes = descriptor_planes(E(m=3, n=3, k=3))
    assert planes.shape == (3, 3, 3)
    assert planes[0].sum() == 9  # all playable
    assert planes[1, 1, 1] == planes[1].max()  # centre sits on the most lines
    assert planes[2].sum() == 9  # every cell legal to open on

    grav = descriptor_planes(E(m=3, n=3, k=3, gravity=True))
    assert grav[2].sum() == 3  # gravity shows up as placement restriction
    assert np.array_equal(grav[2][0], np.ones(3))  # only the bottom row

    forb = descriptor_planes(E(m=3, n=3, k=3, forbidden=frozenset({(1, 1)})))
    assert forb[0, 1, 1] == 0.0 and forb[1, 1, 1] == 0.0


def test_signature_detects_goal_polarity():
    normal = behavioral_signature(E(m=3, n=3, k=3))
    misere = behavioral_signature(E(m=3, n=3, k=3, misere=True))
    assert normal[2] > 0.5   # last mover usually wins
    assert misere[2] < -0.5  # ... and usually loses under goal inversion


def test_signature_detects_scoring_and_capture():
    normal = behavioral_signature(E(m=3, n=3, k=3))
    scoring = behavioral_signature(E(m=3, n=3, k=3, scoring=True))
    assert scoring[1] == 1.0 and normal[1] < 1.0  # scoring always fills the board
    capture = behavioral_signature(E(m=3, n=3, k=3, capture=True))
    assert capture[3] > 0.0 and normal[3] == 0.0  # only capture churns ownership


def test_signature_detects_gravity_via_branching():
    assert behavioral_signature(E(m=4, n=4, k=3, gravity=True))[4] < \
           behavioral_signature(E(m=4, n=4, k=3))[4]


def test_signature_is_deterministic_and_distinguishes_variants():
    e = E(m=3, n=3, k=3)
    assert np.array_equal(behavioral_signature(e), behavioral_signature(e))
    sigs = [
        tuple(np.round(behavioral_signature(E(m=3, n=3, k=3, **kw)), 4))
        for kw in ({}, dict(misere=True), dict(torus=True), dict(gravity=True),
                   dict(capture=True), dict(scoring=True))
    ]
    assert len(set(sigs)) == len(sigs)  # every knob leaves a distinct fingerprint


def test_rule_descriptor_via_the_game_interface():
    d = E(m=4, n=3, k=3, gravity=True).rule_descriptor()
    assert set(d) == {"planes", "signature", "n_actions"}
    assert d["n_actions"] == 12
    assert d["signature"].shape == (7,)
