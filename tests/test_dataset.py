import numpy as np
import pytest

from ruleshift.dataset import (
    build_dataset,
    encode_state,
    load_dataset,
    sample_positions,
    save_dataset,
)
from ruleshift.engine import Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver


def test_sample_positions_distinct_nonterminal_deterministic():
    engine = Engine(Ruleset(m=4, n=4, k=3))
    a = sample_positions(engine, 100, seed=7)
    b = sample_positions(engine, 100, seed=7)
    assert a == b
    assert len(set(a)) == 100
    for s in a:
        assert engine.full_status(s) is None


def test_sample_positions_too_small_variant_raises():
    engine = Engine(Ruleset(m=3, n=3, k=3))
    with pytest.raises(ValueError):
        sample_positions(engine, 100_000, seed=0)


def test_encode_state():
    rules = Ruleset(m=3, n=3, k=3, forbidden=frozenset({(2, 2)}))
    engine = Engine(rules)
    s, _ = engine.play([4, 0])  # X center, O corner; X to move
    arr = encode_state(engine, s)
    assert arr.shape == (3, 3, 3)
    assert arr[0, 1, 1] == 1.0  # current player (X) stone at center
    assert arr[1, 0, 0] == 1.0  # opponent (O) stone
    assert arr[2, 2, 2] == 0.0 and arr[2].sum() == 8  # playable mask
    # perspective flips with the player to move: after X plays (0,1), O is current
    arr2 = encode_state(engine, engine.apply(s, 1))
    assert arr2[0, 0, 0] == 1.0  # O corner now in the current-player plane
    assert arr2[1, 1, 1] == 1.0 and arr2[1, 0, 1] == 1.0  # X stones now opponent


def test_build_save_load_roundtrip(tmp_path):
    rules = Ruleset(m=4, n=3, k=3)
    engine = Engine(rules)
    solver = Solver(engine)
    data = build_dataset(engine, solver, n=40, seed=1)
    assert data["boards"].shape == (40, 3, 3, 4)
    assert data["policy_mask"].shape == (40, 12)
    assert set(np.unique(data["values"])) <= {-1, 0, 1}
    # optimal moves are always legal
    assert np.all(data["policy_mask"] <= data["legal_mask"])
    assert data["policy_mask"].sum(axis=1).min() >= 1
    # labels match the solver
    for i in (0, 13, 39):
        s = tuple(int(x) for x in data["states"][i])
        v, moves = solver.policy(s)
        assert v == data["values"][i]
        assert np.flatnonzero(data["policy_mask"][i]).tolist() == list(moves)

    p = tmp_path / "d.npz"
    save_dataset(p, data, rules)
    loaded, loaded_rules = load_dataset(p)
    assert loaded_rules == rules
    assert np.array_equal(loaded["boards"], data["boards"])
    assert np.array_equal(loaded["states"], data["states"])


def test_build_dataset_explicit_states():
    from ruleshift.dataset import build_dataset as bd

    engine = Engine(Ruleset(m=3, n=3, k=3))
    solver = Solver(engine)
    states = [engine.play([0, 4, 1])[0], engine.play([4])[0]]
    data = bd(engine, solver, n=0, states=states)
    assert [tuple(int(x) for x in s) for s in data["states"]] == states
    assert data["values"][0] == solver.policy(states[0])[0]
