import pytest

from ruleshift.engine import Engine
from ruleshift.metrics import evaluate_policy, move_regret, samples_to_epsilon
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver


@pytest.fixture(scope="module")
def ttt():
    engine = Engine(Ruleset(m=3, n=3, k=3))
    return engine, Solver(engine)


def test_forced_block_regret(ttt):
    engine, solver = ttt
    s, _ = engine.play([0, 4, 1])  # O must block at 2
    assert move_regret(solver, s, 2) == 0
    for mv in engine.legal_moves(s):
        if mv != 2:
            assert move_regret(solver, s, mv) == 1  # draw thrown away for a loss


def test_blunder_regret_two(ttt):
    engine, solver = ttt
    s, _ = engine.play([0, 6, 1, 7])  # X can win at 2 (or 8); 4 loses outright
    assert move_regret(solver, s, 2) == 0
    assert move_regret(solver, s, 8) == 0
    assert move_regret(solver, s, 4) == 2  # win -> loss


def test_misere_opening_regrets():
    engine = Engine(Ruleset(m=3, n=3, k=3, misere=True))
    solver = Solver(engine)
    s = engine.initial()
    assert move_regret(solver, s, 4) == 0  # center: the only non-losing opening
    assert move_regret(solver, s, 0) == 1


def test_evaluate_policy(ttt):
    engine, solver = ttt
    positions = [engine.play([0, 4, 1])[0], engine.play([0, 6, 1, 7])[0]]

    optimal = lambda s: solver.policy(s)[1][0]
    report = evaluate_policy(solver, optimal, positions)
    assert report.mean_regret == 0.0
    assert report.frac_optimal == 1.0

    worst = {positions[0]: 3, positions[1]: 4}
    report = evaluate_policy(solver, worst.__getitem__, positions)
    assert report.per_position == (1, 2)
    assert report.mean_regret == 1.5
    assert report.frac_optimal == 0.0
    assert report.n_positions == 2

    with pytest.raises(ValueError):
        evaluate_policy(solver, optimal, [])


def test_samples_to_epsilon():
    sizes = [10, 100, 1000]
    assert samples_to_epsilon(sizes, [0.9, 0.4, 0.0], eps=0.5) == 100
    assert samples_to_epsilon(sizes, [0.9, 0.4, 0.1], eps=0.0) is None
    assert samples_to_epsilon([100, 10], [0.0, 0.9], eps=0.5) == 100  # order-safe
    with pytest.raises(ValueError):
        samples_to_epsilon([10], [0.1, 0.2], eps=0.5)
