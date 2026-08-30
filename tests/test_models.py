import numpy as np
import pytest
import torch

from ruleshift.dataset import build_dataset
from ruleshift.engine import Engine
from ruleshift.models import (
    INPUT_DIM,
    M0,
    N_CELLS,
    frame_to_native,
    native_to_frame,
    norm_rule_vector,
    pad_cells,
    pad_planes,
)
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver
from ruleshift.training import (
    TrainConfig,
    adapt,
    eval_model_regret,
    model_policy_fn,
    tensorize,
    train,
)


def test_padding_bottom_left_anchor():
    planes = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)  # n=3, m=4
    out = pad_planes(planes)
    assert out.shape == (3, 6, 6)
    assert np.array_equal(out[:, :3, :4], planes)
    assert out[:, 3:, :].sum() == 0 and out[:, :, 4:].sum() == 0

    vec = np.arange(12, dtype=np.float32)
    padded = pad_cells(vec, n=3, m=4)
    assert padded.reshape(6, 6)[2, 3] == vec.reshape(3, 4)[2, 3]
    assert padded.sum() == vec.sum()


def test_frame_index_roundtrip():
    m = 4
    for cell in range(12):
        assert frame_to_native(native_to_frame(cell, m), m) == cell


def test_m0_forward_and_sizes():
    small, big = M0(hidden=64, depth=2), M0(hidden=256, depth=3)
    x = torch.zeros(5, INPUT_DIM)
    pl, vl = small(x)
    assert pl.shape == (5, N_CELLS) and vl.shape == (5, 3)
    assert small.n_params() < big.n_params()


def test_norm_rule_vector():
    rv = norm_rule_vector(Ruleset(m=6, n=3, k=4, misere=True))
    assert rv.tolist() == [1.0, 0.5, 1.0, 0.0, 1.0, 0.0]


@pytest.fixture(scope="module")
def ttt_data():
    rules = Ruleset(m=3, n=3, k=3)
    engine = Engine(rules)
    solver = Solver(engine)
    data = build_dataset(engine, solver, n=400, seed=0)
    return rules, engine, solver, data


def test_tensorize(ttt_data):
    rules, _, _, data = ttt_data
    t = tensorize(data, rules)
    assert t.x.shape == (400, INPUT_DIM)
    assert torch.allclose(t.policy.sum(dim=1), torch.ones(400))
    assert set(t.value.tolist()) <= {0, 1, 2}
    assert torch.all((t.policy > 0) <= (t.legal > 0))
    sub = t.subsample(10, seed=1)
    assert len(sub) == 10 and torch.equal(sub.x, t.subsample(10, seed=1).x)


def test_model_policy_fn_always_legal():
    for kw in (dict(m=4, n=3, k=3), dict(m=4, n=4, k=4, gravity=True)):
        engine = Engine(Ruleset(**kw))
        fn = model_policy_fn(M0(hidden=32, depth=1), engine)
        s = engine.initial()
        for _ in range(5):
            mv = fn(s)
            assert mv in engine.legal_moves(s)
            s = engine.apply(s, mv)


def test_train_reduces_loss_and_adapt_copies(ttt_data):
    rules, engine, solver, data = ttt_data
    t = tensorize(data, rules)
    model = M0(hidden=32, depth=2)
    cfg = TrainConfig(steps=400, batch=64, lr=3e-3, seed=0)
    losses = []
    train(model, t, cfg, log=lambda **kv: losses.append(kv["loss"]))
    assert losses[-1] < losses[0] * 0.8

    base_regret = eval_model_regret(model, engine, solver, [engine.initial()])
    adapted = adapt(model, t, n_samples=50, cfg=TrainConfig(steps=20, batch=16, seed=0))
    assert adapted is not model  # copy, original untouched
    del base_regret


@pytest.mark.slow
def test_m0_learns_tictactoe(ttt_data):
    rules, engine, solver, data = ttt_data
    from ruleshift.dataset import sample_positions

    t = tensorize(data, rules)
    model = M0(hidden=128, depth=3)
    train(model, t, TrainConfig(steps=1500, batch=128, seed=0))
    eval_positions = sample_positions(engine, 150, seed=999)
    report = eval_model_regret(model, engine, solver, eval_positions)
    assert report.mean_regret < 0.2, f"mean regret {report.mean_regret}"
