"""Amendment A5: the harness runs against a formalism-agnostic game interface."""
import pytest

from ruleshift.engine import Engine
from ruleshift.interface import Game, check_game
from ruleshift.rules import Ruleset

VARIANTS = [
    dict(m=3, n=3, k=3),
    dict(m=4, n=4, k=4, gravity=True),
    dict(m=3, n=3, k=3, misere=True, torus=True),
    dict(m=3, n=3, k=3, capture=True),
    dict(m=3, n=3, k=3, scoring=True),
    dict(m=4, n=3, k=3, forbidden=frozenset({(0, 0)})),
]


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: Ruleset(**kw).variant_id)
def test_engine_satisfies_game_protocol(kw):
    engine = Engine(Ruleset(**kw))
    assert isinstance(engine, Game)
    check_game(engine)  # raises if the contract is violated


def test_check_game_rejects_a_broken_implementation():
    class NeverTerminates:
        def initial(self):
            return 0

        def legal_moves(self, state):
            return [0]

        def step(self, state, move):
            return state, None

        def full_status(self, state):
            return None

        def n_actions(self):
            return 1

        def rule_descriptor(self):
            return {}

    with pytest.raises(ValueError, match="did not terminate"):
        check_game(NeverTerminates(), max_plies=50)


def test_downstream_modules_use_only_the_interface():
    """Solver/dataset/metrics must not reach past the Game contract."""
    from ruleshift.dataset import build_dataset
    from ruleshift.metrics import move_regret
    from ruleshift.solver import Solver

    engine = Engine(Ruleset(m=3, n=3, k=3, capture=True))
    solver = Solver(engine)
    data = build_dataset(engine, solver, n=20, seed=0, strict=False)
    assert len(data["values"]) > 0
    s = engine.initial()
    assert move_regret(solver, s, engine.legal_moves(s)[0]) >= 0
