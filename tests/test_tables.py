import pytest

from ruleshift.engine import DRAW, Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver
from ruleshift.tables import (
    TableLimitExceeded,
    build_full_table,
    enumerate_reachable,
    load_table,
    save_table,
    table_path,
)


def test_full_table_tictactoe():
    engine = Engine(Ruleset(m=3, n=3, k=3))
    table = build_full_table(engine)
    init = engine.initial()
    assert table[init] == (DRAW, tuple(range(9)))
    assert 4000 < len(table) < 6000  # reachable non-terminal states
    solver = Solver(engine)
    for s in list(table)[::500]:
        assert table[s] == solver.policy(s)


def test_enumerate_limit():
    engine = Engine(Ruleset(m=4, n=4, k=3))
    with pytest.raises(TableLimitExceeded):
        enumerate_reachable(engine, limit=1000)


def test_table_roundtrip(tmp_path):
    rules = Ruleset(m=3, n=3, k=3, misere=True)
    engine = Engine(rules)
    table = build_full_table(engine)
    p = table_path(tmp_path, rules)
    assert p.name == "m3n3k3_mis.table.pkl"
    save_table(p, table, rules)
    assert load_table(p, rules) == table
    with pytest.raises(ValueError):
        load_table(p, Ruleset(m=3, n=3, k=3))
