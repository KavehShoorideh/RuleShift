"""Pruning-free memoized negamax: slow, exact by construction.

The correctness oracle for Solver: no alpha-beta, no TT flags, no move
ordering, no misere shortcuts -- every reachable node is expanded once.
"""
from ruleshift.engine import DRAW, LOSS, WIN, Engine, State


def plain_value(engine: Engine, state: State, memo: dict | None = None) -> int:
    if memo is None:
        memo = {}
    status = engine.full_status(state)
    if status is not None:
        return status
    return _plain(engine, state[0], state[1], memo)


def _plain(engine: Engine, cur: int, opp: int, memo: dict) -> int:
    key = (cur, opp)
    hit = memo.get(key)
    if hit is not None:
        return hit
    best = -2
    for mv in engine.legal_moves((cur, opp)):
        nxt_state, terminal = engine.step((cur, opp), mv)
        if terminal is not None:
            v = -terminal
        else:
            v = -_plain(engine, nxt_state[0], nxt_state[1], memo)
        if v > best:
            best = v
    memo[key] = best
    return best
