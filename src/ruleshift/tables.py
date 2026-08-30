"""Per-variant solution tables cached to disk (plan par.3).

Full enumeration (every reachable non-terminal state -> exact value + optimal
move set) below a configurable state cap; above it, callers use the solver's
persistent on-demand TT cache (Solver.save/load).
"""
from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

from .engine import Engine, State
from .rules import Ruleset
from .solver import Solver

Table = dict  # State -> (value, optimal_moves)


class TableLimitExceeded(RuntimeError):
    pass


def table_path(cache_dir: str | Path, rules: Ruleset, kind: str = "table") -> Path:
    return Path(cache_dir) / f"{rules.variant_id}.{kind}.pkl"


def enumerate_reachable(engine: Engine, limit: int | None = None) -> list[State]:
    """BFS every reachable NON-terminal state from the initial position."""
    init = engine.initial()
    seen = {init}
    out: list[State] = []
    dq = deque([init])
    while dq:
        s = dq.popleft()
        out.append(s)
        for mv in engine.legal_moves(s):
            s2 = engine.apply(s, mv)
            if engine.status_after(s2, mv) is not None:
                continue
            if s2 not in seen:
                seen.add(s2)
                if limit is not None and len(seen) > limit:
                    raise TableLimitExceeded(
                        f"{engine.rules.variant_id}: more than {limit} reachable states; "
                        "use the on-demand solver cache instead"
                    )
                dq.append(s2)
    return out


def build_full_table(engine: Engine, solver: Solver | None = None, limit: int = 200_000) -> Table:
    """Exact (value, optimal move set) for every reachable non-terminal state."""
    solver = solver or Solver(engine)
    return {s: solver.policy(s) for s in enumerate_reachable(engine, limit)}


def save_table(path: str | Path, table: Table, rules: Ruleset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"variant": rules.to_dict(), "table": table}, f, protocol=pickle.HIGHEST_PROTOCOL
        )


def load_table(path: str | Path, rules: Ruleset) -> Table:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if d["variant"] != rules.to_dict():
        raise ValueError(
            f"table variant mismatch: file is {d['variant']}, expected {rules.to_dict()}"
        )
    return d["table"]
