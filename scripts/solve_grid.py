#!/usr/bin/env python
"""Solve initial-position values across a rule grid; cache per-variant TTs to disk.

Doubles as the solver feasibility benchmark (plan par.3: cache solution tables once).

Examples:
  python scripts/solve_grid.py                       # base grid, no flag axes
  python scripts/solve_grid.py --gravity --misere    # add flag axes to the grid
  python scripts/solve_grid.py --frontier            # include 7x6 Connect Four (+misere)
  python scripts/solve_grid.py --timeout 600 --cache-dir data/tables
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.setrecursionlimit(10000)

from ruleshift.engine import Engine
from ruleshift.rules import Ruleset, standard_grid
from ruleshift.solver import Solver
from ruleshift.tables import table_path

VALUE_NAMES = {1: "first-win", 0: "draw", -1: "second-win"}


class Timeout(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gravity", action="store_true", help="add gravity on/off as a grid axis")
    ap.add_argument("--misere", action="store_true", help="add misere on/off as a grid axis")
    ap.add_argument("--torus", action="store_true", help="add torus on/off as a grid axis")
    ap.add_argument("--frontier", action="store_true", help="include 7x6 gravity (+misere)")
    ap.add_argument("--cache-dir", default="data/tables")
    ap.add_argument("--timeout", type=int, default=900, help="per-variant seconds (0 = none)")
    args = ap.parse_args()

    axes = lambda on: (False, True) if on else (False,)
    variants = standard_grid(
        gravity=axes(args.gravity), misere=axes(args.misere), torus=axes(args.torus)
    )
    if args.frontier:
        variants += [
            Ruleset(m=7, n=6, k=4, gravity=True),
            Ruleset(m=7, n=6, k=4, gravity=True, misere=True),
        ]

    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(Timeout()))
    print(f"{'variant':28s} {'value':>10s} {'nodes':>13s} {'tt':>12s} {'seconds':>9s}")
    for rules in variants:
        engine = Engine(rules)
        solver = Solver(engine)
        cache = table_path(args.cache_dir, rules, kind="tt")
        if cache.exists():
            solver.load(cache)
        t0 = time.time()
        signal.alarm(args.timeout)
        try:
            v = solver.value(engine.initial())
            signal.alarm(0)
            solver.save(cache)
            print(
                f"{rules.variant_id:28s} {VALUE_NAMES[v]:>10s} {solver.nodes:>13,} "
                f"{len(solver.tt):>12,} {time.time() - t0:>8.2f}s",
                flush=True,
            )
        except Timeout:
            print(f"{rules.variant_id:28s} {'TIMEOUT':>10s} {solver.nodes:>13,}", flush=True)
        except MemoryError:
            print(f"{rules.variant_id:28s} {'OOM':>10s} {solver.nodes:>13,}", flush=True)


if __name__ == "__main__":
    main()
