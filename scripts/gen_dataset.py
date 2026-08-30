#!/usr/bin/env python
"""Generate a supervised-distillation dataset for one variant (exact labels).

Example:
  python scripts/gen_dataset.py --m 4 --n 4 --k 3 --n-positions 1000 --seed 0
  python scripts/gen_dataset.py --m 4 --n 4 --k 4 --gravity --misere \
      --forbidden 0.0,3.3 --out-dir data/datasets
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.setrecursionlimit(10000)

from ruleshift.dataset import build_dataset, save_dataset
from ruleshift.engine import Engine
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver
from ruleshift.tables import table_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--gravity", action="store_true")
    ap.add_argument("--misere", action="store_true")
    ap.add_argument("--torus", action="store_true")
    ap.add_argument("--forbidden", default="", help="comma-separated r.c cells, e.g. 0.0,2.1")
    ap.add_argument("--n-positions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/datasets")
    ap.add_argument("--cache-dir", default="data/tables")
    args = ap.parse_args()

    forbidden = frozenset(
        tuple(int(x) for x in cell.split(".")) for cell in args.forbidden.split(",") if cell
    )
    rules = Ruleset(
        m=args.m, n=args.n, k=args.k, gravity=args.gravity,
        misere=args.misere, torus=args.torus, forbidden=forbidden,
    )
    engine = Engine(rules)
    solver = Solver(engine)
    cache = table_path(args.cache_dir, rules, kind="tt")
    if cache.exists():
        solver.load(cache)

    t0 = time.time()
    data = build_dataset(engine, solver, n=args.n_positions, seed=args.seed)
    solver.save(cache)
    out = Path(args.out_dir) / f"{rules.variant_id}.seed{args.seed}.n{args.n_positions}.npz"
    save_dataset(out, data, rules)
    values = data["values"]
    print(
        f"{rules.variant_id}: {len(values)} positions -> {out} "
        f"({time.time() - t0:.1f}s; W/D/L from mover's view: "
        f"{int((values == 1).sum())}/{int((values == 0).sum())}/{int((values == -1).sum())})"
    )


if __name__ == "__main__":
    main()
