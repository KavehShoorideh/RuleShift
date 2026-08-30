#!/usr/bin/env python
"""E1 pilot: baseline transfer curves for M0 (plan par.5, E1; feeds Gate G2).

Feasibility-gated k=3 pool (docs/benchmarks.md): 22 training variants,
6 interpolation held-outs (unseen board+flag pairings, all knob values seen),
4 extrapolation held-outs (flag combos never co-seen in training).
Per held-out variant x seed: zero-shot, few-shot fine-tuning at {10,100,1000}
positions, from-scratch controls at {100,1000}; metric = exact per-move regret
on eval positions disjoint from every adaptation pool; random-play anchor.

Outputs: runs/<stamp>_e1_pilot/ (tracker), figures/e1_pilot_*.png + results json.

  python scripts/e1_baseline.py            # pilot defaults (2 seeds)
  python scripts/e1_baseline.py --seeds 5  # the plan's full stats
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.setrecursionlimit(10000)

import numpy as np
import torch

from ruleshift.dataset import build_dataset, load_dataset, sample_positions, save_dataset
from ruleshift.engine import Engine
from ruleshift.metrics import evaluate_policy
from ruleshift.models import M0
from ruleshift.rules import Ruleset
from ruleshift.solver import Solver
from ruleshift.tables import table_path
from ruleshift.tracker import Run
from ruleshift.training import TrainConfig, adapt, eval_model_regret, tensorize, train

# ---------------------------------------------------------------- variant pool
def V(m, n, **kw):
    return Ruleset(m=m, n=n, k=3, **kw)

# Feasibility-gated pilot pool. Misere/torus labeling cost grows fast with
# board area (no threat pruning under misere), so those flags stay on <= 16-cell
# boards; gravity collapses state spaces and is cheap everywhere.
BOARDS = [(m, n) for m in (3, 4, 5) for n in (3, 4, 5)]

TRAIN = (
    [V(m, n) for m, n in BOARDS]
    + [V(m, n, gravity=True) for m, n in [(3, 3), (4, 3), (3, 4), (4, 4), (5, 5)]]
    + [V(m, n, misere=True) for m, n in [(3, 3), (4, 3), (3, 4), (4, 4)]]
    + [V(m, n, torus=True) for m, n in [(3, 3), (4, 4)]]
)
HELD = (
    [(V(5, 4, gravity=True), "interp"), (V(4, 5, gravity=True), "interp"),
     (V(5, 3, misere=True), "interp"), (V(3, 5, misere=True), "interp"),
     (V(4, 3, torus=True), "interp"), (V(3, 4, torus=True), "interp")]
    + [(V(4, 4, gravity=True, misere=True), "extrap"),
       (V(5, 4, gravity=True, misere=True), "extrap"),
       (V(3, 3, misere=True, torus=True), "extrap"),
       (V(3, 3, gravity=True, misere=True, torus=True), "extrap")]
)

# ------------------------------------------------------------------- pipeline
def get_dataset(rules, n, seed, data_dir, cache_dir):
    path = Path(data_dir) / f"{rules.variant_id}.seed{seed}.n{n}.npz"
    if path.exists():
        return load_dataset(path)[0]
    engine = Engine(rules)
    solver = Solver(engine)
    tt = table_path(cache_dir, rules, "tt")
    if tt.exists():
        solver.load(tt)
    t0 = time.time()
    data = build_dataset(engine, solver, n=n, seed=seed, strict=False)
    solver.save(tt)
    save_dataset(path, data, rules)
    got = len(data["values"])
    note = "" if got == n else f" (variant capacity: {got} < {n} requested)"
    print(f"  labeled {rules.variant_id}: {got} positions in {time.time() - t0:.1f}s{note}", flush=True)
    return data


def held_pack(rules, train_n, n_eval, seed_split, cache_dir):
    """Capacity-aware disjoint split for a held-out variant: sample once,
    shuffle deterministically, carve eval positions and an adaptation pool
    that never overlap; label the pool with the (cached) solver."""
    engine = Engine(rules)
    solver = Solver(engine)
    tt = table_path(cache_dir, rules, "tt")
    if tt.exists():
        solver.load(tt)
    cand = sample_positions(engine, train_n + 2 * n_eval, seed=0, strict=False)
    rng = np.random.default_rng(seed_split)
    rng.shuffle(cand)
    eval_take = min(n_eval, max(100, len(cand) // 4))
    if len(cand) < eval_take + 100:
        raise ValueError(f"{rules.variant_id}: only {len(cand)} states; too small for the protocol")
    positions = cand[:eval_take]
    pool = cand[eval_take : eval_take + train_n]
    data = build_dataset(engine, solver, n=len(pool), states=pool)
    solver.save(tt)
    return engine, solver, positions, data, len(pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--train-positions", type=int, default=2000)
    ap.add_argument("--eval-positions", type=int, default=300)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 10, 100, 1000])
    ap.add_argument("--scratch-budgets", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--pretrain-steps", type=int, default=6000)
    ap.add_argument("--adapt-steps", type=int, default=500)
    ap.add_argument("--scratch-steps", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--data-dir", default=str(ROOT / "data/datasets"))
    ap.add_argument("--cache-dir", default=str(ROOT / "data/tables"))
    ap.add_argument("--fig-dir", default=str(ROOT / "figures"))
    ap.add_argument("--label-only", action="store_true", help="build/cache datasets, then exit")
    args = ap.parse_args()

    cfg = vars(args) | {
        "train_variants": [r.variant_id for r in TRAIN],
        "held": [(r.variant_id, s) for r, s in HELD],
        "model": "M0",
    }
    with Run(ROOT / "runs", "e1_pilot", cfg) as run:
        print(f"run dir: {run.dir}", flush=True)
        t_all = time.time()

        print(f"[1/4] labeling {len(TRAIN)} train + {len(HELD)} held-out variants", flush=True)
        train_tensors = [
            tensorize(get_dataset(r, args.train_positions, 0, args.data_dir, args.cache_dir), r)
            for r in TRAIN
        ]
        held_packs = []
        for i, (rules, split) in enumerate(HELD):
            t0 = time.time()
            engine, solver, positions, data, pool_size = held_pack(
                rules, args.train_positions, args.eval_positions, 9000 + i, args.cache_dir
            )
            print(f"  held-out {rules.variant_id}: pool {pool_size}, eval {len(positions)} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            dist = min(rules.distance(t) for t in TRAIN)
            rng = np.random.default_rng(500 + i)
            rand_fn = lambda s, _e=engine, _r=rng: _e.legal_moves(s)[_r.integers(len(_e.legal_moves(s)))]
            rand_regret = evaluate_policy(solver, rand_fn, positions).mean_regret
            held_packs.append(
                dict(rules=rules, split=split, dist=dist, tensors=tensorize(data, rules),
                     engine=engine, solver=solver, positions=positions, random_regret=rand_regret)
            )
            run.log(kind="variant", variant=rules.variant_id, split=split, dist=dist,
                    random_regret=rand_regret)
        if args.label_only:
            run.finish(label_only=True)
            print("labeling done (label-only mode)", flush=True)
            return

        results = []
        for seed in range(args.seeds):
            print(f"[2/4] seed {seed}: pretraining M0 on {len(TRAIN)} variants", flush=True)
            model = M0(hidden=args.hidden, depth=args.depth)
            torch.manual_seed(seed)
            train(model, train_tensors,
                  TrainConfig(steps=args.pretrain_steps, batch=256, seed=seed),
                  log=lambda **kv: run.log(kind="pretrain", seed=seed, **kv))
            torch.save(model.state_dict(), run.dir / f"m0_seed{seed}.pt")

            print(f"[3/4] seed {seed}: adaptation sweep over {len(held_packs)} held-outs", flush=True)
            for p in held_packs:
                for n in args.budgets:
                    adapted = adapt(model, p["tensors"], n,
                                    TrainConfig(steps=args.adapt_steps, batch=min(64, max(n, 1)),
                                                seed=seed))
                    rep = eval_model_regret(adapted, p["engine"], p["solver"], p["positions"])
                    row = dict(variant=p["rules"].variant_id, split=p["split"], dist=p["dist"],
                               seed=seed, method="finetune", budget=n,
                               mean_regret=rep.mean_regret, frac_optimal=rep.frac_optimal,
                               random_regret=p["random_regret"])
                    results.append(row)
                    run.log(kind="result", **row)
                for n in args.scratch_budgets:
                    scratch = M0(hidden=args.hidden, depth=args.depth)
                    torch.manual_seed(seed * 1000 + n)
                    train(scratch, p["tensors"].subsample(n, seed),
                          TrainConfig(steps=args.scratch_steps, batch=min(64, n), seed=seed))
                    rep = eval_model_regret(scratch, p["engine"], p["solver"], p["positions"])
                    row = dict(variant=p["rules"].variant_id, split=p["split"], dist=p["dist"],
                               seed=seed, method="scratch", budget=n,
                               mean_regret=rep.mean_regret, frac_optimal=rep.frac_optimal,
                               random_regret=p["random_regret"])
                    results.append(row)
                    run.log(kind="result", **row)
                print(f"    {p['rules'].variant_id:28s} done", flush=True)

        print("[4/4] figures", flush=True)
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(exist_ok=True)
        (fig_dir / "e1_pilot_results.json").write_text(json.dumps(
            {"config": {k: v for k, v in cfg.items() if k != "held"} | {"held": cfg["held"]},
             "results": results}, indent=1))
        make_figures(results, args.budgets, fig_dir)
        run.finish(minutes=round((time.time() - t_all) / 60, 1), n_results=len(results))
        print(f"done in {(time.time() - t_all) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------- plots
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def agg(rows, key):
    from collections import defaultdict
    d = defaultdict(list)
    for r in rows:
        d[key(r)].append(r["mean_regret"])
    return {k: float(np.mean(v)) for k, v in d.items()}


def make_figures(results, budgets, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ft = [r for r in results if r["method"] == "finetune"]
    sc = [r for r in results if r["method"] == "scratch"]
    xs = {b: i for i, b in enumerate(budgets)}

    # ---- adaptation curves
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.set_facecolor(SURFACE)
    style_axes(ax)
    for split, color, label in [("interp", BLUE, "held-out: new board size"),
                                ("extrap", ORANGE, "held-out: new knob combo")]:
        rows = [r for r in ft if r["split"] == split]
        for variant in sorted({r["variant"] for r in rows}):
            v = agg([r for r in rows if r["variant"] == variant], lambda r: r["budget"])
            ax.plot([xs[b] for b in budgets], [v[b] for b in budgets],
                    color=color, linewidth=1.1, alpha=0.25, zorder=2)
        m = agg(rows, lambda r: r["budget"])
        ys = [m[b] for b in budgets]
        ax.plot([xs[b] for b in budgets], ys, color=color, linewidth=2.2,
                marker="o", markersize=6.5, label=label, zorder=4)
        ax.annotate(label, (xs[budgets[-1]], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=color)
    msc = agg(sc, lambda r: r["budget"])
    sb = sorted(msc)
    ax.plot([xs[b] for b in sb], [msc[b] for b in sb], color=MUTED, linewidth=2.0,
            linestyle=(0, (4, 3)), marker="o", markersize=6, label="from scratch", zorder=3)
    ax.annotate("from scratch", (xs[sb[0]], msc[sb[0]]), xytext=(-8, 10),
                textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    rand = float(np.mean([r["random_regret"] for r in ft]))
    ax.axhline(rand, color=BASE, linewidth=1.2, linestyle=(0, (2, 3)))
    ax.annotate("random play", (0.02, rand), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xticks(list(xs.values()), [("0\n(zero-shot)" if b == 0 else str(b)) for b in budgets])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("adaptation samples from the new variant", color=INK2, fontsize=10)
    ax.set_ylabel("mean exact regret", color=INK2, fontsize=10)
    ax.set_title("M0 baseline: adapting to changed rules (E1 pilot)",
                 color=INK, fontsize=12, loc="left", pad=14, fontweight="semibold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "e1_pilot_adaptation.png", facecolor=SURFACE)
    plt.close(fig)

    # ---- zero-shot regret vs rule distance
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.set_facecolor(SURFACE)
    style_axes(ax)
    zs = [r for r in ft if r["budget"] == 0]
    per_var = {}
    for r in zs:
        per_var.setdefault(r["variant"], dict(dist=r["dist"], vals=[], rand=r["random_regret"]))
        per_var[r["variant"]]["vals"].append(r["mean_regret"])
    jitter = {v: (i % 3 - 1) * 0.06 for i, v in enumerate(sorted(per_var))}
    for i, (v, d) in enumerate(sorted(per_var.items())):
        x, y = d["dist"] + jitter[v], float(np.mean(d["vals"]))
        ax.scatter([x], [y], s=64, color=BLUE, zorder=4)
        ax.annotate(v, (x, y), xytext=(0, -12 if i % 2 else 8), textcoords="offset points",
                    ha="center", fontsize=6.5, color=MUTED)
    rand = float(np.mean([d["rand"] for d in per_var.values()]))
    ax.axhline(rand, color=BASE, linewidth=1.2, linestyle=(0, (2, 3)))
    ax.annotate("random play (mean)", (0.02, rand), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xticks(sorted({d["dist"] for d in per_var.values()}))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("rule distance to nearest training variant", color=INK2, fontsize=10)
    ax.set_ylabel("zero-shot mean exact regret", color=INK2, fontsize=10)
    ax.set_title("Zero-shot regret vs. rule distance (E1 pilot)",
                 color=INK, fontsize=12, loc="left", pad=14, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(fig_dir / "e1_pilot_zeroshot.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
