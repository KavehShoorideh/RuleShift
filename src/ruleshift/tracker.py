"""Minimal file-based experiment tracker (plan, Sep w1-2).

One run = one directory under the runs root: config.json (with git commit),
metrics.jsonl (append-only), summary.json on finish. No services, no deps.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except Exception:
        return None


class Run:
    def __init__(self, root: str | Path, name: str, config: dict):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = Path(root) / f"{stamp}_{name}"
        run_dir = base
        for suffix in range(2, 100):
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                run_dir = base.with_name(f"{base.name}-{suffix}")
        self.dir = run_dir
        cfg = dict(config)
        cfg["_git_commit"] = _git_commit()
        cfg["_started"] = stamp
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        self._metrics = open(self.dir / "metrics.jsonl", "a")
        self._finished = False

    def log(self, **metrics) -> None:
        self._metrics.write(json.dumps(metrics, default=str) + "\n")
        self._metrics.flush()

    def finish(self, status: str = "ok", **summary) -> None:
        if self._finished:
            return
        self._finished = True
        self._metrics.close()
        summary = {"status": status, **summary}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(status="ok" if exc_type is None else f"error: {exc_type.__name__}")


def load_metrics(run_dir: str | Path) -> list[dict]:
    with open(Path(run_dir) / "metrics.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_runs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "config.json").exists())
