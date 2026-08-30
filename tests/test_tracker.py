import json

from ruleshift.tracker import Run, list_runs, load_metrics


def test_run_roundtrip(tmp_path):
    with Run(tmp_path, "e1_baseline", {"model": "m0", "seed": 3}) as run:
        run.log(step=1, regret=0.5)
        run.log(step=2, regret=0.25)
    rows = load_metrics(run.dir)
    assert [r["regret"] for r in rows] == [0.5, 0.25]
    cfg = json.loads((run.dir / "config.json").read_text())
    assert cfg["model"] == "m0" and "_git_commit" in cfg
    assert json.loads((run.dir / "summary.json").read_text())["status"] == "ok"
    assert list_runs(tmp_path) == [run.dir]


def test_run_error_status_and_name_collision(tmp_path):
    try:
        with Run(tmp_path, "boom", {}) as run:
            run.log(step=1)
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert "error" in json.loads((run.dir / "summary.json").read_text())["status"]
    r2 = Run(tmp_path, "boom", {})
    r2.finish()
    assert len(list_runs(tmp_path)) == 2
