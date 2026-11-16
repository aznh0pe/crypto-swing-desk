"""End-to-end smoke test on synthetic data: calibrate, scan, review. Run: python3 tests/test_smoke.py"""
import json, pathlib, shutil, subprocess, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.make_synth import make  # noqa: E402

def run(*a):
    r = subprocess.run([sys.executable, "-m", *a], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout

def main():
    w = pathlib.Path(tempfile.mkdtemp())
    make(w / "data")
    run("engine.calibrate", "backtest", "--data", str(w / "data"), "--out", str(w / "bt"))
    P = json.loads((w / "bt" / "params.json").read_text())
    assert P["strategies"] and "grade_table" in P
    (w / "state" / "strategy").mkdir(parents=True)
    shutil.copy(w / "bt" / "params.json", w / "state" / "strategy" / "params.json")
    # a retired strategy never returns, even when the research log still lists it as a candidate
    shipped = json.loads((ROOT / "engine" / "params.json").read_text())
    for rid, why in (shipped.get("retired") or {}).items():
        (w / "state" / "research").mkdir(parents=True, exist_ok=True)
        spec = {"id": rid, "label": rid, "family": "t", "about": "t", "universe": "all", "conditions": ["rsi2 < 15"], "entry": {"kind": "limit_next", "price": "close - 0.5*atr10"},
                "stop": {"atr": 2.5}, "target": {"atr": 2.5}, "exit": "sma5", "max_hold": 6, "valid_days": 1, "params": {}, "grid": {}}
        (w / "state" / "research" / f"{rid}.json").write_text(json.dumps({"id": rid, "verdict": "candidate", "spec": spec}))
    if shipped.get("retired"):
        run("engine.calibrate", "backtest", "--data", str(w / "data"), "--state", str(w / "state"), "--out", str(w / "bt2"))
        P2 = json.loads((w / "bt2" / "params.json").read_text())
        assert not (set(shipped["retired"]) & (set(P2["strategies"]) | set(P2.get("custom") or {}))), "a retired strategy came back"
        assert set(shipped["retired"]) <= set(P2.get("retired") or {})
        run("engine.research", "promote", "--data", str(w / "data"), "--state", str(w / "state"), "--out", str(w / "pr"))
        assert not (w / "pr" / "writes.json").exists() or not (set(shipped["retired"]) & set(json.loads((w / "pr" / "docs" / "strategy" / "params.json").read_text()).get("custom") or {}))
    out = json.loads(run("engine.scan", "run", "--mode", "close", "--data", str(w / "data"), "--state", str(w / "state"), "--out", str(w / "scan")))
    assert out["status"] in ("ok", "partial")
    for wr in json.loads((w / "scan" / "writes.json").read_text()):
        assert pathlib.Path(wr["file_path"]).exists() and {"op", "collection", "doc_id"} <= set(wr)
    run("engine.learn", "review", "--data", str(w / "data"), "--state", str(w / "state"), "--out", str(w / "learn"))
    assert (w / "learn" / "journal.json").exists()
    nx = json.loads(run("engine.research", "next", "--state", str(w / "state"), "--out", str(w / "rs")))
    assert "focus" in nx
    # the fee model: a limit buy that exits at its target pays the maker fee twice
    from engine import backtest as bt
    assert abs(((110 * (1 - 0.005)) / (100 * (1 + 0.005)) - 1) - 0.08905) < 1e-4 and bt.DEFAULT_COSTS["maker_pct"] == 0.5
    print("smoke test passed")

if __name__ == "__main__":
    main()
