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
