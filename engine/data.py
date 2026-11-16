"""Price data for the engine. The feed (pipeline/fetch_data.py, a GitHub Action) publishes the `data`
branch of the crypto-swing-desk repo; everything here is defensive so a scan never dies on bad input."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume"]
DATA_FILES = ["daily.csv.gz", "hourly.csv.gz", "universe.json", "meta.json", "fng.json"]
REPO = "aznh0pe/crypto-swing-desk"


def fetch_branch(dest: str, repo: str = REPO, branch: str = "data", attempts: int = 3) -> dict:
    """Shallow-clone the data branch; fall back to per-file raw downloads."""
    dest_p, report = pathlib.Path(dest), {"method": None, "errors": []}
    for i in range(attempts):
        shutil.rmtree(dest_p, ignore_errors=True)
        try:
            r = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--branch", branch,
                                f"https://github.com/{repo}.git", str(dest_p)], capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                report["method"] = "git"
                return report
            report["errors"].append(r.stderr.strip()[-200:])
        except Exception as e:
            report["errors"].append(repr(e)[:200])
        time.sleep(5 * (i + 1))
    dest_p.mkdir(parents=True, exist_ok=True)
    got = 0
    for f in DATA_FILES:
        try:
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{branch}/{f}", timeout=120) as resp:
                (dest_p / f).write_bytes(resp.read())
            got += 1
        except Exception as e:
            report["errors"].append(f"{f}: {e!r}"[:160])
    if got < 2:
        raise RuntimeError(f"could not fetch the data branch: {report['errors'][-3:]}")
    report["method"] = f"raw({got} files)"
    return report


def _json(path, default=None):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except Exception:
        return default


def load_meta(d) -> dict:
    return _json(pathlib.Path(d) / "meta.json", {}) or {}


def load_universe(d) -> dict:
    js = _json(pathlib.Path(d) / "universe.json", {}) or {}
    return {i["ticker"]: i for i in js.get("items", []) if i.get("ticker")}


def load_fng(d) -> pd.Series | None:
    rows = _json(pathlib.Path(d) / "fng.json", None)
    if not rows:
        return None
    s = pd.Series({pd.Timestamp(r["date"]): float(r["value"]) for r in rows}).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _wide(df: pd.DataFrame, col: str) -> dict:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    for c in FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])].drop_duplicates([col, "ticker"], keep="last")
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    out = {f: df.pivot(index=col, columns="ticker", values=f).sort_index().astype("float64") for f in FIELDS}
    return out


def load_daily(d, drop_partial: bool = True):
    """(wide frames of finished daily bars, the forming bar as a dict of Series or None, meta).

    Daily bars close at 00:00 UTC. The feed's meta.partial_day names the bar still forming; it is split off
    so no rule ever reads an unfinished bar as final."""
    d = pathlib.Path(d)
    meta = load_meta(d)
    w = _wide(pd.read_csv(d / "daily.csv.gz", dtype={"ticker": str}), "date")
    part = None
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    pday = pd.Timestamp(meta["partial_day"]) if meta.get("partial_day") else today
    cut = min(pday, today)
    if drop_partial and len(w["close"]) and w["close"].index.max() >= cut:
        mask = w["close"].index >= cut
        last = w["close"].index[mask][-1]
        part = {f: w[f].loc[last].dropna() for f in FIELDS}
        part["day"] = last
        w = {f: v[~mask] for f, v in w.items()}
    full = pd.date_range(w["close"].index.min(), w["close"].index.max(), freq="D")
    w = {f: v.reindex(full) for f, v in w.items()}
    first = w["close"].notna().idxmax()
    for f in ("open", "high", "low", "close"):   # bridge isolated missing days (no trades / feed gaps)
        w[f] = w[f].ffill(limit=2)
    w["volume"] = w["volume"].fillna(0.0)
    for f in FIELDS:  # nothing before a coin's first bar
        w[f] = w[f].where(pd.DataFrame(np.greater.outer(w[f].index.values, first.reindex(w[f].columns).values) |
                                       np.equal.outer(w[f].index.values, first.reindex(w[f].columns).values),
                                       index=w[f].index, columns=w[f].columns))
    return w, part, meta


def load_hourly(d) -> dict | None:
    p = pathlib.Path(d) / "hourly.csv.gz"
    if not p.exists():
        return None
    try:
        return _wide(pd.read_csv(p, dtype={"ticker": str}), "time")
    except Exception:
        return None


def health(w: dict, meta: dict, now=None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    last = w["close"].index.max()
    expected = now.normalize() - pd.Timedelta(days=1)
    lag = int((expected - last).days)
    notes, status = [], "ok"
    if lag > 0:
        status = "stale"
        notes.append(f"Latest finished daily bar is {last.date()}, expected {expected.date()} ({lag} day(s) behind).")
    age = None
    if meta.get("generated_utc"):
        try:
            age = (now - pd.Timestamp(meta["generated_utc"]).tz_convert(None)).total_seconds() / 60
            if age > 180:
                status = "stale" if status == "ok" else status
                notes.append(f"The price feed last refreshed {age / 60:.1f} hours ago.")
        except Exception:
            pass
    if meta.get("problems"):
        notes.append("Feed problems: " + "; ".join(meta["problems"][:3]))
    return {"status": status, "last_bar": str(last.date()), "expected_bar": str(expected.date()), "days_behind": max(lag, 0),
            "feed_age_min": None if age is None else round(age), "pairs": int(w["close"].shape[1]), "notes": notes}
