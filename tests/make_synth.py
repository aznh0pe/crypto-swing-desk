"""Synthetic data in the feed's format, for smoke tests (random walks with a shared market factor)."""
import datetime as dt, gzip, json, pathlib, sys
import numpy as np, pandas as pd

def make(dest, n=24, start="2019-01-01", seed=7):
    rng = np.random.default_rng(seed)
    dest = pathlib.Path(dest); dest.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    days = pd.date_range(start, today, freq="D")
    mkt = rng.normal(0.0006, 0.03, len(days)) + 0.012 * np.sin(np.arange(len(days)) / 180)
    rows, items = [], []
    names = ["BTC", "ETH"] + [f"C{i:02d}" for i in range(n - 2)]
    for k, tk in enumerate(names):
        beta = 1.0 if tk == "BTC" else rng.uniform(0.8, 1.6)
        r = beta * mkt + rng.normal(0, 0.01 if tk == "BTC" else 0.04, len(days))
        c = (30000 if tk == "BTC" else rng.uniform(0.5, 300)) * np.exp(np.cumsum(r - r.mean() + 0.0004))
        o = np.r_[c[0], c[:-1]]
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.02, len(days))))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.02, len(days))))
        v = rng.lognormal(0, 0.5, len(days)) * (5e7 if k < 6 else 6e6) / c
        first = 0 if k < 10 else int(rng.integers(100, 900))
        for i in range(first, len(days)):
            rows.append((days[i].strftime("%Y-%m-%d"), tk, o[i], hi[i], lo[i], c[i], v[i]))
        items.append({"ticker": tk, "product": f"{tk}-USD", "status": "online", "tradable": True, "tick": "0.0001", "dollar_vol_30d": 1e7})
    df = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    df.to_csv(dest / "daily.csv.gz", index=False, compression="gzip")
    h = df[df["date"] >= (today - pd.Timedelta(days=3)).strftime("%Y-%m-%d")].copy()
    h["time"] = h["date"] + "T12:00Z"
    h[["time", "ticker", "open", "high", "low", "close", "volume"]].to_csv(dest / "hourly.csv.gz", index=False, compression="gzip")
    (dest / "universe.json").write_text(json.dumps({"items": items}))
    (dest / "meta.json").write_text(json.dumps({"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "partial_day": today.strftime("%Y-%m-%d"),
                                                "last_daily_bar": (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "source": "synthetic"}))

if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth")
