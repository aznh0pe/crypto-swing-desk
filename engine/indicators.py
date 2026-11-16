"""Indicator frames (dates x coins) and the market series every rule can read."""
from __future__ import annotations

import numpy as np
import pandas as pd

CORE_DV = 2_000_000    # median 30-day dollar volume of the core list (the most liquid coins)
MIN_DV = 150_000       # ... and of the wider list: every coin at or above this is scanned; below it nothing is
MIN_AGE = 120          # days of history before a coin can signal
N_MAJORS = 12
CONDITIONS = ("bull_calm", "bull_stress", "bear_calm", "bear_stress")
CONDITION_LABEL = {"bull_calm": "Calm bull market", "bull_stress": "Shake-out in a bull market",
                   "bear_calm": "Quiet bear market", "bear_stress": "Bear-market selloff"}
CONDITION_RULE = ("Bull = Bitcoin above its 200-day average. Stressed = Bitcoin down 7% or more over 7 days, or its "
                  "7-day volatility at least 1.35x its 60-day volatility.")


def wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(c, n):
    d = c.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).where(dn != 0, 100.0).where(up.notna())


def _streak(flag: pd.DataFrame) -> pd.DataFrame:
    a = flag.values.astype(bool)
    out = np.zeros(a.shape)
    for i in range(1, a.shape[0]):
        out[i] = np.where(a[i], out[i - 1] + 1, 0)
    return pd.DataFrame(out, index=flag.index, columns=flag.columns)


def compute(w: dict, fng: pd.Series | None = None, min_dv: float = MIN_DV) -> dict:
    o, h, l, c, v = (w[k] for k in ("open", "high", "low", "close", "volume"))
    ind = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    for n in (5, 10, 20, 50, 100, 200):
        ind[f"sma{n}"] = c.rolling(n, min_periods=n).mean()
    for n in (10, 20, 50):
        ind[f"ema{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    for n in (2, 3, 14):
        ind[f"rsi{n}"] = rsi(c, n)
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    ind["atr10"], ind["atr14"] = wilder(tr, 10), wilder(tr, 14)
    ind["atr_pct"] = ind["atr14"] / c
    rng = (h - l).replace(0, np.nan)
    ind["ibs"] = ((c - l) / rng).fillna(0.5)
    sd20 = c.rolling(20, min_periods=20).std()
    ind["pctb"] = (c - (ind["sma20"] - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    ind["min7c"], ind["max7c"] = c.rolling(7, min_periods=7).min(), c.rolling(7, min_periods=7).max()
    for n in (10, 20, 55, 90):
        ind[f"hh{n}"] = h.rolling(n, min_periods=n).max()
    for n in (10, 20, 55):
        ind[f"ll{n}"] = l.rolling(n, min_periods=n).min()
    ind["streak"], ind["upstreak"] = _streak(c < pc), _streak(c > pc)
    ind["slope200"] = ind["sma200"] / ind["sma200"].shift(20) - 1
    ind["slope50"] = ind["sma50"] / ind["sma50"].shift(10) - 1
    for n in (20, 50, 200):
        ind[f"dist{n}"] = c / ind[f"sma{n}"] - 1
    ind["v20"], ind["v5"] = v.rolling(20, min_periods=10).mean(), v.rolling(5, min_periods=3).mean()
    ind["dv30"] = (c * v).rolling(30, min_periods=20).median()
    ind["vol_ratio"] = v / ind["v20"].shift(1).replace(0, np.nan)
    for n in (1, 3, 7, 14, 30, 90):
        ind[f"ret{n}"] = c / c.shift(n) - 1
    # ADX(14)
    up, dn = h.diff(), -l.diff()
    pdm, ndm = up.where((up > dn) & (up > 0), 0.0), dn.where((dn > up) & (dn > 0), 0.0)
    atr14 = ind["atr14"].replace(0, np.nan)
    pdi, ndi = 100 * wilder(pdm, 14) / atr14, 100 * wilder(ndm, 14) / atr14
    ind["adx14"] = wilder(100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan), 14)
    tp = (h + l + c) / 3
    ind["vwap20"] = (tp * v).rolling(20, min_periods=20).sum() / v.rolling(20, min_periods=20).sum().replace(0, np.nan)
    ind["vstd20"] = (c - ind["vwap20"]).rolling(20, min_periods=20).std()
    ind["dd90"] = c / ind["hh90"] - 1
    ind["age"] = c.notna().cumsum().astype(float)
    r = h - l
    ind["nr7"] = (r <= r.rolling(7, min_periods=7).min()).astype(float)
    ind["inside"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)
    ind["body"] = ((c - o) / rng).fillna(0.0)
    ind["upper_wick"] = ((h - np.maximum(o, c)) / rng).fillna(0.0)
    ind["lower_wick"] = ((np.minimum(o, c) - l) / rng).fillna(0.0)
    liquid = (ind["dv30"] >= min_dv) & (ind["age"] >= MIN_AGE) & (c > 0) & (v > 0)
    ind["liquid"] = liquid
    core = liquid & (ind["dv30"] >= CORE_DV)
    ind["is_core"] = core
    ind["vol_rank"] = ind["dv30"].where(liquid).rank(axis=1, ascending=False)
    ind["is_major"] = (ind["vol_rank"] <= N_MAJORS).fillna(False)
    ind["rs_pct"] = ind["ret90"].where(liquid).rank(axis=1, pct=True)
    ind["rs30_pct"] = ind["ret30"].where(liquid).rank(axis=1, pct=True)

    # ---- market series (Bitcoin is the market)
    mk = pd.DataFrame(index=c.index)
    b = c["BTC"] if "BTC" in c else c.median(axis=1)
    mk["btc"] = b
    mk["btc_sma200"], mk["btc_sma50"] = b.rolling(200, min_periods=200).mean(), b.rolling(50, min_periods=50).mean()
    mk["btc_above200"] = (b > mk["btc_sma200"]).astype(float).where(mk["btc_sma200"].notna())
    mk["btc_above50"] = (b > mk["btc_sma50"]).astype(float).where(mk["btc_sma50"].notna())
    mk["btc_dist200"], mk["btc_dist50"] = b / mk["btc_sma200"] - 1, b / mk["btc_sma50"] - 1
    for n in (1, 7, 30):
        mk[f"btc_ret{n}"] = b / b.shift(n) - 1
    bf = b.to_frame("x")
    mk["btc_rsi2"], mk["btc_rsi14"] = rsi(bf, 2)["x"], rsi(bf, 14)["x"]
    r1 = b.pct_change()
    mk["btc_vol_ratio"] = r1.rolling(7, min_periods=7).std() / r1.rolling(60, min_periods=40).std().replace(0, np.nan)
    mk["btc_dd90"] = b / b.rolling(90, min_periods=30).max() - 1
    # breadth and the alt-market move are measured on the core list, so they mean the same thing however wide the scan is
    nliq = core.sum(axis=1).replace(0, np.nan)
    for n in (20, 50, 200):
        mk[f"breadth{n}"] = ((c > ind[f"sma{n}"]) & core).sum(axis=1) / nliq
    alts = [t for t in c.columns if t != "BTC"]
    mk["alt_ret7"] = ind["ret7"][alts].where(core[alts]).median(axis=1)
    # how much of the market is being flushed at once: a coin falling alone is usually its own bad news,
    # a coin falling with everything else is forced selling
    mk["washed"] = ((ind["rsi2"] < 15) & core).sum(axis=1) / nliq          # share of core coins with RSI(2) under 15
    mk["flush7"] = ((ind["ret7"] <= -0.15) & core).sum(axis=1) / nliq      # share of core coins down 15%+ in a week
    mk["up7"] = ((ind["ret7"] >= 0.10) & core).sum(axis=1) / nliq          # share of core coins up 10%+ in a week
    if "ETH" in c and "BTC" in c:
        eb = c["ETH"] / c["BTC"]
        mk["eth_btc_up"] = (eb > eb.rolling(50, min_periods=50).mean()).astype(float)
    else:
        mk["eth_btc_up"] = 0.5
    mk["fng"] = fng.reindex(c.index).ffill(limit=5) if fng is not None else np.nan
    mk["bull"] = mk["btc_above200"]
    mk["stress"] = ((mk["btc_ret7"] <= -0.07) | (mk["btc_vol_ratio"] >= 1.35)).astype(float)
    ind["market"] = mk
    return ind


def condition_series(mk: pd.DataFrame) -> pd.Series:
    bull, stress = mk["bull"], mk["stress"] >= 0.5
    s = pd.Series(np.where(bull >= 0.5, np.where(stress, "bull_stress", "bull_calm"),
                           np.where(stress, "bear_stress", "bear_calm")), index=mk.index, dtype=object)
    return s.where(bull.notna())


def condition_gate(ind: dict, conds, like: pd.DataFrame) -> pd.DataFrame:
    day = condition_series(ind["market"]).reindex(like.index).isin(list(conds)).values
    return pd.DataFrame(np.broadcast_to(day[:, None], like.shape), index=like.index, columns=like.columns)


def base_masks(ind: dict) -> dict:
    return {"liquid": ind["liquid"], "is_major": ind["is_major"], "is_core": ind["is_core"]}
