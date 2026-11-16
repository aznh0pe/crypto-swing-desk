"""Trade-level and account-level backtests. Every result is net of trading fees.

Conservative fill and exit rules:
* A limit entry fills only if the day's low trades below the limit (a touch is not a fill); fill = the
  better of the open and the limit. It pays the maker fee. A market entry buys the next open, pays the
  taker fee and slippage.
* On the day a limit fills, only the stop and close-based exits count (we cannot know whether the high
  came before the fill).
* If a later day touches both stop and target, the stop is assumed first. Gaps through a level fill at the open.
* Targets are resting limit sells (maker fee). Stops, signal exits and time exits are market sells
  (taker fee plus slippage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import condition_series

# Wider-list coins trade thinner books: a limit there only counts as filled when price trades 0.30% through it,
# and market exits (stops, signal and time exits) slip 0.35% instead of 0.10%.
DEFAULT_COSTS = {"maker_pct": 0.50, "taker_pct": 0.90, "slippage_pct": 0.10, "wide_slippage_pct": 0.35, "wide_fill_margin_pct": 0.30}
FEATURES = ["rsi2", "rsi14", "atr_pct", "dist20", "dist50", "dist200", "dd90", "rs_pct", "rs30_pct", "vol_ratio", "ret7",
            "ret30", "adx14", "ibs", "streak", "is_major", "btc_dist200", "btc_dist50", "btc_ret7", "btc_ret30",
            "btc_vol_ratio", "btc_rsi14", "breadth50", "breadth200", "alt_ret7", "fng", "dv30", "washed", "flush7", "up7"]


def _a(x):
    return x.values if hasattr(x, "values") else x


def features_at(ind: dict, t_idx, j_idx) -> pd.DataFrame:
    mk = ind["market"]
    out = {}
    for f in FEATURES:
        if f in mk.columns:
            out[f] = mk[f].values[t_idx]
        else:
            out[f] = np.asarray(ind[f].values[t_idx, j_idx], dtype=float)
    return pd.DataFrame(out)


def simulate(plan: dict, ind: dict, costs: dict | None = None, last_signal_row: int | None = None,
             include_unfilled: bool = False, manage: dict | None = None) -> pd.DataFrame:
    """Every signal of `plan` as a trade row. `manage` replays the stop rule the scans apply live:
    {"lock_atr": 2.0, "lock_to": 0.5} raises the stop to fill + 0.5 ATR after a close 2 ATR above the fill."""
    O, H, L, C = (_a(ind[k]) for k in ("open", "high", "low", "close"))
    ATR, SMA5, RSI2, MAX7, EMA10 = (_a(ind[k]) for k in ("atr10", "sma5", "rsi2", "max7c", "ema10"))
    k = {**DEFAULT_COSTS, **(costs or {})}
    maker, taker, slip_core = k["maker_pct"] / 100, k["taker_pct"] / 100, k["slippage_pct"] / 100
    slip_wide, margin_wide = k.get("wide_slippage_pct", 0.35) / 100, k.get("wide_fill_margin_pct", 0.30) / 100
    CORE = _a(ind["is_core"]) if "is_core" in ind else np.ones(C.shape, dtype=bool)
    last_bar = np.where(np.isfinite(C), np.arange(C.shape[0])[:, None], -1).max(axis=0)   # each coin's last priced day
    mg = manage or {}
    lock_atr = float(mg.get("lock_atr") or 0) or None
    lock_to = float(mg.get("lock_to", 0.5))
    T = C.shape[0]
    mask = _a(plan["mask"]).copy()
    mask[(T - 1 if last_signal_row is None else last_signal_row):, :] = False
    t_idx, j_idx = np.nonzero(mask)
    if len(t_idx) == 0:
        return pd.DataFrame()
    LIM = _a(plan["limit"])
    ss, ts = plan["stop"], plan["target"]
    STOP_LV = _a(ss["frame"]) if ss["kind"] == "level" else None
    TGT_LV = _a(ts["frame"]) if ts["kind"] == "level" else None
    market_entry = plan["entry_kind"] == "open_next"
    valid, hold, dyn = max(1, int(plan.get("valid_days", 1))), int(plan["max_hold"]), plan.get("dyn_exit", "none")
    rows = []
    for t, j in zip(t_idx.tolist(), j_idx.tolist()):
        atr = ATR[t, j]
        if not np.isfinite(atr) or atr <= 0:
            continue
        limit = C[t, j] if market_entry else LIM[t, j]
        if not np.isfinite(limit) or limit <= 0:
            continue
        core = bool(CORE[t, j])
        slip, touch = (slip_core, limit) if core else (slip_wide, limit * (1 - margin_wide))
        fill_i, fill, fee_in = -1, np.nan, maker
        if market_entry:
            if t + 1 < T and np.isfinite(O[t + 1, j]):
                fill_i, fill, fee_in = t + 1, O[t + 1, j] * (1 + slip), taker
        else:
            for d in range(t + 1, min(t + valid, T - 1) + 1):
                lo = L[d, j]
                if np.isfinite(lo) and lo < touch:
                    fill_i, fill = d, min(O[d, j], limit)
                    break
        if fill_i < 0:
            if include_unfilled:
                rows.append((t, j, -1, np.nan, -1, np.nan, "unfilled", 0, np.nan, np.nan, limit, np.nan, core))
            continue
        if STOP_LV is not None:
            stop = STOP_LV[t, j]
            if not np.isfinite(stop) or stop >= fill:
                stop = fill - 2.0 * atr
        else:
            stop = fill - ss["mult"] * atr
        stop = max(stop, fill * 0.5)
        if TGT_LV is not None:
            tgt = TGT_LV[t, j]
            lo_t, hi_t = fill + ts.get("min_atr", 0.5) * atr, fill + ts.get("cap_atr", 5.0) * atr
            tgt = lo_t if not np.isfinite(tgt) else min(max(tgt, lo_t), hi_t)
        else:
            tgt = fill + ts["mult"] * atr
        exit_i, exit_p, reason, fee_out, held = -1, np.nan, "open", taker, 0
        for d in range(fill_i, T):
            o, hi, lo, cl = O[d, j], H[d, j], L[d, j], C[d, j]
            if not np.isfinite(cl):
                continue
            held += 1
            if d == fill_i and not market_entry:
                if lo <= stop:
                    exit_i, exit_p, reason = d, stop * (1 - slip), "stop"
                    break
            else:
                if d > fill_i and o <= stop:
                    exit_i, exit_p, reason = d, o * (1 - slip), "stop"
                    break
                if d > fill_i and o >= tgt:
                    exit_i, exit_p, reason, fee_out = d, o, "target", maker
                    break
                if lo <= stop:
                    exit_i, exit_p, reason = d, stop * (1 - slip), "stop"
                    break
                if hi >= tgt:
                    exit_i, exit_p, reason, fee_out = d, tgt, "target", maker
                    break
            sig = False
            if dyn == "sma5":
                sig = cl > SMA5[d, j]
            elif dyn == "prev_high":
                sig = d > 0 and cl > H[d - 1, j]
            elif dyn == "high7":
                sig = cl >= MAX7[d, j]
            elif dyn == "rsi2_70":
                sig = RSI2[d, j] > 70
            elif dyn == "ema10_lost":
                sig = d > fill_i and cl < EMA10[d, j]
            if sig:
                exit_i, exit_p, reason = d, cl * (1 - slip), "signal"
                break
            if held >= hold:
                exit_i, exit_p, reason = d, cl * (1 - slip), "time"
                break
            if lock_atr and cl - fill >= lock_atr * atr and stop < fill + lock_to * atr:
                stop = fill + lock_to * atr
        if exit_i < 0 and 0 <= last_bar[j] < T - 4 and last_bar[j] >= fill_i:
            # the coin stopped trading (delisted) while the trade was open: out at its last price, never dropped
            exit_i, exit_p, reason = int(last_bar[j]), C[last_bar[j], j] * (1 - slip), "delisted"
        net = (exit_p * (1 - fee_out)) / (fill * (1 + fee_in)) - 1 if exit_i >= 0 else np.nan
        rows.append((t, j, fill_i, fill, exit_i, exit_p, reason, held, stop, tgt, limit, net, core))
    if not rows:
        return pd.DataFrame()
    a = np.array(rows, dtype=object)
    df = pd.DataFrame({"t": a[:, 0].astype(int), "j": a[:, 1].astype(int), "fill_i": a[:, 2].astype(int),
                       "fill": a[:, 3].astype(float), "exit_i": a[:, 4].astype(int), "exit": a[:, 5].astype(float),
                       "reason": a[:, 6], "held": a[:, 7].astype(int), "stop": a[:, 8].astype(float),
                       "target": a[:, 9].astype(float), "limit": a[:, 10].astype(float), "ret": a[:, 11].astype(float),
                       "tier": np.where(a[:, 12].astype(bool), "core", "wide")})
    dates, tick = ind["close"].index, ind["close"].columns
    df["strategy"] = plan["name"]
    df["ticker"] = tick[df["j"].values]
    df["signal_date"] = dates[df["t"].values]
    df["filled"] = df["fill_i"] >= 0
    ok = df["filled"] & (df["exit_i"] >= 0)
    df["gross"] = np.where(ok, df["exit"] / df["fill"] - 1, np.nan)
    risk = (df["fill"] - df["stop"]).where(df["fill"] > df["stop"])
    df["r_mult"] = np.where(ok, df["ret"] * df["fill"] / risk, np.nan)
    df["win"] = df["ret"] > 0
    df["exit_date"] = pd.NaT
    df.loc[ok, "exit_date"] = dates[df.loc[ok, "exit_i"].values]
    df["fill_date"] = pd.NaT
    df.loc[df["filled"], "fill_date"] = dates[df.loc[df["filled"], "fill_i"].values]
    df["condition"] = condition_series(ind["market"]).values[df["t"].values]
    return pd.concat([df.reset_index(drop=True), features_at(ind, df["t"].values, df["j"].values)], axis=1)


def done(tr: pd.DataFrame) -> pd.DataFrame:
    return tr[tr["filled"] & tr["ret"].notna()] if len(tr) else tr


def stats(tr: pd.DataFrame) -> dict:
    t = done(tr)
    n = int(len(t))
    if n == 0:
        return {"trades": 0}
    wins, losses = t[t["ret"] > 0]["ret"], t[t["ret"] <= 0]["ret"]
    gw, gl = wins.sum(), -losses.sum()
    years = max((t["signal_date"].max() - t["signal_date"].min()).days / 365.25, 0.25)
    streak = worst = 0
    for r in t.sort_values("signal_date")["ret"].values:
        streak = streak + 1 if r <= 0 else 0
        worst = max(worst, streak)
    days = t.groupby("signal_date")["ret"].mean()   # one vote per signal day: results from the same day are not independent
    return {"trades": n, "win_rate": round(float((t["ret"] > 0).mean()), 4), "avg_ret": round(float(t["ret"].mean()), 5),
            "median_ret": round(float(t["ret"].median()), 5), "avg_win": round(float(wins.mean()), 5) if len(wins) else 0.0,
            "avg_loss": round(float(losses.mean()), 5) if len(losses) else 0.0,
            "profit_factor": round(float(gw / gl), 3) if gl > 0 else None, "avg_r": round(float(t["r_mult"].mean()), 3),
            "avg_hold": round(float(t["held"].mean()), 2), "trades_per_year": round(n / years, 1),
            "worst_loss": round(float(t["ret"].min()), 4), "max_consec_losses": int(worst),
            "signal_days": int(len(days)), "day_win_rate": round(float((days > 0).mean()), 4),
            "exit_mix": t["reason"].value_counts(normalize=True).round(3).to_dict(),
            "fill_rate": round(float(tr["filled"].mean()), 3) if len(tr) else None}


def yearly(tr: pd.DataFrame) -> dict:
    t = done(tr)
    if t.empty:
        return {}
    g = t.groupby(t["signal_date"].dt.year)["ret"]
    return {str(y): {"trades": int(n), "win_rate": round(float(w), 3), "avg_ret": round(float(a), 4)}
            for y, n, w, a in zip(g.size().index, g.size(), g.apply(lambda s: (s > 0).mean()), g.mean())}


def by_condition(tr: pd.DataFrame) -> dict:
    t = done(tr)
    out = {}
    for cnd, g in t.groupby("condition"):
        w, lo = g[g["ret"] > 0]["ret"].sum(), -g[g["ret"] <= 0]["ret"].sum()
        out[str(cnd)] = {"trades": int(len(g)), "win_rate": round(float((g["ret"] > 0).mean()), 3),
                         "avg_ret": round(float(g["ret"].mean()), 4), "profit_factor": round(float(w / lo), 2) if lo > 0 else None}
    return out


def portfolio(trades: pd.DataFrame, start_capital=1500.0, max_open=4, risk_pct=1.5, max_pos_pct=30.0,
              score_col="score", min_score=None) -> dict:
    """Day-by-day account replay over simulated trades: each signal day the best-scoring setups take the
    free slots; size = risk_pct of capital at the stop, capped at max_pos_pct of capital and by free cash."""
    t = done(trades).copy()
    if t.empty:
        return {"trades": 0, "end_capital": start_capital, "cagr": 0.0, "max_dd": 0.0}
    if score_col not in t:
        t[score_col] = 0.0
    if min_score is not None:
        t = t[t[score_col] >= min_score]
    t = t.sort_values(["signal_date", score_col], ascending=[True, False])
    cap, open_pos, curve, taken, peak, mdd = start_capital, [], [], [], start_capital, 0.0
    for day, g in t.groupby("signal_date", sort=True):
        still = []
        for p in open_pos:
            if p["exit_date"] <= day:
                cap += p["pnl"]
            else:
                still.append(p)
        open_pos = still
        peak = max(peak, cap)
        mdd = max(mdd, 1 - cap / peak)
        held = {p["ticker"] for p in open_pos}
        used = sum(p["cost"] for p in open_pos)
        for r in g.itertuples():
            if len(open_pos) >= max_open:
                break
            if r.ticker in held:
                continue
            risk_frac = max(1e-6, 1 - r.stop / r.fill)
            cost = min(cap * risk_pct / 100 / risk_frac, cap * max_pos_pct / 100, max(0.0, cap - used))
            if cost < 10:
                continue
            open_pos.append({"ticker": r.ticker, "exit_date": r.exit_date, "pnl": cost * r.ret, "cost": cost})
            held.add(r.ticker)
            used += cost
            taken.append(r.ret)
        curve.append((day, cap))
    cap += sum(p["pnl"] for p in open_pos)
    peak = max(peak, cap)
    mdd = max(mdd, 1 - cap / peak)
    yrs = max((t["signal_date"].max() - t["signal_date"].min()).days / 365.25, 0.25)
    tk = np.array(taken) if taken else np.array([0.0])
    return {"trades": len(taken), "win_rate": round(float((tk > 0).mean()), 4), "end_capital": round(cap, 2),
            "cagr": round(float((cap / start_capital) ** (1 / yrs) - 1), 4), "max_dd": round(float(mdd), 4),
            "per_year": round(len(taken) / yrs, 1)}
