"""Backtests, parameter selection, grading and the report the dashboard shows.

  python -m engine.calibrate backtest --data D [--state S] --out O    full calibration (weekly, and at set-up)

Method: each strategy's small parameter grid is searched on the training years only (before
strategies.SPLIT); the chosen setting is then judged once on unseen data. A strategy trades live only
while it passes strategies.ENABLE_RULE, overall or as a specialist in named market conditions.
Grading is by evidence: a setup is A+ when its strategy's record in today's market condition is strong
in training AND on unseen data. Everything is after fees.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data as D
from . import indicators as I
from . import rules, strategies as S

SPECIALIST = {"train_win": 0.62, "train_pf": 1.30, "train_n": 30, "oos_win": 0.58, "oos_pf": 1.10, "oos_n": 12}
WEAK = {"win": 0.50, "train_n": 30, "oos_n": 10}
WIDE_SPLIT = "2025-01-01"     # the wider list has little history before 2023: its own check splits its record here
TOPUP = {"floors": (0.58, 0.60, 0.62, 0.64), "win": 0.62, "avg": 0.004, "pf": 1.25, "min_trades": 40}
DV_BANDS = [(75_000, 100_000, "$75-100K"), (100_000, 150_000, "$100-150K"), (150_000, 250_000, "$150-250K"), (250_000, 500_000, "$250-500K"),
            (500_000, 2_000_000, "$0.5-2M"), (2_000_000, 1e15, "core $2M+")]
FLOORS = [75_000, 100_000, 150_000, 250_000, 500_000]      # the liquidity floors the weekly probe may move between, one step at a time
FLOOR_RULE = {"lower_win": 0.72, "lower_avg": 0.015, "lower_n": 150, "lower_days": 40, "raise_win": 0.62, "raise_n": 100}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ind(data_dir, P=None):
    w, part, meta = D.load_daily(data_dir)
    min_dv = float(((P or {}).get("universe") or {}).get("min_dv") or I.MIN_DV)
    ind = I.compute(w, D.load_fng(data_dir), min_dv=min_dv)
    return ind, w, part, meta


def split_trades(tr: pd.DataFrame, split: str):
    if tr is None or tr.empty:
        return tr, tr
    cut = pd.Timestamp(split)
    return tr[tr["signal_date"] < cut], tr[tr["signal_date"] >= cut]


def tier_of(tr: pd.DataFrame, tier: str) -> pd.DataFrame:
    return tr[tr["tier"] == tier] if tr is not None and len(tr) and "tier" in tr else tr


def pick_combo(spec: dict, ind: dict, P: dict, b: dict):
    """Grid search on the core list's training years only. Win rate first, with floors that keep it honest."""
    best, tried = None, []
    for pr in rules.combos(spec):
        try:
            plan = rules.build_plan(spec, ind, P, b, pr, name=spec["id"])
            tr = bt.simulate(plan, ind, P["costs"], include_unfilled=True, manage=P.get("manage"))
        except Exception as e:
            tried.append({"params": pr, "error": str(e)[:120]})
            continue
        a, _ = split_trades(tier_of(tr, "core"), P["split"])
        st = bt.stats(a) if a is not None and len(a) else {"trades": 0}
        ok = st.get("trades", 0) >= S.ENABLE_RULE["train_n"] and (st.get("fill_rate") or 0) >= 0.30
        obj = (st.get("win_rate", 0) + 0.02 * min(3.0, 100 * st.get("avg_ret", 0))
               + (0.0 if (st.get("profit_factor") or 0) >= 1.2 else -0.15)) if ok else -1 + 0.001 * st.get("trades", 0)
        tried.append({"params": pr, "trades": st.get("trades", 0), "win_rate": st.get("win_rate"), "avg_ret": st.get("avg_ret"),
                      "profit_factor": st.get("profit_factor"), "fill_rate": st.get("fill_rate")})
        if best is None or obj > best[0]:
            best = (obj, pr, tr)
    if best is None:
        return dict(spec.get("params") or {}), pd.DataFrame(), tried
    return best[1], best[2], tried


def cell_ok(tr_c: dict, te_c: dict, q=SPECIALIST) -> bool:
    return bool((tr_c.get("trades", 0) >= q["train_n"] and (tr_c.get("win_rate") or 0) >= q["train_win"]
                 and (tr_c.get("profit_factor") or 9) >= q["train_pf"] and te_c.get("trades", 0) >= q["oos_n"]
                 and (te_c.get("win_rate") or 0) >= q["oos_win"] and (te_c.get("profit_factor") or 9) >= q["oos_pf"]))


def evaluate(spec: dict, ind: dict, P: dict, b: dict) -> dict:
    """One strategy end to end: parameters chosen and judged on the core list (training years, then unseen data),
    per market condition, with the wider list's own record alongside."""
    pr, tr, tried = pick_combo(spec, ind, P, b)
    core = tier_of(tr, "core")
    a, o = split_trades(core, P["split"])
    st_a, st_o = (bt.stats(a) if a is not None and len(a) else {"trades": 0}), (bt.stats(o) if o is not None and len(o) else {"trades": 0})
    ok, checks, reason = S.enable_checks(st_a, st_o)
    bc_a, bc_o = (bt.by_condition(a) if len(tr) else {}), (bt.by_condition(o) if len(tr) else {})
    conds, mode = [], "off"
    if ok:
        mode = "all"   # trade everywhere except conditions where it is clearly weak in both periods
        weak = [c for c in I.CONDITIONS if bc_a.get(c, {}).get("trades", 0) >= WEAK["train_n"] and bc_a[c]["win_rate"] < WEAK["win"]
                and bc_o.get(c, {}).get("trades", 0) >= WEAK["oos_n"] and bc_o[c]["win_rate"] < WEAK["win"]]
        if weak:
            conds, mode = [c for c in I.CONDITIONS if c not in weak], "all_but_weak"
            reason += " Paused in " + ", ".join(I.CONDITION_LABEL[c].lower() for c in weak) + " (under 50% wins there in both periods)."
    else:
        good = [c for c in I.CONDITIONS if cell_ok(bc_a.get(c, {}), bc_o.get(c, {}))]
        if good:
            conds, mode, ok = good, "specialist", True
            reason = ("Specialist: fails the bar overall but passes it in " + ", ".join(I.CONDITION_LABEL[c].lower() for c in good)
                      + f" ({', '.join(f'{100 * bc_a[c]['win_rate']:.0f}% training / {100 * bc_o[c]['win_rate']:.0f}% unseen' for c in good)}). It trades only there.")
    wide = tier_of(tr, "wide")
    if conds and wide is not None and len(wide):
        wide = wide[wide["condition"].isin(conds)]
    st_w = bt.stats(wide) if wide is not None and len(wide) else {"trades": 0}
    return {"id": spec["id"], "label": spec["label"], "family": spec.get("family"), "about": spec.get("about"), "params": pr,
            "enabled": bool(ok), "mode": mode, "market_conditions": conds, "reason": reason, "checks": checks,
            "train": st_a, "oos": st_o, "wide": st_w, "by_condition": {"train": bc_a, "oos": bc_o},
            "yearly": bt.yearly(core) if len(tr) else {}, "grid": tried[:40], "rules": rules.describe({**spec, "params": {**(spec.get("params") or {}), **pr}}),
            "_trades": tr}


def grade_table(trades: pd.DataFrame, split: str, G: dict, train_only: bool = False) -> dict:
    """Core list: win odds per strategy x market condition, shrunk toward the strategy's overall rate.
    {sid: {cond: {"p", "n", "days", "train", "oos", "grade"}}}. With train_only the unseen data is not used (honesty check)."""
    out = {}
    t = bt.done(trades)
    if t.empty:
        return out
    cut = pd.Timestamp(split)
    for sid, g in t.groupby("strategy"):
        src = g[g["signal_date"] < cut] if train_only else g
        base = float((src["ret"] > 0).mean()) if len(src) else 0.5
        out[sid] = {}
        for c in I.CONDITIONS:
            x = src[src["condition"] == c]
            a, o = g[(g["condition"] == c) & (g["signal_date"] < cut)], g[(g["condition"] == c) & (g["signal_date"] >= cut)]
            n, w = len(x), int((x["ret"] > 0).sum())
            days = int(x["signal_date"].nunique())
            p = (w + 20 * base) / (n + 20)
            aw, ow = (float((a["ret"] > 0).mean()) if len(a) else None), (float((o["ret"] > 0).mean()) if len(o) else None)
            avg = float(x["ret"].mean()) if n else 0.0
            dwin = float((x.groupby("signal_date")["ret"].mean() > 0).mean()) if n else 0.0     # same-day setups move together: days are the honest count
            strong = (n >= G["aplus_min_n"] and days >= G.get("aplus_min_days", 10) and p >= G["aplus_win"] and avg > 0.004 and dwin >= G.get("aplus_day_win", 0.58)
                      and (aw is None or len(a) < 10 or aw >= G["watch_win"])
                      and (train_only or (ow is not None and len(o) >= 8 and ow >= G["watch_win"])))
            grade = "A+" if strong else ("A" if p >= G["watch_win"] and avg > 0 else "B")
            out[sid][c] = {"p": round(p, 4), "n": int(n), "days": days, "day_win": round(dwin, 3), "avg_ret": round(avg, 4), "train": None if aw is None else round(aw, 3),
                           "train_n": int(len(a)), "oos": None if ow is None else round(ow, 3), "oos_n": int(len(o)), "grade": grade}
    return out


def grade_table_wide(trades: pd.DataFrame, core_table: dict, G: dict) -> dict:
    """Wider list: graded on its own record. A cell is A+ only if the strategy has already earned A or A+ there on
    the core list AND the wider list's own trades clear the A+ bar (more trades and days are asked for, and the
    later half of them must hold up, because this list has no training-years history to check against)."""
    out = {}
    t = bt.done(trades)
    if t.empty:
        return out
    for sid, g in t.groupby("strategy"):
        base = float((g["ret"] > 0).mean())
        out[sid] = {}
        for c in I.CONDITIONS:
            x = g[g["condition"] == c].sort_values("signal_date")
            n, w, days = len(x), int((x["ret"] > 0).sum()), int(x["signal_date"].nunique())
            p = (w + 20 * base) / (n + 20)
            avg = float(x["ret"].mean()) if n else 0.0
            dwin = float((x.groupby("signal_date")["ret"].mean() > 0).mean()) if n else 0.0
            late = x.iloc[n // 2:]
            lw = float((late["ret"] > 0).mean()) if len(late) >= 10 else None
            core_g = ((core_table.get(sid) or {}).get(c) or {}).get("grade", "B")
            strong = (core_g in ("A+", "A") and n >= G.get("wide_min_n", 40) and days >= G.get("wide_min_days", 15) and p >= G["aplus_win"]
                      and avg > 0.004 and dwin >= G.get("aplus_day_win", 0.58) and lw is not None and lw >= G["watch_win"])
            grade = "A+" if strong else ("A" if n >= 10 and p >= G["watch_win"] and avg > 0 else "B")
            out[sid][c] = {"p": round(p, 4), "n": int(n), "days": days, "day_win": round(dwin, 3), "avg_ret": round(avg, 4), "late": None if lw is None else round(lw, 3),
                           "core_grade": core_g, "grade": grade}
    return out


def apply_grades(trades: pd.DataFrame, table: dict, table_wide: dict | None = None) -> pd.DataFrame:
    t = trades.copy()
    tw = table_wide or {}
    cells = [((tw if tier == "wide" else table).get(s, {}).get(c) or {}) for s, c, tier in zip(t["strategy"], t["condition"], t["tier"])]
    t["p"] = [c.get("p", np.nan) for c in cells]
    t["grade"] = [c.get("grade", "B") for c in cells]
    rr = ((t["target"] - t["limit"]) / (t["limit"] - t["stop"]).replace(0, np.nan)).clip(0, 4).fillna(1.0)
    t["score"] = (100 * t["p"].fillna(0.5) + 2.0 * rr + 3.0 * t["rs_pct"].fillna(0.5)).round(1)
    return t


CONFLUENCE_BONUS = 10.0     # score points per extra A+ strategy that flags the same coin on the same day (capped at two extra)


def nested_groups(trades: pd.DataFrame, share: float = 0.9) -> dict:
    """Strategies whose signals are (almost) the same trades count as one voice. When 90%+ of one strategy's
    (coin, day) signals are also another's, their agreement is mechanical, not independent confirmation, so they
    share a group and never earn the agreement bonus from each other. Returns {strategy: group name}."""
    if not len(trades):
        return {}
    sigs = {sid: set(zip(x["ticker"], x["signal_date"])) for sid, x in trades.groupby("strategy")}
    ids = sorted(sigs, key=lambda k: -len(sigs[k]))
    grp = {k: k for k in ids}
    find = lambda k: k if grp[k] == k else find(grp[k])
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            small = min(len(sigs[a]), len(sigs[b]))
            if small and len(sigs[a] & sigs[b]) >= share * small:
                grp[find(b)] = find(a)
    return {k: find(k) for k in ids}


def add_confluence(g: pd.DataFrame, groups: dict | None = None) -> pd.DataFrame:
    """How many independent A+ strategies flag the same coin on the same day. Agreement is the strongest ranking
    signal the desk has measured, so it lifts the score (which decides what the account funds first). Strategies in
    one nested group (see nested_groups) count once."""
    g = g.copy()
    groups = groups or {}
    ap = g[g["grade"] == "A+"].assign(_grp=lambda x: x["strategy"].map(lambda s: groups.get(s, s)))
    k = ap.groupby(["ticker", "signal_date"])["_grp"].nunique().rename("confluence")
    g = g.join(k, on=["ticker", "signal_date"])
    g["confluence"] = g["confluence"].fillna(1).astype(int)
    g.loc[g["grade"] == "A+", "score"] = (g["score"] + CONFLUENCE_BONUS * (g["confluence"].clip(upper=3) - 1)).round(1)
    return g


def confluence_report(ap: pd.DataFrame, split: str) -> list:
    o = one_per_coin(ap)
    o = o[o["signal_date"] >= pd.Timestamp(split)]
    out = []
    for k, label in ((1, "1 strategy"), (2, "2 strategies"), (3, "3 or more")):
        x = o[o["confluence"] == k] if k < 3 else o[o["confluence"] >= 3]
        if len(x):
            out.append({"agree": label, "trades": int(len(x)), "win_rate": round(float((x["ret"] > 0).mean()), 3), "avg_ret": round(float(x["ret"].mean()), 4),
                        "days": int(x["signal_date"].nunique())})
    return out


ZONE_RULE = {"min_n": 150, "min_days": 30, "win": 0.58, "avg": 0.004}


def apply_zones(g: pd.DataFrame, ind: dict, P: dict) -> tuple[pd.DataFrame, dict]:
    """Danger zones. A+ results are split by where Bitcoin sits against its 200-day average; a zone where the A+ book
    wins under 58% with no real average gain, on enough trades and days, in training AND on unseen data, is blocked:
    its setups drop to the watchlist until a calibration shows the zone winning again."""
    g = g.copy()
    d = ind["market"]["btc_dist200"].reindex(pd.to_datetime(g["signal_date"])).values
    g["zone"] = [I.zone_of(x) for x in d]
    cut, R, G = pd.Timestamp(P["split"]), ZONE_RULE, P["grading"]
    apz_all = bt.done(g[g["grade"] == "A+"])
    # 1. Each strategy x coin tier is judged on its own record inside each zone: one that clears the full A+ bar
    #    there has earned its place in that zone whatever the rest of the book does.
    exempt = {}
    for (zone, sid, tier), x in apz_all.groupby(["zone", "strategy", "tier"]):
        tr_, un_ = x[x["signal_date"] < cut], x[x["signal_date"] >= cut]
        n, days = len(x), x["signal_date"].nunique()
        base = min(0.5, float((apz_all[apz_all["zone"] == zone]["ret"] > 0).mean()))   # shrink toward a coin flip, never toward the zone's own specialists
        p = (float((x["ret"] > 0).sum()) + 20 * base) / (n + 20)
        dwin = float((x.groupby("signal_date")["ret"].mean() > 0).mean())
        ok = (n >= G.get("wide_min_n", 40) and days >= G.get("wide_min_days", 15) and p >= G["aplus_win"] and float(x["ret"].mean()) > 0.004 and dwin >= G.get("aplus_day_win", 0.58)
              and len(tr_) >= 10 and len(un_) >= 5 and float((tr_["ret"] > 0).mean()) >= G["watch_win"] and float((un_["ret"] > 0).mean()) >= G["watch_win"])
        if ok:
            exempt.setdefault(zone, {})[f"{sid}|{tier}"] = {"p": round(p, 4), "n": int(n), "days": int(days), "day_win": round(dwin, 3), "avg_ret": round(float(x["ret"].mean()), 4),
                                                            "train": round(float((tr_["ret"] > 0).mean()), 3), "oos": round(float((un_["ret"] > 0).mean()), 3)}
    # 2. A zone is judged on the rest of the book: strategies that earned their own place there must not lift the
    #    zone's average and so reopen it to the strategies that lose in it.
    isx = pd.Series([f"{s_}|{t_}" in (exempt.get(z_) or {}) for z_, s_, t_ in zip(apz_all["zone"], apz_all["strategy"], apz_all["tier"])], index=apz_all.index, dtype=bool)
    ap = one_per_coin(apz_all[~isx])
    table, blocked = {}, []
    for name, lo, hi, label in I.ZONES:
        x = ap[ap["zone"] == name]
        row = {"label": label}
        for key, y in (("all", x), ("train", x[x["signal_date"] < cut]), ("unseen", x[x["signal_date"] >= cut])):
            row[key] = {"trades": int(len(y)), "days": int(y["signal_date"].nunique()), "win_rate": round(float((y["ret"] > 0).mean()), 3) if len(y) else None,
                        "avg_ret": round(float(y["ret"].mean()), 4) if len(y) else None}
        # bad = no real edge: the average trade is under +0.4%, and either it loses money or it wins under 58%.
        # A win rate that edges over 58% must never reopen a zone where the average trade still loses.
        bad = lambda s: bool(s["trades"]) and s["avg_ret"] < R["avg"] and (s["win_rate"] < R["win"] or s["avg_ret"] < 0)
        row["blocked"] = bool(row["all"]["trades"] >= R["min_n"] and row["all"]["days"] >= R["min_days"] and bad(row["all"]) and bad(row["train"]) and bad(row["unseen"]))
        if row["blocked"]:
            blocked.append(name)
        table[name] = row
    exempt = {z: v for z, v in exempt.items() if z in blocked}
    keep = pd.Series([f"{s}|{t}" in (exempt.get(z) or {}) for z, s, t in zip(g["zone"], g["strategy"], g["tier"])], index=g.index)
    g.loc[(g["grade"] == "A+") & g["zone"].isin(blocked) & ~keep, "grade"] = "A"
    return g, {"blocked": blocked, "table": table, "exempt": exempt,
               "rule": f"A zone is blocked when A+ setups there make no real edge (an average trade under +{100 * R['avg']:.1f}% that either loses money or wins under {100 * R['win']:.0f}%), in the training years and on unseen data alike, on {R['min_n']}+ trades over {R['min_days']}+ days. A strategy still trades a blocked zone on a coin tier where its own record there clears the full A+ bar (40+ trades on 15+ days, both periods). Re-tested at every calibration."}


def one_per_coin(t: pd.DataFrame) -> pd.DataFrame:
    """One setup per coin and day, the best-scored, as the scanner sends them."""
    return t.sort_values(["score", "p"], ascending=False).drop_duplicates(["signal_date", "ticker"])


def week_of(dates: pd.Series) -> pd.Series:
    return (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.normalize()


def pace_report(g: pd.DataFrame, since: str, min_week: int) -> dict:
    """How often the A+ bar alone fills a week: setups a week since `since` (every signal, filled or not)."""
    s = one_per_coin(g[(g["grade"] == "A+") & (g["signal_date"] >= pd.Timestamp(since))])
    weeks = pd.date_range(pd.Timestamp(since), g["signal_date"].max(), freq="W-MON")
    wk = week_of(s["signal_date"]).value_counts().reindex(weeks, fill_value=0) if len(weeks) else pd.Series(dtype=float)
    days = s["signal_date"].nunique()
    return {"since": since, "weeks": int(len(wk)), "per_week": round(float(wk.mean()), 2) if len(wk) else 0.0,
            "weeks_with_any": round(float((wk >= 1).mean()), 3) if len(wk) else 0.0,
            "weeks_at_min": round(float((wk >= min_week).mean()), 3) if len(wk) else 0.0, "min_week": min_week,
            "days_with_setups_pct": round(100 * days / max(1, (g["signal_date"].max() - pd.Timestamp(since)).days), 1)}


def simulate_topups(g: pd.DataFrame, floor: float, min_week: int, since: str) -> tuple[pd.DataFrame, pd.Series]:
    """Replays the weekly pace rule day by day: every A+ setup counts first; when the week is behind its pace
    (one more setup each day Monday to Friday, up to min_week), the best-scored grade-A setups with win odds
    at or above `floor` are added. Returns (the top-ups sent, setups per week)."""
    s = one_per_coin(g[g["signal_date"] >= pd.Timestamp(since)])
    picks, counts = [], {}
    for day, d in s.groupby("signal_date", sort=True):
        wk = day - pd.Timedelta(days=day.dayofweek)
        na = int((d["grade"] == "A+").sum())
        so_far = counts.get(wk, 0)
        need = max(0, min(min_week, day.dayofweek + 1) - so_far - na)
        tu = d[(d["grade"] == "A") & (d["p"] >= floor)].sort_values("score", ascending=False).head(need) if need else d.iloc[0:0]
        counts[wk] = so_far + na + len(tu)
        if len(tu):
            picks.append(tu)
    weeks = pd.date_range(pd.Timestamp(since), s["signal_date"].max(), freq="W-MON") if len(s) else []
    wkc = pd.Series({w: counts.get(w, 0) for w in weeks}, dtype=float)
    return (pd.concat(picks) if picks else s.iloc[0:0]), wkc


def calibrate_topups(g: pd.DataFrame, since: str, min_week: int) -> dict:
    """The weekly top-up rule: the lowest win-odds floor whose top-ups still won on unseen data. Off when none does."""
    best, tried = None, []
    for floor in TOPUP["floors"]:
        tu, wk = simulate_topups(g, floor, min_week, since)
        st = bt.stats(tu) if len(tu) else {"trades": 0}
        row = {"floor": floor, "trades": st.get("trades", 0), "win_rate": st.get("win_rate"), "avg_ret": st.get("avg_ret"), "profit_factor": st.get("profit_factor"),
               "day_win_rate": st.get("day_win_rate"), "weeks_at_min": round(float((wk >= min_week).mean()), 3) if len(wk) else 0.0,
               "per_week": round(float(wk.mean()), 2) if len(wk) else 0.0}
        tried.append(row)
        ok = (row["trades"] >= TOPUP["min_trades"] and (row["win_rate"] or 0) >= TOPUP["win"] and (row["avg_ret"] or 0) >= TOPUP["avg"]
              and (row["profit_factor"] or 0) >= TOPUP["pf"] and (row["day_win_rate"] or 0) >= 0.55)
        if ok and best is None:
            best = row
    if best is None:
        return {"ok": False, "min_week": min_week, "tried": tried,
                "note": f"Top-ups are off: no grade-A rule won {100 * TOPUP['win']:.0f}%+ with a +{100 * TOPUP['avg']:.1f}% average trade on unseen data. The weekly pace is served by A+ setups only."}
    return {"ok": True, "min_week": min_week, **best, "tried": tried,
            "note": (f"When a week is behind its pace (one setup a day Monday to Friday, {min_week} a week), the best grade-A setups with win odds of "
                     f"{100 * best['floor']:.0f}%+ are sent as top-ups. On unseen data they won {100 * best['win_rate']:.0f}% of {best['trades']} "
                     f"(average {100 * best['avg_ret']:+.1f}% after fees) and {100 * best['weeks_at_min']:.0f}% of weeks reached {min_week} setups.")}


def band_stats(t: pd.DataFrame, lo: float, hi: float) -> dict | None:
    x = t[(t["dv30"] >= lo) & (t["dv30"] < hi)]
    if not len(x):
        return None
    return {"trades": int(len(x)), "win_rate": round(float((x["ret"] > 0).mean()), 3), "avg_ret": round(float(x["ret"].mean()), 4),
            "days": int(x["signal_date"].nunique()), "day_win_rate": round(float((x.groupby("signal_date")["ret"].mean() > 0).mean()), 3)}


def floor_probe(data_dir, P: dict) -> dict:
    """Should the coin list be wider or narrower? Replays the live A+ book on the band of coins just below the
    floor, with DOUBLE the wider-list friction (thin coins have wide spreads), on unseen data. The floor moves one
    step down only when that band clearly holds up, and one step up when the lowest band in the list stops winning."""
    floor = float(P["universe"]["min_dv"])
    lower = max([f for f in FLOORS if f < floor], default=None)
    upper = min([f for f in FLOORS if f > floor], default=None)
    out = {"floor": floor, "decision": "keep", "note": ""}
    cut, R = pd.Timestamp(P["split"]), FLOOR_RULE
    if lower:
        w, _, _ = D.load_daily(data_dir)
        ind = I.compute(w, D.load_fng(data_dir), min_dv=lower)
        costs = {**P["costs"], "wide_fill_margin_pct": 2 * P["costs"].get("wide_fill_margin_pct", 0.3), "wide_slippage_pct": 2 * P["costs"].get("wide_slippage_pct", 0.35)}
        frames = []
        for sid, plan in S.all_plans(ind, P).items():
            tr = bt.simulate(plan, ind, costs, manage=P.get("manage"))
            mc = P["strategies"][sid].get("market_conditions")
            frames.append(tr[tr["condition"].isin(mc)] if mc else tr)
        t = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(t):
            g = bt.done(apply_grades(t, P.get("grade_table") or {}, P.get("grade_table_wide") or {}))
            ap = one_per_coin(g[(g["grade"] == "A+") & (g["signal_date"] >= cut)])
            st = band_stats(ap, lower, floor)
            out["below"] = {"band": f"${lower / 1000:.0f}-{floor / 1000:.0f}K", "coins_today": int((ind["liquid"].iloc[-1] & (ind["dv30"].iloc[-1] < floor)).sum()), **(st or {"trades": 0})}
            if st and st["trades"] >= R["lower_n"] and st["days"] >= R["lower_days"] and st["win_rate"] >= R["lower_win"] and st["avg_ret"] >= R["lower_avg"] and st["day_win_rate"] >= 0.6:
                out.update({"decision": "lower", "new_floor": lower,
                            "note": f"Coins trading ${lower / 1000:.0f}-{floor / 1000:.0f}K a day held up under doubled friction on unseen data ({100 * st['win_rate']:.0f}% of {st['trades']} A+ trades on {st['days']} days), so the list widens to ${lower / 1000:.0f}K."})
            else:
                out["note"] = (f"Coins just below the floor (${lower / 1000:.0f}-{floor / 1000:.0f}K a day) won {100 * st['win_rate']:.0f}% of {st['trades']} A+ trades on {st['days']} days under doubled friction: "
                               f"short of the {100 * R['lower_win']:.0f}% on {R['lower_days']}+ days needed to widen the list." if st else "No A+ trades in the band below the floor yet.")
    return out


def universe_report(ap: pd.DataFrame, ind: dict, P: dict) -> dict:
    """Is the coin list as wide as the evidence supports? A+ results by liquidity band, unseen years."""
    cut = pd.Timestamp(P["split"])
    t = bt.done(ap)
    t = t[t["signal_date"] >= cut]
    floor = float(P["universe"]["min_dv"])
    bands = []
    for lo, hi, label in DV_BANDS:
        st = band_stats(t, lo, hi) if hi > floor else None
        if st:
            bands.append({"band": label, **st})
    liq, core = ind["liquid"].iloc[-1], ind["is_core"].iloc[-1]
    low = bands[0] if bands else None
    upper = min([f for f in FLOORS if f > floor], default=None)
    raise_it = bool(low and upper and low["trades"] >= FLOOR_RULE["raise_n"] and low["win_rate"] < FLOOR_RULE["raise_win"])
    return {"min_dv": floor, "core_dv": I.CORE_DV, "scanned_today": int(liq.sum()), "core_today": int(core.sum()),
            "wide_today": int((liq & ~core).sum()), "bands": bands, "raise_to": upper if raise_it else None,
            "rule": (f"Every Coinbase USD pair that trades normally, with a median daily dollar volume of ${floor / 1000:.0f}K+ over 30 days and 120+ days of history, is scanned; "
                     "the list is rebuilt from fresh volumes at every scan. Coins under $2M a day are the wider list: graded on their own record, with stricter fill and slippage assumptions. "
                     "Each weekly calibration re-tests the floor and moves it one step when the evidence says so.")}


def run_backtest(data_dir, state_params: dict | None = None, extra_specs: list | None = None, probe: bool = True) -> tuple[dict, dict]:
    P = S.params_with_defaults({k: v for k, v in (state_params or {}).items() if k in ("costs", "risk", "manage", "grading", "custom", "retired", "split", "universe", "pace")})
    ind, w, part, meta = load_ind(data_dir, P)
    b = I.base_masks(ind)
    specs = S.all_specs(P)
    for sp in extra_specs or []:
        specs[sp["id"]] = sp
    results, frames = {}, []
    for sid, spec in specs.items():
        r = evaluate(spec, ind, P, b)
        tr = r.pop("_trades")
        results[sid] = r
        P["strategies"][sid] = {"enabled": r["enabled"], "params": r["params"], "market_conditions": r["market_conditions"], "mode": r["mode"]}
        if r["enabled"] and len(tr):
            if r["market_conditions"]:
                tr = tr[tr["condition"].isin(r["market_conditions"])]
            frames.append(tr)
    allt = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    G, min_week = P["grading"], int(P["pace"]["min_week"])
    table = grade_table(tier_of(allt, "core"), P["split"], G) if len(allt) else {}
    table_w = grade_table_wide(tier_of(allt, "wide"), table, G) if len(allt) else {}
    P["grade_table"], P["grade_table_wide"] = table, table_w
    grading = {"rule": (f"A+ = the strategy's record in today's market condition shows {100 * G['aplus_win']:.0f}%+ wins after fees "
                        f"(at least {G['aplus_min_n']} trades on {G.get('aplus_min_days', 10)}+ separate days, shrunk toward its overall rate), most of its signal days were winners, and it held up on unseen data. "
                        f"Wider-list coins are graded on their own record ({G.get('wide_min_n', 40)}+ trades on {G.get('wide_min_days', 15)}+ days, the later half holding up) "
                        "and only where the strategy has already earned A or A+ on the core list. A = 58%+: watchlist only. When several A+ strategies flag the same coin on the same day, it ranks first.")}
    pace, universe = {}, {}
    if len(allt):
        P["confluence_groups"] = {k: v for k, v in nested_groups(allt).items() if k != v}
        g, zones = apply_zones(add_confluence(apply_grades(allt, table, table_w), P["confluence_groups"]), ind, P)
        P["zones"] = zones
        done_g = bt.done(g)
        ap = done_g[done_g["grade"] == "A+"]
        grading["aplus"] = {**bt.stats(ap), "by_year": bt.yearly(ap), "by_condition": bt.by_condition(ap)}
        grading["aplus_core"], grading["aplus_wide"] = bt.stats(tier_of(ap, "core")), bt.stats(tier_of(ap, "wide"))
        grading["confluence"] = confluence_report(ap, P["split"])
        grading["zones"] = zones
        P["confluence"] = grading["confluence"]
        core_all = tier_of(allt, "core")
        honest = apply_grades(core_all, grade_table(core_all, P["split"], G, train_only=True))
        ho = bt.done(honest[(honest["grade"] == "A+") & (honest["signal_date"] >= pd.Timestamp(P["split"]))])
        grading["honest_check"] = {"what": "Core list: A+ cells chosen from the training years only, measured on unseen data", **bt.stats(ho)}
        wide_all, wcut = tier_of(allt, "wide"), pd.Timestamp(WIDE_SPLIT)
        if len(wide_all):
            hw = apply_grades(wide_all, table, grade_table_wide(wide_all[wide_all["signal_date"] < wcut], table, G))
            hwo = bt.done(hw[(hw["grade"] == "A+") & (hw["signal_date"] >= wcut)])
            grading["honest_check_wide"] = {"what": f"Wider list: A+ cells chosen from its trades before {WIDE_SPLIT[:4]} only, measured on {WIDE_SPLIT[:4]} to now", **bt.stats(hwo)}
        grading["all_enabled"] = bt.stats(done_g)
        rk = P["risk"]
        cutp = pd.Timestamp(P["split"])
        grading["account_replay"] = {"what": f"$1,500 start, A+ setups only, {rk['max_open_positions']} slots, {rk['risk_pct']}% risk a trade, position cap {rk['max_position_pct']}%",
                                     "full_history": bt.portfolio(ap, 1500.0, rk["max_open_positions"], rk["risk_pct"], rk["max_position_pct"]),
                                     "unseen_only": bt.portfolio(ap[ap["signal_date"] >= cutp], 1500.0, rk["max_open_positions"], rk["risk_pct"], rk["max_position_pct"])}
        grading["sizing"] = [{"risk_pct": rp, **{k: bt.portfolio(x, 1500.0, rk["max_open_positions"], rp, rk["max_position_pct"]) for k, x in (("full_history", ap), ("unseen_only", ap[ap["signal_date"] >= cutp]))}}
                             for rp in (1.0, 2.0, 3.0, 5.0)]
        grading["slots"] = [{"slots": n, **bt.portfolio(ap[ap["signal_date"] >= cutp], 1500.0, n, rk["risk_pct"], rk["max_position_pct"])} for n in (4, 6, 8, 12)]
        pace = {"aplus_only": pace_report(g, P["split"], min_week), "last_52_weeks": pace_report(g, str((g["signal_date"].max() - pd.Timedelta(weeks=52)).date()), min_week)}
        P["topup"] = calibrate_topups(g, P["split"], min_week)
        pace["topup"] = P["topup"]
        universe = universe_report(ap, ind, P)
        if probe:                                    # keep the coin list as wide as the evidence supports, and no wider
            try:
                pr = floor_probe(data_dir, P)
            except Exception as e:
                pr = {"decision": "keep", "note": f"The floor probe could not run ({str(e)[:100]})."}
            if universe.get("raise_to"):
                pr = {**pr, "decision": "raise", "new_floor": universe["raise_to"],
                      "note": f"The lowest band in the list ({universe['bands'][0]['band']}) won only {100 * universe['bands'][0]['win_rate']:.0f}% of its A+ trades on unseen data, so the floor rises to ${universe['raise_to'] / 1000:.0f}K."}
            universe["probe"] = pr
            if pr.get("new_floor"):
                sp2 = {**(state_params or {}), "universe": {**P["universe"], "min_dv": float(pr["new_floor"]), "changed": now_iso()[:10], "why": pr["note"]}}
                P2, rep2 = run_backtest(data_dir, sp2, extra_specs, probe=False)
                rep2["universe"]["probe"] = pr
                return P2, rep2
    mk = ind["market"]
    cs = I.condition_series(mk).dropna()
    share = cs.value_counts(normalize=True).round(3).to_dict()
    P.update({"calibrated_at": now_iso(), "data_through": str(ind["close"].index.max().date()), "rev": int((state_params or {}).get("rev", 0)) + 1})
    report = {"generated_at": now_iso(), "data": {"first_bar": str(ind["close"].index.min().date()), "last_bar": str(ind["close"].index.max().date()),
                                                  "pairs": int(ind["close"].shape[1]), "liquid_today": int(ind["liquid"].iloc[-1].sum()),
                                                  "core_today": int(ind["is_core"].iloc[-1].sum()), "source": meta.get("source")},
              "split": P["split"], "costs": P["costs"], "enable_rule": S.ENABLE_RULE, "specialist_rule": SPECIALIST,
              "condition_rule": I.CONDITION_RULE, "condition_share": share, "condition_labels": I.CONDITION_LABEL,
              "strategies": results, "grading": grading, "pace": pace, "universe": universe, "grade_table": table}
    return P, report


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date, dt.datetime)):
        return str(o)
    return o


KEEP_STATS = ("trades", "win_rate", "avg_ret", "profit_factor", "avg_hold", "fill_rate", "signal_days", "day_win_rate", "worst_loss")


def slim_report(report: dict) -> dict:
    """The version of the report the dashboard stores: everything it shows, without the bulky search detail
    (the full report stays in backtest_report.json)."""
    out = {k: v for k, v in report.items() if k not in ("strategies", "grade_table", "specialist_rule", "enable_rule")}
    g = out.get("grading") or {}
    for k in ("aplus", "aplus_core", "aplus_wide", "honest_check", "honest_check_wide", "all_enabled"):   # drop the bulky exit mix
        if isinstance(g.get(k), dict):
            g[k] = {a: b for a, b in g[k].items() if a not in ("exit_mix", "median_ret", "avg_r")}
    out["strategies"] = {}
    for sid, r in report["strategies"].items():
        out["strategies"][sid] = {**{k: r.get(k) for k in ("label", "family", "about", "params", "enabled", "mode", "market_conditions", "reason", "rules")},
                                  "train": {k: r["train"].get(k) for k in KEEP_STATS}, "oos": {k: r["oos"].get(k) for k in KEEP_STATS},
                                  "wide": {k: (r.get("wide") or {}).get(k) for k in ("trades", "win_rate", "avg_ret", "signal_days", "day_win_rate")},
                                  "checks": [{k: c.get(k) for k in ("key", "value", "need", "ok")} for c in r.get("checks", [])]}
    return out


def write_docs_coll(out, docs: list) -> list:
    """docs = [(collection, doc_id, data)]. Writes each under out/docs/ plus out/writes.json for the database."""
    out = pathlib.Path(out)
    writes = []
    for coll, did, doc in docs:
        f = out / "docs" / coll / f"{did}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(clean(doc), separators=(",", ":"), default=str))
        writes.append({"op": "set", "collection": coll, "doc_id": did, "file_path": str(f.resolve())})
    (out / "writes.json").write_text(json.dumps(writes, indent=1))
    return writes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp, extra = None, []
    if a.state:
        from .portfolio import load_state
        st = load_state(a.state)
        sp = st.get("params")
        extra = [r["spec"] for r in st.get("research", {}).values() if r.get("verdict") == "candidate" and r.get("spec")
                 and r["id"] not in ((sp or {}).get("retired") or {})]
    f = S.HERE / "params.json"
    shipped = json.loads(f.read_text()) if f.exists() else {}
    if not (sp and sp.get("strategies")):   # nothing saved yet: start from the settings shipped with the engine
        sp = shipped
    retired = {**(shipped.get("retired") or {}), **(sp.get("retired") or {})}
    custom = {**(shipped.get("custom") or {}), **(sp.get("custom") or {})}   # promoted strategies are never lost...
    rev_of = lambda c: int(((c or {}).get("spec") or c or {}).get("rev", 0) or 0)
    for k, c in (shipped.get("custom") or {}).items():                     # ...and a rule revised in the engine replaces its older stored version
        if rev_of(c) > rev_of(custom.get(k)):
            custom[k] = c
    sp = {**sp, "custom": {k: v for k, v in custom.items() if k not in retired}, "retired": retired}   # ...and retired ones never return
    extra = [e for e in extra if e["id"] not in retired and e["id"] not in sp["custom"]]   # a live strategy's rule lives in custom, not in its old research record
    if a.state:   # backtests use the fees the account really pays (Funds, Desk settings)
        s_ = st.get("settings") or {}
        if s_.get("maker_fee_pct") is not None and s_.get("taker_fee_pct") is not None:
            sp = {**sp, "costs": {**(sp.get("costs") or {}), "maker_pct": float(s_["maker_fee_pct"]), "taker_pct": float(s_["taker_fee_pct"])}}
        if s_.get("max_open_positions"):
            sp = {**sp, "risk": {**(sp.get("risk") or {}), "max_open_positions": int(s_["max_open_positions"]), "risk_pct": float(s_.get("risk_pct") or 2.0),
                                 "max_position_pct": float(s_.get("max_position_pct") or 30.0)}}
    P, report = run_backtest(a.data, sp, extra)
    P["retired"] = retired
    for sid in [e["id"] for e in extra]:
        if sid in retired:
            continue
        if P["strategies"].get(sid, {}).get("enabled"):
            P["custom"][sid] = {"spec": next(e for e in extra if e["id"] == sid), "promoted_at": now_iso()}
        elif sid not in (P.get("custom") or {}):
            P["strategies"].pop(sid, None)
    (out / "params.json").write_text(json.dumps(clean(P), indent=1))
    (out / "backtest_report.json").write_text(json.dumps(clean(report), indent=1))
    write_docs_coll(out, [("strategy", "params", P), ("strategy", "backtest", slim_report(report))])
    for sid, r in report["strategies"].items():
        t, o = r["train"], r["oos"]
        print(f"{sid:16s} {'ON ' if r['enabled'] else 'off'} {r['mode']:12s} train n={t.get('trades', 0):5d} win={t.get('win_rate')} pf={t.get('profit_factor')} avg={t.get('avg_ret')} | "
              f"unseen n={o.get('trades', 0):5d} win={o.get('win_rate')} pf={o.get('profit_factor')} avg={o.get('avg_ret')}")
    print(json.dumps(clean(report["grading"]), indent=1)[:3000])


if __name__ == "__main__":
    main()
