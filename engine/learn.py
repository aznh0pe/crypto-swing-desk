"""The daily learning review: grade every alert against what the market did, keep the record by strategy,
market condition and grade, adjust live grading from it, pause what stops working, and write the journal.

  python -m engine.learn review --data D --state S --out O
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as I
from . import strategies as S
from .calibrate import clean, now_iso
from .portfolio import closed_trades, funds, load_state, risk_governor

PAUSE = {"n": 12, "below": 0.45, "restore": 0.55}
LIVE_WEIGHT = 2.0      # a graded live alert counts as two backtest trades when re-estimating win odds


def shadow(a: dict, w: dict, costs: dict) -> dict:
    """What an alert would have done if taken exactly as written, on finished daily bars (same rules as the backtest)."""
    tk = a.get("ticker")
    if tk not in w["close"]:
        return {"state": "no_data"}
    vf, vu = pd.Timestamp(a["valid_from"]).tz_convert(None), pd.Timestamp(a["valid_until"]).tz_convert(None)
    O, H, L, C = (w[k][tk] for k in ("open", "high", "low", "close"))
    days = C.index[(C.index >= vf.normalize())]
    entry, stop, tgt, hold = float(a["entry"]), float(a["stop"]), float(a["target"]), int(a.get("hold_max_days") or 7)
    maker, taker, slip = costs["maker_pct"] / 100, costs["taker_pct"] / 100, costs["slippage_pct"] / 100
    market = a.get("entry_kind") == "open_next"
    fill_i, fill, fee_in = None, None, maker
    for d in days:
        if d >= vu.normalize():
            break
        if market:
            fill_i, fill, fee_in = d, float(O[d]) * (1 + slip), taker
            break
        if np.isfinite(L[d]) and L[d] < entry:
            fill_i, fill = d, min(float(O[d]), entry)
            break
    if fill_i is None:
        over = len(days) and days[-1] >= (vu.normalize() - pd.Timedelta(days=1))
        return {"state": "unfilled" if over else "pending"}
    sma5, ema10, rsi2 = C.rolling(5).mean(), C.ewm(span=10, adjust=False).mean(), None
    dyn, held = a.get("dyn_exit") or "none", 0
    for d in C.index[C.index >= fill_i]:
        o, hi, lo, cl = float(O[d]), float(H[d]), float(L[d]), float(C[d])
        if not np.isfinite(cl):
            continue
        held += 1
        res = None
        if d == fill_i and not market:
            if lo <= stop:
                res = (stop * (1 - slip), "stop", taker)
        else:
            if d > fill_i and o <= stop:
                res = (o * (1 - slip), "stop", taker)
            elif d > fill_i and o >= tgt:
                res = (o, "target", maker)
            elif lo <= stop:
                res = (stop * (1 - slip), "stop", taker)
            elif hi >= tgt:
                res = (tgt, "target", maker)
        if res is None:
            sig = {"sma5": cl > sma5[d], "prev_high": cl > float(H.shift(1)[d]) if np.isfinite(H.shift(1)[d]) else False,
                   "high7": cl >= float(C.rolling(7).max()[d]), "ema10_lost": d > fill_i and cl < ema10[d]}.get(dyn, False)
            if dyn == "rsi2_70":
                rsi2 = I.rsi(C.to_frame("x"), 2)["x"] if rsi2 is None else rsi2
                sig = rsi2[d] > 70
            if sig:
                res = (cl * (1 - slip), "signal", taker)
            elif held >= hold:
                res = (cl * (1 - slip), "time", taker)
        if res is not None:
            ret = res[0] * (1 - res[2]) / (fill * (1 + fee_in)) - 1
            return {"state": "closed", "ret": round(float(ret), 5), "win": bool(ret > 0), "reason": res[1], "days": held,
                    "fill_date": str(fill_i.date()), "exit_date": str(d.date()), "fill": round(float(fill), 10)}
    return {"state": "open", "fill_date": str(fill_i.date()), "days": held, "mark_pct": round(100 * (float(C.iloc[-1]) / fill - 1), 2)}


def agg(rows: list) -> dict:
    cl = [r for r in rows if r.get("state") == "closed"]
    n, wins = len(cl), sum(1 for r in cl if r["win"])
    filled = len([r for r in rows if r.get("state") in ("closed", "open")])
    decided = len([r for r in rows if r.get("state") in ("closed", "open", "unfilled")])
    return {"n": n, "wins": wins, "win_rate": round(wins / n, 3) if n else None, "avg_ret": round(float(np.mean([r["ret"] for r in cl])), 4) if n else None,
            "open": len([r for r in rows if r.get("state") == "open"]), "fill_rate": round(filled / decided, 3) if decided else None}


def apply_live(P: dict, adjust: dict | None) -> dict:
    """Blend the desk's own graded alerts into the win-odds table (so grading learns every day), then re-grade."""
    G, table = P["grading"], P.get("grade_table") or {}
    for key, c in ((adjust or {}).get("cells") or {}).items():
        sid, cond = key.split("|")
        cell = (table.get(sid) or {}).get(cond)
        if not cell or not c.get("n"):
            continue
        n_bt = min(cell.get("n", 0) + 20, 80)
        p = (cell["p"] * n_bt + LIVE_WEIGHT * c["wins"]) / (n_bt + LIVE_WEIGHT * c["n"])
        cell["p_backtest"], cell["p"], cell["live_n"], cell["live_wins"] = cell["p"], round(p, 4), c["n"], c["wins"]
        if cell["grade"] == "A+" and p < G["aplus_win"]:
            cell["grade"] = "A" if p >= G["watch_win"] else "B"
        elif cell["grade"] == "A" and p < G["watch_win"]:
            cell["grade"] = "B"
    return P


def review(data_dir, state_dir, out_dir) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, part, meta = D.load_daily(data_dir)
    ind = I.compute(w, D.load_fng(data_dir))
    state = load_state(state_dir)
    P = S.load_params(state["params"])
    specs = S.all_specs(P)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    prev = state["learning"].get("stats") or {}
    graded = dict(prev.get("alerts") or {})
    newly = []
    # every A+ alert ever sent, plus the watchlist setups scans listed (graded on paper too)
    cands = {a["id"]: a for a in state["alerts"].values() if a.get("type") == "entry" and a.get("valid_from")}
    for a in cands.values():
        old = graded.get(a["id"]) or {}
        if old.get("state") in ("closed", "unfilled", "no_data"):
            continue
        try:
            r = shadow(a, w, P["costs"])
        except Exception as e:
            r = {"state": "error", "error": str(e)[:100]}
        rec = {"ticker": a["ticker"], "strategy": a["strategy"], "condition": a.get("condition"), "grade": a.get("grade"),
               "signal_date": a.get("signal_date"), "score": a.get("score"), "win_pct": a.get("win_pct"), "featured": not a.get("also_qualified"),
               "user": a.get("status"), "news": (a.get("news") or {}).get("verdict"), **r}
        if r.get("state") in ("closed", "unfilled") and old.get("state") != r.get("state"):
            newly.append({"id": a["id"], **rec})
        graded[a["id"]] = rec
    rows = list(graded.values())
    feat = [r for r in rows if r.get("featured", True)]
    by = lambda key, src: {k: agg([r for r in src if r.get(key) == k]) for k in sorted({str(r.get(key)) for r in src if r.get(key)})}
    cells = {}
    for r in rows:
        if r.get("state") == "closed" and r.get("condition"):
            c = cells.setdefault(f"{r['strategy']}|{r['condition']}", {"n": 0, "wins": 0})
            c["n"] += 1
            c["wins"] += int(r["win"])
    # the real account
    closed = closed_trades(state)
    live = {"n": len(closed), "wins": sum(1 for t in closed if float(t.get("pnl") or 0) > 0), "pnl": round(sum(float(t.get("pnl") or 0) for t in closed), 2)}
    live["win_rate"] = round(live["wins"] / live["n"], 3) if live["n"] else None
    live_by = {}
    for t in closed:
        d = live_by.setdefault(t.get("strategy") or "Manual", {"n": 0, "wins": 0, "pnl": 0.0})
        d["n"] += 1
        d["wins"] += int(float(t.get("pnl") or 0) > 0)
        d["pnl"] = round(d["pnl"] + float(t.get("pnl") or 0), 2)
    # pauses: a strategy whose recent graded alerts stop winning sits out until it recovers or the weekly re-test clears it
    old_adj = state["learning"].get("adjust") or {}
    paused, changes = dict(old_adj.get("paused") or {}), []
    for sid in {r["strategy"] for r in rows}:
        cl = sorted([r for r in rows if r["strategy"] == sid and r.get("state") == "closed"], key=lambda r: str(r.get("exit_date")))[-PAUSE["n"]:]
        if len(cl) < PAUSE["n"]:
            continue
        wr = sum(1 for r in cl if r["win"]) / len(cl)
        label = specs.get(sid, {}).get("label", sid)
        if sid not in paused and wr < PAUSE["below"]:
            paused[sid] = f"its last {len(cl)} graded alerts won {100 * wr:.0f}% (paused {now.date()})"
            changes.append(f"Paused {label}: last {len(cl)} graded alerts won {100 * wr:.0f}%.")
        elif sid in paused and wr >= PAUSE["restore"]:
            paused.pop(sid)
            changes.append(f"Resumed {label}: last {len(cl)} graded alerts won {100 * wr:.0f}%.")
    mk = ind["market"]
    cond_now = I.condition_series(mk).iloc[-1]
    cs = I.condition_series(mk).dropna()
    run_len = int((cs[::-1] != cond_now).values.argmax()) if (cs != cond_now).any() else len(cs)
    table = P.get("grade_table") or {}
    active_here = [f"{specs[s]['label']} ({100 * c[cond_now]['p']:.0f}%, {c[cond_now]['grade']})" for s, c in table.items()
                   if s in specs and cond_now in c and P["strategies"].get(s, {}).get("enabled") and s not in paused
                   and (not P["strategies"][s].get("market_conditions") or cond_now in P["strategies"][s]["market_conditions"])]
    insights = []
    a_all = agg(feat)
    if a_all["n"]:
        insights.append(f"A+ alerts graded so far: {a_all['wins']} of {a_all['n']} won ({100 * a_all['win_rate']:.0f}%), average {100 * a_all['avg_ret']:+.1f}% after fees; {100 * (a_all['fill_rate'] or 0):.0f}% of orders filled.")
    else:
        insights.append("No A+ alert has finished yet, so the record starts from the backtest. Every alert is graded here once it closes or expires.")
    for k, v in by("strategy", rows).items():
        if v["n"] >= 5:
            insights.append(f"{specs.get(k, {}).get('label', k)}: {v['wins']}/{v['n']} graded alerts won ({100 * v['win_rate']:.0f}%).")
    insights.append(f"Market condition today: {I.CONDITION_LABEL.get(cond_now, cond_now)} (day {run_len} of this stretch). " +
                    ("Strategies that trade here: " + "; ".join(active_here) + "." if active_here else "No strategy has an A+ record in this condition yet: that is the first research target."))
    fund, gov = funds(state), None
    gov = risk_governor(state, fund["capital"])
    jid = now.strftime("%Y-%m-%d")
    journal = {"id": jid, "date": jid, "created_at": now_iso(), "market_condition": cond_now, "market_label": I.CONDITION_LABEL.get(cond_now),
               "market": {"btc": float(mk["btc"].iloc[-1]), "btc_vs_200d_pct": round(100 * float(mk["btc_dist200"].iloc[-1]), 1), "btc_7d_pct": round(100 * float(mk["btc_ret7"].iloc[-1]), 1),
                          "breadth50_pct": round(100 * float(mk["breadth50"].iloc[-1])), "fear_greed": None if not np.isfinite(mk["fng"].iloc[-1]) else float(mk["fng"].iloc[-1])},
               "graded_today": newly, "shadow": {"featured": a_all, "all": agg(rows)}, "live": live, "by_strategy": by("strategy", rows),
               "by_condition": by("condition", rows), "changes": changes, "insights": insights, "active_here": active_here,
               "account": {"capital": fund["capital"], "cash": fund["cash"], "open": fund["open_count"], "risk": gov["reason"]},
               "notes": "", "research": [], "lessons": []}
    stats = {"id": "stats", "updated_at": now_iso(), "alerts": dict(sorted(graded.items())[-600:]), "featured": a_all, "all": agg(rows),
             "by_strategy": by("strategy", rows), "by_condition": by("condition", rows), "by_grade": by("grade", rows), "cells": cells,
             "live": live, "live_by_strategy": live_by}
    adjust = {"id": "adjust", "updated_at": now_iso(), "paused": paused, "cells": cells}
    governor = {"id": "governor", "updated_at": now_iso(), **gov}
    writes = []
    for coll, did, doc in (("journal", jid, journal), ("learning", "stats", stats), ("learning", "adjust", adjust), ("learning", "governor", governor)):
        p = out / "docs" / coll / f"{did}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(clean(doc), default=str))
        writes.append({"op": "set", "collection": coll, "doc_id": did, "file_path": str(p.resolve())})
    (out / "writes.json").write_text(json.dumps(writes, indent=1))
    (out / "journal.json").write_text(json.dumps(clean(journal), indent=1, default=str))
    return {"journal": jid, "graded_today": len(newly), "record": a_all, "changes": changes, "condition": cond_now, "insights": insights}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["review"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(clean(review(a.data, a.state, a.out)), indent=1))


if __name__ == "__main__":
    main()
