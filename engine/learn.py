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
    vf, vu = D.naive_ts(a.get("valid_from")), D.naive_ts(a.get("valid_until"))
    if vf is None or vu is None:
        return {"state": "no_data"}
    O, H, L, C = (w[k][tk] for k in ("open", "high", "low", "close"))
    days = C.index[(C.index >= vf.normalize())]
    entry, stop, tgt, hold = float(a["entry"]), float(a["stop"]), float(a["target"]), int(a.get("hold_max_days") or 7)
    wide = a.get("tier") == "wide"                 # thinner coins: same harsher fills and slippage as the backtest
    maker, taker = costs["maker_pct"] / 100, costs["taker_pct"] / 100
    slip = (costs.get("wide_slippage_pct", 0.35) if wide else costs["slippage_pct"]) / 100
    margin = costs.get("wide_fill_margin_pct", 0.30) / 100 if wide else 0.0
    market = a.get("entry_kind") == "open_next"
    fill_i, fill, fee_in = None, None, maker
    for d in days:
        if d >= vu.normalize():
            break
        if market:
            fill_i, fill, fee_in = d, float(O[d]) * (1 + slip), taker
            break
        if np.isfinite(L[d]) and L[d] < entry * (1 - margin):
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
    """Blend the desk's own graded alerts into the win-odds tables (so grading learns every day), then re-grade.
    Cells are keyed strategy|condition|tier; an old two-part key means the core tier."""
    G = P["grading"]
    tables = {"core": P.get("grade_table") or {}, "wide": P.get("grade_table_wide") or {}}
    for key, c in ((adjust or {}).get("cells") or {}).items():
        parts = key.split("|")
        sid, cond, tier = parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else "core"
        cell = (tables.get(tier, {}).get(sid) or {}).get(cond)
        if not cell or not c.get("n") or cell.get("p") is None:
            continue
        n_bt = min(cell.get("n", 0) + 20, 80)
        p = (cell["p"] * n_bt + LIVE_WEIGHT * c["wins"]) / (n_bt + LIVE_WEIGHT * c["n"])
        cell["p_backtest"], cell["p"], cell["live_n"], cell["live_wins"] = cell["p"], round(p, 4), c["n"], c["wins"]
        if cell["grade"] == "A+" and p < G["aplus_win"]:
            cell["grade"] = "A" if p >= G["watch_win"] else "B"
        elif cell["grade"] == "A" and p < G["watch_win"]:
            cell["grade"] = "B"
    return P


def week_start(t: pd.Timestamp) -> pd.Timestamp:
    t = t.normalize()
    return t - pd.Timedelta(days=t.weekday())


def pace_stats(alerts: dict, now: pd.Timestamp, min_week: int, weeks: int = 8) -> dict:
    """A+ setups served per week (Monday to Sunday, UTC, by the day the order was live)."""
    wk_now = week_start(now)
    counts = {}
    for a in alerts.values():
        if a.get("type") != "entry" or a.get("grade") != "A+" or a.get("topup"):
            continue
        vf = D.naive_ts(a.get("valid_from"))
        if vf is not None:
            counts[week_start(vf)] = counts.get(week_start(vf), 0) + 1
    first = min(counts) if counts else wk_now
    hist = [{"week": str((wk_now - pd.Timedelta(weeks=i)).date()), "served": counts.get(wk_now - pd.Timedelta(weeks=i), 0)}
            for i in range(weeks) if wk_now - pd.Timedelta(weeks=i) >= first]
    done = hist[1:]                                              # finished weeks only
    pace = min(min_week, now.weekday() + 1)
    return {"min_week": min_week, "this_week": counts.get(wk_now, 0), "pace": pace, "behind": counts.get(wk_now, 0) < pace,
            "weeks": len(done), "short_weeks": len([h for h in done if h["served"] < min_week]), "history": hist}


def run_health(state: dict, meta: dict, health: dict, now: pd.Timestamp) -> dict:
    """Did the desk's own machinery run? Scans expected at 01:20 (daily close), 03:20, 13:20 and 19:20 UTC; the feed should be minutes old."""
    scans = sorted([s for s in state.get("scans", {}).values() if s.get("started_at")], key=lambda s: s["started_at"])
    since = now - pd.Timedelta(hours=26)
    recent = [s for s in scans if (D.naive_ts(s["started_at"]) or pd.Timestamp(0)) >= since]
    slots, missed = [], []
    for back in range(0, 2):
        for hh in (1, 3, 13, 19):
            t = (now.normalize() - pd.Timedelta(days=back)) + pd.Timedelta(hours=hh, minutes=20)
            if since <= t <= now - pd.Timedelta(minutes=45) and scans and t >= (D.naive_ts(scans[0]["started_at"]) or t):
                slots.append(t)                                   # nothing was due before the desk's first scan
    for t in slots:
        if not any(abs(((D.naive_ts(s["started_at"]) or pd.Timestamp(0)) - t).total_seconds()) <= 3 * 3600 for s in recent):
            missed.append(t.strftime("%Y-%m-%d %H:%M UTC"))
    problems = [f"{s['id']}: {'; '.join((s.get('errors') or []) + (s.get('run_issues') or []))[:200]}" for s in recent if s.get("errors") or s.get("run_issues")]
    stale = [s["id"] for s in recent if (s.get("data") or {}).get("status") not in (None, "ok")]
    ok = not missed and not problems and not stale and health["status"] == "ok"
    return {"ok": ok, "scans_26h": len(recent), "expected": len(slots), "missed": missed, "problems": problems[:6], "stale_scans": stale,
            "feed_status": health["status"], "feed_age_min": health.get("feed_age_min"), "feed_notes": health["notes"],
            "last_scan": scans[-1]["started_at"] if scans else None}


def review(data_dir, state_dir, out_dir) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, part, meta = D.load_daily(data_dir)
    state = load_state(state_dir)
    P = S.load_params(state["params"])
    min_dv = float((P.get("universe") or {}).get("min_dv") or I.MIN_DV)
    ind = I.compute(w, D.load_fng(data_dir), min_dv=min_dv)
    specs = S.all_specs(P)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    health = D.health(w, meta, now)
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
               "signal_date": a.get("signal_date"), "score": a.get("score"), "win_pct": a.get("win_pct"), "featured": not a.get("topup") and a.get("grade", "A+") == "A+", "funded": a.get("funded", not a.get("also_qualified")),
               "tier": a.get("tier") or "core", "topup": bool(a.get("topup")),
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
            c = cells.setdefault(f"{r['strategy']}|{r['condition']}|{r.get('tier') or 'core'}", {"n": 0, "wins": 0})
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
    table, table_w = P.get("grade_table") or {}, P.get("grade_table_wide") or {}
    live_here = lambda s: (s in specs and P["strategies"].get(s, {}).get("enabled") and s not in paused
                           and (not P["strategies"][s].get("market_conditions") or cond_now in P["strategies"][s]["market_conditions"]))
    aplus_here = [s for s in specs if live_here(s) and "A+" in (((table.get(s) or {}).get(cond_now) or {}).get("grade"), ((table_w.get(s) or {}).get(cond_now) or {}).get("grade"))]
    tier_txt = lambda s: ", ".join(f"{nm} {100 * c['p']:.0f}%" for nm, c in (("core", (table.get(s) or {}).get(cond_now) or {}), ("wider list", (table_w.get(s) or {}).get(cond_now) or {})) if c.get("grade") == "A+")
    active_here = [f"{specs[s]['label']} ({tier_txt(s)})" for s in aplus_here]
    # the coin list: what is scanned today, what joined and what dropped out in the last week
    liq = ind["liquid"]
    now_set, old_set = set(liq.columns[liq.iloc[-1].values]), set(liq.columns[liq.iloc[-8].values]) if len(liq) > 8 else set()
    core_set = set(ind["is_core"].columns[ind["is_core"].iloc[-1].values]) & now_set
    uni = D.load_universe(data_dir)
    online = {tk for tk, u in uni.items() if u.get("status") == "online" and not u.get("peg")} if uni else now_set
    just_below = sorted([(tk, float(ind["dv30"][tk].iloc[-1])) for tk in online - now_set if tk in ind["dv30"] and np.isfinite(ind["dv30"][tk].iloc[-1]) and ind["dv30"][tk].iloc[-1] >= 0.5 * min_dv], key=lambda x: -x[1])
    universe = {"scanned": len(now_set & online), "core": len(core_set & online), "wide": len((now_set - core_set) & online), "listed": len(online), "min_dollar_vol": min_dv,
                "joined_7d": sorted(now_set - old_set)[:30], "left_7d": sorted(old_set - now_set)[:30], "just_below": [t for t, _ in just_below[:15]]}
    pace = pace_stats(state["alerts"], now, int((P.get("pace") or {}).get("min_week", 5)))
    runs = run_health(state, meta, health, now)
    insights = []
    a_all = agg(feat)
    if a_all["n"]:
        insights.append(f"A+ alerts graded so far: {a_all['wins']} of {a_all['n']} won ({100 * a_all['win_rate']:.0f}%), average {100 * a_all['avg_ret']:+.1f}% after fees; {100 * (a_all['fill_rate'] or 0):.0f}% of orders filled.")
    else:
        insights.append("No A+ alert has finished yet, so the record starts from the backtest. Every alert is graded here once it closes or expires.")
    for k, v in by("strategy", rows).items():
        if v["n"] >= 5:
            insights.append(f"{specs.get(k, {}).get('label', k)}: {v['wins']}/{v['n']} graded alerts won ({100 * v['win_rate']:.0f}%).")
    regime = I.regime_now(ind)
    cover = I.alert_cover(P, regime["condition"], regime["zone"])
    can_now = [f"{specs.get(s_, {}).get('label', s_)} ({', '.join(('core' if t_ == 'core' else 'wider list') + f' {100 * p_:.0f}%' for t_, p_ in v.items())})" for s_, v in sorted(cover["can_alert"].items())]
    benched = sorted(specs.get(s_, {}).get("label", s_) for s_ in cover["benched_by_zone"])
    insights.append(f"Market today: {regime['label']} (day {run_len} of this stretch), {regime['zone_label']} (day {regime['days_in_zone']} in that zone). "
                    + ("Able to send an A+ alert in exactly this market: " + "; ".join(can_now) + "." if can_now else "Nothing in the live book can send an A+ alert in exactly this market: that is today's first research target.")
                    + (" A+ for this condition but benched by the danger zone: " + ", ".join(benched) + "." if benched else ""))
    insights.append(f"Coin list: {universe['scanned']} Coinbase coins scanned today ({universe['core']} core, {universe['wide']} wider list) of {universe['listed']} listed"
                    + (f"; joined this week: {', '.join(universe['joined_7d'][:10])}" if universe["joined_7d"] else "") + (f"; dropped out: {', '.join(universe['left_7d'][:10])}" if universe["left_7d"] else "") + ".")
    insights.append(f"Weekly pace: {pace['this_week']} A+ setup{'s' if pace['this_week'] != 1 else ''} served this week, pace {pace['pace']} by today, minimum {pace['min_week']}."
                    + (f" {pace['short_weeks']} of the last {pace['weeks']} finished weeks fell short." if pace["weeks"] else ""))
    # what could be better: written every day so nothing that needs attention stays quiet
    opps = []
    if not runs["ok"]:
        if runs["missed"]:
            opps.append({"area": "Reliability", "text": f"Scheduled scans did not run at: {', '.join(runs['missed'])}. Check the scheduled task before trusting 'no alerts'."})
        if runs["feed_status"] != "ok":
            opps.append({"area": "Reliability", "text": "The price feed is behind: " + " ".join(runs["feed_notes"])})
        for pr in runs["problems"]:
            opps.append({"area": "Reliability", "text": "A scan reported a problem: " + pr})
    if not aplus_here:
        opps.append({"area": "Coverage", "text": f"No strategy holds an A+ record in today's market ({I.CONDITION_LABEL.get(cond_now, cond_now)}). Today's research goes there first."})
    elif not can_now:
        opps.append({"area": "Coverage", "text": f"No strategy can send an A+ alert in today's market ({regime['label']}, {regime['zone_label']}): the ones with an A+ record for the condition are benched by the danger zone. Today's research goes to this market first; a silent book is right only where the lessons show no edge."})
    if pace["behind"]:
        opps.append({"area": "Pace", "text": f"{pace['this_week']} A+ setups this week against a pace of {pace['pace']}. Lower grades lost money in testing, so the gap is closed by new A+ strategies, not by loosening the bar."})
    for key, c in cells.items():
        parts = key.split("|")
        cell = (((table_w if (len(parts) > 2 and parts[2] == "wide") else table).get(parts[0]) or {}).get(parts[1]) or {})
        if c["n"] >= 6 and cell.get("p") and c["wins"] / c["n"] < cell["p"] - 0.2:
            opps.append({"area": "Strategy", "text": f"{specs.get(parts[0], {}).get('label', parts[0])} in {I.CONDITION_LABEL.get(parts[1], parts[1])}: live alerts won {c['wins']}/{c['n']} against {100 * cell['p']:.0f}% expected. Re-test it."})
    if universe["just_below"]:
        opps.append({"area": "Coin list", "text": "Close to joining the scan list (30-day trading just under the floor): " + ", ".join(universe["just_below"][:8]) + "."})
    fill = a_all.get("fill_rate")
    if fill is not None and a_all["n"] + a_all["open"] >= 8 and fill < 0.4:
        opps.append({"area": "Execution", "text": f"Only {100 * fill:.0f}% of A+ limit orders filled. Research: test a shallower entry for the strategies that miss most."})
    fund, gov = funds(state), None
    gov = risk_governor(state, fund["capital"])
    day = now.normalize() - pd.Timedelta(days=1)       # the UTC trading day that just finished: the day this review is about
    jid = day.strftime("%Y-%m-%d")
    journal = {"id": jid, "date": jid, "created_at": now_iso(), "market_condition": cond_now, "market_label": I.CONDITION_LABEL.get(cond_now),
               "market": {"btc": float(mk["btc"].iloc[-1]), "btc_vs_200d_pct": round(100 * float(mk["btc_dist200"].iloc[-1]), 1), "btc_7d_pct": round(100 * float(mk["btc_ret7"].iloc[-1]), 1),
                          "breadth50_pct": round(100 * float(mk["breadth50"].iloc[-1])), "fear_greed": None if not np.isfinite(mk["fng"].iloc[-1]) else float(mk["fng"].iloc[-1])},
               "graded_today": newly, "shadow": {"featured": a_all, "all": agg(rows)}, "live": live, "by_strategy": by("strategy", rows),
               "by_condition": by("condition", rows), "changes": changes, "insights": insights, "active_here": active_here, "regime": regime, "can_alert_now": can_now, "benched_by_zone": benched,
               "all_weather": [specs.get(s_, {}).get("label", s_) for s_ in cover["all_weather"]], "aplus_strategies_per_condition": cover["aplus_strategies_per_condition"],
               "bar_date": str(w["close"].index[-1].date()), "run_at": now_iso(), "opportunities": opps, "pace": pace, "universe": universe, "runs": runs,
               "account": {"capital": fund["capital"], "cash": fund["cash"], "open": fund["open_count"], "risk": gov["reason"]},
               "notes": "", "research": [], "lessons": []}
    stats = {"id": "stats", "updated_at": now_iso(), "alerts": dict(sorted(graded.items())[-600:]), "featured": a_all, "all": agg(rows),
             "by_strategy": by("strategy", rows), "by_condition": by("condition", rows), "by_grade": by("grade", rows), "cells": cells,
             "by_tier": by("tier", rows), "pace": pace, "universe": universe, "runs": runs, "live": live, "live_by_strategy": live_by}
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
    return {"journal": jid, "graded_today": len(newly), "record": a_all, "changes": changes, "condition": cond_now, "insights": insights,
            "opportunities": opps, "pace": {k: pace[k] for k in ("this_week", "pace", "min_week", "behind")}, "runs_ok": runs["ok"]}


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
