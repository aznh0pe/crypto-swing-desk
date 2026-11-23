"""Strategy research: test new ideas honestly, find the market and coin states where a trade wins,
and promote what passes. Ideas are JSON rule specs (see rules.py), so none of this needs code changes.

  python -m engine.research next    --state S --out O               what to test next + everything already known
  python -m engine.research test    --spec F --data D --state S --out O
  python -m engine.research states  --spec F --data D --out O      where does this simple trade win? (design aid)
  python -m engine.research promote --data D --state S --out O     put passing candidates into the live book
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from . import backtest as bt
from . import calibrate as C
from . import indicators as I
from . import rules, strategies as S
from .portfolio import load_state

HERE = pathlib.Path(__file__).resolve().parent
STATE_FEATURES = ["rsi2", "rsi14", "atr_pct", "dist20", "dist50", "dist200", "dd90", "rs_pct", "rs30_pct", "vol_ratio", "ret7", "ret30",
                  "adx14", "btc_dist200", "btc_dist50", "btc_ret7", "btc_ret30", "btc_vol_ratio", "btc_rsi14", "breadth50", "breadth200", "alt_ret7", "fng",
                  "washed", "flush7", "up7", "dv30"]


def seeds() -> list:
    out = []
    for f in sorted(HERE.glob("research_seed*.json")):
        out += json.loads(f.read_text())
    return out


def cmd_next(state_dir, out_dir, data_dir=None) -> dict:
    st = load_state(state_dir)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tested = {k: v for k, v in st["research"].items()}
    P = S.load_params(st["params"])
    known = set(tested) | set(S.all_specs(P))
    todo = [s for s in seeds() if s["id"] not in known]
    seed = todo[0] if todo else None
    if seed:
        (out / "seed_spec.json").write_text(json.dumps(seed, indent=1))
    table, live_conds = P.get("grade_table") or {}, {}
    for c in I.CONDITIONS:
        live_conds[c] = [s for s, t in table.items() if (t.get(c) or {}).get("grade") == "A+" and P["strategies"].get(s, {}).get("enabled")
                         and (not P["strategies"][s].get("market_conditions") or c in P["strategies"][s]["market_conditions"])]
    thin = [c for c in I.CONDITIONS if not live_conds[c]]
    fam = {}
    for v in list(tested.values()) + list(S.builtin_specs().values()):
        f = v.get("family") or (v.get("spec") or {}).get("family") or "Other"
        fam[f] = fam.get(f, 0) + 1
    # Two standing priorities, in this order: (1) the market as it is today, (2) strategies that hold in every condition.
    lab = lambda sid: (S.all_specs(P).get(sid) or {}).get("label", sid)
    market_now, cover = None, None
    focus = ""
    if data_dir:
        ind, _, _, _ = C.load_ind(data_dir, P)
        market_now = I.regime_now(ind)
        cover = I.alert_cover(P, market_now["condition"], market_now["zone"])
        can = cover["can_alert"]
        focus += (f"PRIORITY 1, the market today ({market_now['bar_date']} close): {market_now['label']}, day {market_now['day_of_stretch']} of this stretch"
                  + (f" (before it: {market_now['previous_condition']})" if market_now.get("previous_condition") else "")
                  + f"; {market_now['zone_label']} (day {market_now['days_in_zone']} in that zone); Bitcoin {100 * (market_now['btc_7d'] or 0):+.1f}% in 7 days, "
                  f"{100 * (market_now['breadth_above_50d'] or 0):.0f}% of core coins above their 50-day, fear and greed {market_now['fear_greed']:.0f}. "
                  + ("Able to send an A+ alert in exactly this market: " + "; ".join(f"{lab(s_)} ({', '.join(f'{t_} {100 * p_:.0f}%' for t_, p_ in v.items())})" for s_, v in sorted(can.items())) + ". "
                     if can else "NOTHING in the live book can send an A+ alert in exactly this market. ")
                  + ("A+ for this condition but benched by the danger zone (no record of their own in it): " + ", ".join(sorted(lab(s_) for s_ in cover["benched_by_zone"])) + ". " if cover["benched_by_zone"] else "")
                  + "Today's first own design is written for this market: its condition, its zone and where the stretch stands (first days, late, just ended). "
                    "Judge it on its record in this market in the training years and on unseen data, win rate first, average trade second. "
                    "If the lessons already show this exact state has no edge, say so in the journal and test the nearest state that has not been tried; never lower the bar to fill a quiet day. ")
        aw = cover["all_weather"]
        focus += ("PRIORITY 2, every market condition: "
                  + (f"{len(aw)} live strategies grade A+ in three or more of the four conditions ({', '.join(lab(s_) for s_ in aw)}). " if aw else "no live strategy grades A+ in three or more of the four conditions; every one is a specialist. ")
                  + "A+ strategies per condition (core list): " + ", ".join(f"{k} {v}" for k, v in cover["aplus_strategies_per_condition"].items()) + ". "
                  "Today's second own design aims at a trade that wins in at least three of the four conditions in both periods (the test's \"fit\" block counts them); "
                  "an idea that only wins in one condition is a specialist and is judged under priority 1 or on the thinnest condition. ")
    focus += ("No A+ strategy trades in: " + ", ".join(I.CONDITION_LABEL[c] for c in thin) + ". Design for those conditions first." if thin else
              "Every market condition has at least one A+ strategy on paper. Raise the win rate: look for a filter or a new trigger that beats the weakest A+ cell.")
    pace = (st["learning"].get("stats") or {}).get("pace") or {}
    if pace:
        focus += (f" Weekly pace: {pace.get('this_week', 0)} A+ setups sent this week against a pace of {pace.get('pace')} by today (minimum {pace.get('min_week')} a week); "
                  f"{pace.get('short_weeks', 0)} of the last {pace.get('weeks', 4)} weeks fell short." + (" The desk needs more genuine A+ setups in quiet weeks: prefer triggers that fire on days when the live book is silent." if pace.get("short_weeks") else ""))
    zs = P.get("zones") or {}
    if zs.get("blocked"):
        focus += (" Danger zones (where the A+ book as a whole loses, so only strategies with their own A+ record there are sent): "
                  + "; ".join(f"{I.ZONE_LABEL[z]} (open there: {', '.join(sorted((zs.get('exempt') or {}).get(z) or {})) or 'nothing yet'})" for z in zs["blocked"])
                  + ". A strategy written for such a zone (put the zone in its conditions with btc_dist200) earns its place automatically at the next calibration if it clears the A+ bar there on 40+ trades over 15+ days in both periods; "
                    "on 2026-11-17 some 170 entry, exit and coin-state combinations were screened for the 10-35% zone and none won 60% in both periods except deep market-wide flushes on core coins, so look for a different kind of trigger, and test core and wider-list coins apart. ")
    focus += (" What is already known (see lessons): dip-buying only clears the bar in stressed markets or when Bitcoin is surging; plain dips, range lows, deep resting bids, streaks and rebound-confirmation entries all failed in calm bull and quiet bear markets at 0.5%/0.9% fees. "
              "New quiet-market ideas should use a different trigger family or a much larger edge per trade; lower fees also move calm-bull dips over the bar.")
    res = {"next_seed": seed["id"] if seed else None, "spec_file": str((out / "seed_spec.json").resolve()) if seed else None,
           "backlog_left": [s["id"] for s in todo[1:]], "tested": [{"id": k, "label": v.get("label") or (v.get("spec") or {}).get("label"), "family": v.get("family") or (v.get("spec") or {}).get("family"),
                                                                     "verdict": v.get("verdict"), "summary": v.get("summary")} for k, v in sorted(tested.items())],
           "families_covered": fam, "a_plus_by_condition": live_conds, "market_now": market_now, "cover": cover, "focus": focus, "fields": sorted(rules.FRAME_NAMES), "market_fields": sorted(rules.MARKET_NAMES),
           "lessons": (st["learning"].get("lessons") or {}).get("items", [])}
    (out / "next.json").write_text(json.dumps(res, indent=1))
    return res


def overlap(tr: pd.DataFrame, live: pd.DataFrame) -> float:
    """Share of this idea's filled trades that a live strategy already takes (same coin, signal within 2 days)."""
    t = bt.done(tr)
    if t.empty or live.empty:
        return 0.0
    keys = set()
    for tk, d in zip(live["ticker"], live["signal_date"]):
        for k in range(-2, 3):
            keys.add((tk, d + pd.Timedelta(days=k)))
    return float(np.mean([(tk, d) in keys for tk, d in zip(t["ticker"], t["signal_date"])]))


def fit_report(tr: pd.DataFrame, ind: dict, P: dict) -> dict:
    """How an idea fits the desk's two standing priorities, on core-list trades: its record in the market as it is
    today (same condition, and same condition and Bitcoin zone), and in how many of the four market conditions it wins
    in both periods. A condition counts as won with 58%+ wins and a +0.4% average trade on 15+ trades in the training
    years and 8+ on unseen data."""
    now = I.regime_now(ind)
    G, cut = P["grading"], pd.Timestamp(P["split"])
    t = bt.done(C.tier_of(tr, "core")) if tr is not None and len(tr) else pd.DataFrame(columns=["ret", "signal_date", "condition"])
    if len(t):
        t = t.assign(zone=t["signal_date"].map(ind["market"]["btc_dist200"]).map(I.zone_of))
    def st(x):
        return {"trades": int(len(x)), "days": int(x["signal_date"].nunique()) if len(x) else 0, "win_rate": round(float((x["ret"] > 0).mean()), 3) if len(x) else None,
                "avg_ret": round(float(x["ret"].mean()), 4) if len(x) else None}
    def both(x):
        return {"train": st(x[x["signal_date"] < cut]), "unseen": st(x[x["signal_date"] >= cut])}
    won = lambda b: bool(b["train"]["trades"] >= 15 and b["unseen"]["trades"] >= 8 and min(b["train"]["win_rate"], b["unseen"]["win_rate"]) >= G["watch_win"]
                         and min(b["train"]["avg_ret"], b["unseen"]["avg_ret"]) > 0.004)
    per = {c: both(t[t["condition"] == c]) if len(t) else both(t) for c in I.CONDITIONS}
    wins = [c for c in I.CONDITIONS if won(per[c])]
    cur = per[now["condition"]]
    curz = both(t[(t["condition"] == now["condition"]) & (t["zone"] == now["zone"])]) if len(t) else both(t)
    kind = ("all-weather" if len(wins) >= 3 else "current market" if won(curz) or (won(cur) and now["zone"] == "bull") else "specialist" if wins else "none")
    pc = lambda b: (f"{100 * b['train']['win_rate']:.0f}% of {b['train']['trades']} in training, " if b["train"]["trades"] else "no training trades, ") + (f"{100 * b['unseen']['win_rate']:.0f}% of {b['unseen']['trades']} unseen" if b["unseen"]["trades"] else "no unseen trades")
    text = (f"Fit: in today's market ({now['label']}, {now['zone_label']}) {pc(curz)}; in {now['label']} overall {pc(cur)}. "
            f"Wins in {len(wins)} of 4 market conditions in both periods" + (f" ({', '.join(I.CONDITION_LABEL[c] for c in wins)})" if wins else "") + f": {kind}.")
    return {"kind": kind, "market_now": {k: now[k] for k in ("bar_date", "condition", "label", "zone", "zone_label")}, "current_condition": cur, "current_condition_and_zone": curz,
            "by_condition": per, "conditions_won": wins, "wins_current_market": bool(won(curz)), "text": text}


def cmd_test(spec_file, data_dir, state_dir, out_dir) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(pathlib.Path(spec_file).read_text())
    errs = rules.validate(spec)
    st = load_state(state_dir)
    P = S.load_params(st["params"])
    sig = rules.signature(spec)
    known = {**{k: (v.get("signature") or (rules.signature(v["spec"]) if v.get("spec") else None)) for k, v in st["research"].items()},
             **{k: rules.signature(v) for k, v in S.all_specs(P).items()}}
    for k, v in known.items():
        if v and k != spec.get("id") and v == sig:
            errs.append(f"same rules as {k}: change the idea, not the name")
    if spec.get("id") in st["research"]:
        errs.append(f"{spec.get('id')} is already in the research log: give a variant its own id")
    if spec.get("id") in S.builtin_specs():
        errs.append("that id belongs to a built-in strategy")
    if errs:
        res = {"ok": False, "errors": errs}
        print(json.dumps(res, indent=1))
        return res
    ind, w, part, meta = C.load_ind(data_dir, P)
    b = I.base_masks(ind)
    r = C.evaluate(spec, ind, P, b)
    tr = r.pop("_trades")
    frames = []
    for sid, plan in S.all_plans(ind, P).items():
        try:
            frames.append(bt.done(bt.simulate(plan, ind, P["costs"], manage=P.get("manage")))[["ticker", "signal_date"]])
        except Exception:
            pass
    ov = overlap(tr, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    tr_n, te = r["train"], r["oos"]
    if r["enabled"] and ov < 0.7:
        verdict = "candidate"
    elif r["enabled"]:
        verdict = "redundant"
    elif (tr_n.get("win_rate") or 0) >= 0.55 and (tr_n.get("profit_factor") or 0) >= 1.15 and (te.get("profit_factor") or 0) >= 1.0:
        verdict = "watch"
    else:
        verdict = "rejected"
    fit = fit_report(tr, ind, P)
    f = lambda s: f"{100 * s['win_rate']:.0f}% of {s['trades']} (PF {s.get('profit_factor')}, avg {100 * s['avg_ret']:+.1f}%)" if s.get("trades") else "no trades"
    wd = r.get("wide") or {}
    summary = (f"Training {f(tr_n)}; unseen {f(te)}; " + (f"wider list {f(wd)}; " if wd.get("trades") else "")
               + f"overlap with live strategies {100 * ov:.0f}%. {r['reason']} {fit['text']}")
    rid = spec["id"]
    doc = {"id": rid, "spec": spec, "signature": sig, "about": spec.get("about"), "verdict": verdict, "summary": summary, "tested_at": C.now_iso(), "family": spec.get("family"),
           "label": spec.get("label"), "params": r["params"], "market_conditions": r["market_conditions"], "mode": r["mode"],
           "fit": fit, "source": spec.get("source"), "own_design": "own design" in str((spec.get("source") or {}).get("title", "")).lower()}
    result = {"id": f"{rid}-{C.now_iso()[:10]}", "idea": rid, "tested_at": C.now_iso(), "verdict": verdict, "overlap": round(ov, 3),
              **{k: r[k] for k in ("label", "family", "params", "enabled", "mode", "market_conditions", "reason", "checks", "train", "oos", "by_condition", "yearly", "rules")}}
    C.write_docs_coll(out, [("research", rid, doc), ("research_results", result["id"], result)])
    res = {"ok": True, "id": rid, "verdict": verdict, "summary": summary, "fit": fit, "by_condition": r["by_condition"], "params": r["params"], "yearly": r["yearly"]}
    (out / "result.json").write_text(json.dumps(C.clean(res), indent=1))
    print(json.dumps(C.clean(res), indent=1))
    return res


def cmd_states(spec_file, data_dir, out_dir) -> dict:
    """Bucket a simple trade's results by market and coin state, training and unseen side by side."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(pathlib.Path(spec_file).read_text())
    errs = rules.validate(spec)
    if errs:
        print(json.dumps({"ok": False, "errors": errs}))
        return {"ok": False}
    P = S.load_params(None)
    ind, *_ = C.load_ind(data_dir, P)
    plan = rules.build_plan(spec, ind, P, I.base_masks(ind), name=spec["id"])
    t = bt.done(bt.simulate(plan, ind, P["costs"], manage=P.get("manage")))
    if t.empty:
        print(json.dumps({"ok": True, "trades": 0}))
        return {"ok": True, "trades": 0}
    cut = pd.Timestamp(P["split"])
    a, o = t[t["signal_date"] < cut], t[t["signal_date"] >= cut]
    rows = []

    def add(name, rule, ma, mo):
        xa, xo = a[ma], o[mo]
        if len(xa) < 25 or len(xo) < 10:
            return
        da, do_ = xa.groupby("signal_date")["ret"].mean(), xo.groupby("signal_date")["ret"].mean()
        rows.append({"state": name, "rule": rule, "train_n": int(len(xa)), "train_win": round(float((xa["ret"] > 0).mean()), 3), "train_avg": round(float(xa["ret"].mean()), 4),
                     "train_days": int(len(da)), "train_day_win": round(float((da > 0).mean()), 3),
                     "oos_n": int(len(xo)), "oos_win": round(float((xo["ret"] > 0).mean()), 3), "oos_avg": round(float(xo["ret"].mean()), 4),
                     "oos_days": int(len(do_)), "oos_day_win": round(float((do_ > 0).mean()), 3)})
    for c in I.CONDITIONS:
        add(I.CONDITION_LABEL[c], f'market_conditions: ["{c}"]', a["condition"] == c, o["condition"] == c)
    add("Core list coins", "is_core > 0.5", a["tier"] == "core", o["tier"] == "core")
    add("Wider list coins", "is_core < 0.5", a["tier"] == "wide", o["tier"] == "wide")
    for f in STATE_FEATURES:
        if f not in t or t[f].notna().sum() < 60:
            continue
        qs = a[f].quantile([0.2, 0.4, 0.6, 0.8]).values
        edges = [-np.inf, *qs, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            name = f"{f} " + (f"<= {hi:.4g}" if not np.isfinite(lo) else f"> {lo:.4g}" if not np.isfinite(hi) else f"{lo:.4g} to {hi:.4g}")
            rule = (f"{f} <= {hi:.4g}" if not np.isfinite(lo) else f"{f} > {lo:.4g}" if not np.isfinite(hi) else f"{f} > {lo:.4g} and {f} <= {hi:.4g}")
            add(name, rule, (a[f] > lo) & (a[f] <= hi), (o[f] > lo) & (o[f] <= hi))
    base = {"train": bt.stats(a), "oos": bt.stats(o)}
    stand = sorted([r for r in rows if r["train_win"] >= 0.64 and r["oos_win"] >= 0.62 and r["train_avg"] > 0.004 and r["oos_avg"] > 0.004 and r["train_day_win"] >= 0.58 and r["oos_day_win"] >= 0.55],
                   key=lambda r: -(r["train_win"] + r["oos_win"]))
    res = {"ok": True, "base": base, "standouts": stand[:15], "states": rows,
           "how_to_read": "A state must win in BOTH periods, by trade and by signal day, to be more than noise. Turn the most convincing one into a condition (the 'rule' text) and test it as a new idea."}
    (out / "states.json").write_text(json.dumps(C.clean(res), indent=1))
    print(json.dumps(C.clean({"base": {k: {x: v.get(x) for x in ("trades", "win_rate", "avg_ret", "profit_factor")} for k, v in base.items()}, "standouts": stand[:15], "n_states": len(rows)}), indent=1))
    return res


def cmd_promote(data_dir, state_dir, out_dir) -> dict:
    """Candidates join the live book: parameters, market conditions and their win-odds cells are added to strategy/params."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = load_state(state_dir)
    P = S.load_params(st["params"])
    cands = [r for r in st["research"].values() if r.get("verdict") == "candidate" and r.get("spec") and r["id"] not in (P.get("custom") or {})
             and r["id"] not in (P.get("retired") or {})]        # a retired strategy never comes back through the side door
    if not cands:
        res = {"promoted": [], "note": "No new candidate waiting for promotion."}
        print(json.dumps(res))
        return res
    ind, *_ = C.load_ind(data_dir, P)
    b, promoted, notes = I.base_masks(ind), [], []
    for r in sorted(cands, key=lambda r: str(r.get('tested_at')))[:3]:
        ev = C.evaluate(r["spec"], ind, P, b)
        tr = ev.pop("_trades")
        if not ev["enabled"]:
            notes.append(f"{r['id']}: no longer passes on today's data ({ev['reason']})")
            continue
        if ev["market_conditions"]:
            tr = tr[tr["condition"].isin(ev["market_conditions"])]
        cells = C.grade_table(C.tier_of(tr, "core"), P["split"], P["grading"]).get(r["id"], {})
        cells_w = C.grade_table_wide(C.tier_of(tr, "wide"), {r["id"]: cells}, P["grading"]).get(r["id"], {})
        if not any(c.get("grade") in ("A+", "A") for c in cells.values()):
            notes.append(f"{r['id']}: passes the bar but none of its condition cells grades A or A+, so it would never alert")
            continue
        P.setdefault("custom", {})[r["id"]] = {"spec": r["spec"], "promoted_at": C.now_iso()}
        P["strategies"][r["id"]] = {"enabled": True, "params": ev["params"], "market_conditions": ev["market_conditions"], "mode": ev["mode"]}
        P.setdefault("grade_table", {})[r["id"]] = cells
        P.setdefault("grade_table_wide", {})[r["id"]] = cells_w
        promoted.append(r["id"])
        notes.append(f"{r['id']} promoted: {ev['reason']} A+ in: " + (", ".join(c for c, v in cells.items() if v.get("grade") == "A+") or "none yet (watchlist grade)"))
    res = {"promoted": promoted, "note": " ".join(notes)}
    if promoted:
        P["rev"] = int(P.get("rev", 0)) + 1
        C.write_docs_coll(out, [("strategy", "params", P)])
    (out / "promote_report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["next", "test", "states", "promote"])
    ap.add_argument("--spec")
    ap.add_argument("--data")
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "next":
        print(json.dumps(cmd_next(a.state, a.out, a.data), indent=1))
    elif a.cmd == "test":
        cmd_test(a.spec, a.data, a.state, a.out)
    elif a.cmd == "states":
        cmd_states(a.spec, a.data, a.out)
    else:
        cmd_promote(a.data, a.state, a.out)


if __name__ == "__main__":
    main()
