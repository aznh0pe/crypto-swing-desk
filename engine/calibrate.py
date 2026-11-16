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


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ind(data_dir, P=None):
    w, part, meta = D.load_daily(data_dir)
    ind = I.compute(w, D.load_fng(data_dir))
    return ind, w, part, meta


def split_trades(tr: pd.DataFrame, split: str):
    if tr is None or tr.empty:
        return tr, tr
    cut = pd.Timestamp(split)
    return tr[tr["signal_date"] < cut], tr[tr["signal_date"] >= cut]


def pick_combo(spec: dict, ind: dict, P: dict, b: dict):
    """Grid search on the training years only. Win rate first, with floors that keep it honest."""
    best, tried = None, []
    for pr in rules.combos(spec):
        try:
            plan = rules.build_plan(spec, ind, P, b, pr, name=spec["id"])
            tr = bt.simulate(plan, ind, P["costs"], include_unfilled=True, manage=P.get("manage"))
        except Exception as e:
            tried.append({"params": pr, "error": str(e)[:120]})
            continue
        a, _ = split_trades(tr, P["split"])
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
    """One strategy end to end: chosen parameters, training and unseen results, per condition, verdict."""
    pr, tr, tried = pick_combo(spec, ind, P, b)
    a, o = split_trades(tr, P["split"])
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
    return {"id": spec["id"], "label": spec["label"], "family": spec.get("family"), "about": spec.get("about"), "params": pr,
            "enabled": bool(ok), "mode": mode, "market_conditions": conds, "reason": reason, "checks": checks,
            "train": st_a, "oos": st_o, "by_condition": {"train": bc_a, "oos": bc_o},
            "yearly": bt.yearly(tr) if len(tr) else {}, "grid": tried[:40], "rules": rules.describe({**spec, "params": {**(spec.get("params") or {}), **pr}}),
            "_trades": tr}


def grade_table(trades: pd.DataFrame, split: str, G: dict, train_only: bool = False) -> dict:
    """Win odds per strategy x market condition, shrunk toward the strategy's overall rate.
    {sid: {cond: {"p", "n", "train", "oos", "grade"}}}. With train_only the unseen data is not used (honesty check)."""
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
            p = (w + 20 * base) / (n + 20)
            aw, ow = (float((a["ret"] > 0).mean()) if len(a) else None), (float((o["ret"] > 0).mean()) if len(o) else None)
            avg = float(x["ret"].mean()) if n else 0.0
            strong = (n >= G["aplus_min_n"] and p >= G["aplus_win"] and avg > 0.004 and (aw is None or len(a) < 10 or aw >= G["watch_win"])
                      and (train_only or (ow is not None and len(o) >= 8 and ow >= G["watch_win"])))
            grade = "A+" if strong else ("A" if p >= G["watch_win"] and avg > 0 else "B")
            out[sid][c] = {"p": round(p, 4), "n": int(n), "avg_ret": round(avg, 4), "train": None if aw is None else round(aw, 3),
                           "train_n": int(len(a)), "oos": None if ow is None else round(ow, 3), "oos_n": int(len(o)), "grade": grade}
    return out


def apply_grades(trades: pd.DataFrame, table: dict) -> pd.DataFrame:
    t = trades.copy()
    cells = [table.get(s, {}).get(c) or {} for s, c in zip(t["strategy"], t["condition"])]
    t["p"] = [c.get("p", np.nan) for c in cells]
    t["grade"] = [c.get("grade", "B") for c in cells]
    rr = ((t["target"] - t["limit"]) / (t["limit"] - t["stop"]).replace(0, np.nan)).clip(0, 4).fillna(1.0)
    t["score"] = (100 * t["p"].fillna(0.5) + 2.0 * rr + 3.0 * t["rs_pct"].fillna(0.5)).round(1)
    return t


def run_backtest(data_dir, state_params: dict | None = None, extra_specs: list | None = None) -> tuple[dict, dict]:
    P = S.params_with_defaults({k: v for k, v in (state_params or {}).items() if k in ("costs", "risk", "manage", "grading", "custom", "split")})
    ind, w, part, meta = load_ind(data_dir)
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
    G = P["grading"]
    table = grade_table(allt, P["split"], G) if len(allt) else {}
    P["grade_table"] = table
    grading = {"rule": (f"A+ = the strategy's record in today's market condition shows {100 * G['aplus_win']:.0f}%+ wins after fees "
                        f"(at least {G['aplus_min_n']} trades, shrunk toward its overall rate) and held up on unseen data. "
                        "A = 58%+; listed as a watchlist setup, not sent as an alert.")}
    if len(allt):
        g = apply_grades(allt, table)
        ap = bt.done(g[g["grade"] == "A+"])
        grading["aplus"] = {**bt.stats(ap), "by_year": bt.yearly(ap), "by_condition": bt.by_condition(ap)}
        honest = apply_grades(allt, grade_table(allt, P["split"], G, train_only=True))
        ho = bt.done(honest[(honest["grade"] == "A+") & (honest["signal_date"] >= pd.Timestamp(P["split"]))])
        grading["honest_check"] = {"what": "A+ cells chosen from the training years only, measured on unseen data", **bt.stats(ho)}
        allst = bt.done(g)
        grading["all_enabled"] = bt.stats(allst)
        rk = P["risk"]
        grading["account_replay"] = {"what": f"$1,500 start, A+ setups only, {rk['max_open_positions']} slots, {rk['risk_pct']}% risk a trade",
                                     "full_history": bt.portfolio(ap, 1500.0, rk["max_open_positions"], rk["risk_pct"], rk["max_position_pct"]),
                                     "unseen_only": bt.portfolio(ap[ap["signal_date"] >= pd.Timestamp(P["split"])], 1500.0, rk["max_open_positions"], rk["risk_pct"], rk["max_position_pct"])}
    mk = ind["market"]
    cs = I.condition_series(mk).dropna()
    share = cs.value_counts(normalize=True).round(3).to_dict()
    P.update({"calibrated_at": now_iso(), "data_through": str(ind["close"].index.max().date()), "version": int((state_params or {}).get("version", 0)) + 1})
    report = {"generated_at": now_iso(), "data": {"first_bar": str(ind["close"].index.min().date()), "last_bar": str(ind["close"].index.max().date()),
                                                  "pairs": int(ind["close"].shape[1]), "liquid_today": int(ind["liquid"].iloc[-1].sum()), "source": meta.get("source")},
              "split": P["split"], "costs": P["costs"], "enable_rule": S.ENABLE_RULE, "specialist_rule": SPECIALIST,
              "condition_rule": I.CONDITION_RULE, "condition_share": share, "condition_labels": I.CONDITION_LABEL,
              "strategies": results, "grading": grading, "grade_table": table}
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


def write_docs(out, docs: dict, versions: dict | None = None):
    """docs = {"collection/doc": data}. Writes each as a file under out/docs/ and out/writes.json for the database."""
    out = pathlib.Path(out)
    writes = []
    for path, doc in docs.items():
        f = out / "docs" / (path + ".json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(clean(doc), separators=(",", ":")))
        writes.append({"op": "set", "path": path, "file": str(f)})
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
        extra = [r["spec"] for r in st.get("research", {}).values() if r.get("verdict") == "candidate" and r.get("spec")]
    P, report = run_backtest(a.data, sp, extra)
    for sid in [e["id"] for e in extra]:
        if P["strategies"].get(sid, {}).get("enabled"):
            P["custom"][sid] = {"spec": next(e for e in extra if e["id"] == sid), "promoted_at": now_iso()}
        elif sid not in (P.get("custom") or {}):
            P["strategies"].pop(sid, None)
    (out / "params.json").write_text(json.dumps(clean(P), indent=1))
    (out / "backtest_report.json").write_text(json.dumps(clean(report), indent=1))
    write_docs(out, {"strategy/params": P, "strategy/backtest": report})
    for sid, r in report["strategies"].items():
        t, o = r["train"], r["oos"]
        print(f"{sid:16s} {'ON ' if r['enabled'] else 'off'} {r['mode']:12s} train n={t.get('trades', 0):5d} win={t.get('win_rate')} pf={t.get('profit_factor')} avg={t.get('avg_ret')} | "
              f"unseen n={o.get('trades', 0):5d} win={o.get('win_rate')} pf={o.get('profit_factor')} avg={o.get('avg_ret')}")
    print(json.dumps(clean(report["grading"]), indent=1)[:3000])


if __name__ == "__main__":
    main()
