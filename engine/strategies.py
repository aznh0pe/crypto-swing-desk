"""The strategy book: built-in specs (strategies.json), promoted research specs (params["custom"]),
the live-trading bar, and plan building."""
from __future__ import annotations

import json
import pathlib

from . import rules
from .indicators import base_masks, CONDITIONS

HERE = pathlib.Path(__file__).resolve().parent
SPLIT = "2023-01-01"     # training years end here; everything after is unseen data
# The live-trading bar, after fees. Win rate comes first; profit factor and average trade keep it honest.
ENABLE_RULE = {"train_win": 0.58, "train_pf": 1.30, "train_avg": 0.004, "train_n": 40, "train_day_win": 0.55, "oos_n": 20, "oos_pf": 1.10, "oos_decay": 0.08}
DEFAULT_PARAMS = {
    "version": 1, "split": SPLIT,
    "costs": {"maker_pct": 0.50, "taker_pct": 0.90, "slippage_pct": 0.10},
    "risk": {"risk_pct": 2.0, "max_position_pct": 30.0, "max_open_positions": 4, "min_order_usd": 25.0, "max_per_day": 3},
    "manage": {"lock_atr": 2.0, "lock_to": 0.5},
    "grading": {"aplus_win": 0.66, "aplus_min_n": 25, "aplus_min_days": 10, "watch_win": 0.58},
    "strategies": {}, "custom": {},
}


def builtin_specs() -> dict:
    return {s["id"]: s for s in json.loads((HERE / "strategies.json").read_text())}


def all_specs(P: dict | None = None) -> dict:
    out = builtin_specs()
    for k, v in ((P or {}).get("custom") or {}).items():
        out[k] = v["spec"] if "spec" in v else v
    return out


def params_with_defaults(p: dict | None) -> dict:
    P = json.loads(json.dumps(DEFAULT_PARAMS))
    for k, v in (p or {}).items():
        if isinstance(v, dict) and isinstance(P.get(k), dict) and k not in ("strategies", "custom"):
            P[k].update(v)
        else:
            P[k] = v
    return P


def load_params(state_params: dict | None = None) -> dict:
    """Live parameters: the dashboard's strategy/params document if given, else the copy shipped with the engine."""
    if state_params and state_params.get("strategies"):
        return params_with_defaults(state_params)
    f = HERE / "params.json"
    return params_with_defaults(json.loads(f.read_text()) if f.exists() else None)


def build(sid: str, ind: dict, P: dict, b: dict | None = None, params: dict | None = None, force: bool = False):
    spec = all_specs(P).get(sid)
    rec = (P.get("strategies") or {}).get(sid) or {}
    if spec is None or (not force and not rec.get("enabled")):
        return None
    spec = dict(spec)
    if rec.get("market_conditions"):
        spec["market_conditions"] = rec["market_conditions"]
    return rules.build_plan(spec, ind, P, b or base_masks(ind), params if params is not None else rec.get("params"), name=sid)


def all_plans(ind: dict, P: dict, include_off: bool = False) -> dict:
    b, out = base_masks(ind), {}
    for sid in all_specs(P):
        try:
            pl = build(sid, ind, P, b, force=include_off)
        except Exception:
            pl = None
        if pl is not None:
            out[sid] = pl
    return out


def enable_checks(tr: dict, te: dict) -> tuple[bool, list, str]:
    """(enabled, checks, plain-English reason) for the live-trading bar."""
    R = ENABLE_RULE
    tw, tpf, tav, tn = tr.get("win_rate"), tr.get("profit_factor"), tr.get("avg_ret"), tr.get("trades", 0)
    ow, opf, on = te.get("win_rate"), te.get("profit_factor"), te.get("trades", 0)
    dec = (tw - ow) if (tw is not None and ow is not None) else None
    checks = [
        {"key": "train_n", "label": "Trades in the training years", "value": tn, "need": f">= {R['train_n']}", "ok": tn >= R["train_n"]},
        {"key": "train_win", "label": "Win rate, training years", "value": tw, "need": f">= {R['train_win']:.0%}", "ok": tw is not None and tw >= R["train_win"]},
        {"key": "train_pf", "label": "Profit factor, training years", "value": tpf, "need": f">= {R['train_pf']:.2f}", "ok": tpf is not None and tpf >= R["train_pf"]},
        {"key": "train_avg", "label": "Average trade after fees, training years", "value": tav, "need": f"> +{100 * R['train_avg']:.1f}%", "ok": tav is not None and tav > R["train_avg"]},
        {"key": "train_day_win", "label": "Winning signal days, training years", "value": tr.get("day_win_rate"), "need": f">= {R['train_day_win']:.0%}", "ok": (tr.get("day_win_rate") or 0) >= R["train_day_win"]},
        {"key": "oos_n", "label": "Trades on unseen data", "value": on, "need": f">= {R['oos_n']}", "ok": on >= R["oos_n"]},
        {"key": "oos_pf", "label": "Profit factor, unseen data", "value": opf, "need": f">= {R['oos_pf']:.2f}", "ok": bool(on) and opf is not None and opf >= R["oos_pf"]},
        {"key": "oos_decay", "label": "Win-rate drop on unseen data", "value": dec, "need": f"<= {100 * R['oos_decay']:.0f} pts", "ok": dec is not None and dec <= R["oos_decay"] + 1e-9},
    ]
    ok = all(c["ok"] for c in checks)
    if ok:
        reason = f"Active: passed every live-trading check ({100 * tw:.0f}% wins in training, {100 * ow:.0f}% on unseen data, profit factor {opf:.2f}, all after fees)."
    else:
        bad = [f"{c['label'].lower()} {('%.3g' % c['value']) if isinstance(c['value'], (int, float)) else 'n/a'} (needs {c['need']})" for c in checks if not c["ok"]]
        reason = "Off: " + "; ".join(bad) + ". Re-tested at every weekly recalibration; it switches on by itself if it passes."
    return ok, checks, reason
