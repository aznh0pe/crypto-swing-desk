"""Account state (from the dashboard database export) and position sizing."""
from __future__ import annotations

import json
import math
import pathlib

DEFAULT_SETTINGS = {"initial_balance": 1500.0, "income_default_pct": 0.0, "max_open_positions": 8, "risk_pct": 2.0,
                    "max_position_pct": 30.0, "maker_fee_pct": 0.50, "taker_fee_pct": 0.90, "exchange": "Coinbase Advanced"}
# Risk governor: full risk per trade is earned by the live record. Size steps down to the floor while the account is
# 8%+ below its peak (realised) or the last 15 closed trades won under 55%; it steps back up once the drawdown is
# inside 4% and the trailing win rate is back above 60%.
GOVERNOR = {"floor_pct": 0.75, "dd_step_down": 0.08, "dd_restore": 0.04, "window": 15, "win_step_down": 0.55, "win_restore": 0.60}


def _unwrap(obj):
    """Database exports may wrap a document; return (doc, version)."""
    if not isinstance(obj, dict):
        return {}, None
    version = obj.get("version") or obj.get("_version")
    for key in ("data", "fields", "doc", "document", "value"):
        if isinstance(obj.get(key), dict) and len(obj) <= 6:
            d = dict(obj[key])
            if obj.get("id") and "id" not in d:
                d["id"] = obj["id"]
            return d, version
    return {k: v for k, v in obj.items() if k not in ("version", "_version")}, version


def load_collection(state_dir, name) -> dict:
    """{doc_id: {'doc': {...}, 'version': n}} from <state_dir>/<name>/*.json, <name>.json (a list or
    {"documents": [...]}) or <name>.jsonl. Tolerant of the shapes the database tools export."""
    out, sd = {}, pathlib.Path(state_dir)
    def add(obj, fallback_id=None):
        doc, ver = _unwrap(obj)
        did = doc.get("id") or (obj.get("id") if isinstance(obj, dict) else None) or fallback_id
        if did and doc:
            out[str(did)] = {"doc": doc, "version": ver}
    p = sd / name
    if p.is_dir():
        for f in sorted(p.rglob("*.json")):
            try:
                js = json.loads(f.read_text())
            except Exception:
                continue
            if isinstance(js, list):
                for o in js:
                    add(o)
            elif isinstance(js, dict) and isinstance(js.get("documents"), list):
                for o in js["documents"]:
                    add(o)
            else:
                add(js, f.stem)
    for f in (sd / f"{name}.json", sd / f"{name}.jsonl"):
        if f.exists():
            txt = f.read_text()
            try:
                js = json.loads(txt)
                for o in (js if isinstance(js, list) else js.get("documents", [])):
                    add(o)
            except Exception:
                for line in txt.splitlines():
                    try:
                        add(json.loads(line))
                    except Exception:
                        pass
    return out


def load_state(state_dir) -> dict:
    sd = pathlib.Path(state_dir) if state_dir else None
    empty = {"settings": dict(DEFAULT_SETTINGS), "trades": {}, "ledger": {}, "alerts": {}, "scans": {}, "learning": {},
             "research": {}, "results": {}, "journal": {}, "params": None, "versions": {}}
    if sd is None or not sd.exists():
        return empty
    g = lambda n: load_collection(sd, n)
    cfg, strat, learn = g("config"), g("strategy"), g("learning")
    st = dict(empty)
    if "settings" in cfg:
        st["settings"].update({k: v for k, v in cfg["settings"]["doc"].items() if v is not None})
    st.update({"trades": {k: v["doc"] for k, v in g("trades").items()}, "ledger": {k: v["doc"] for k, v in g("ledger").items()},
               "alerts": {k: v["doc"] for k, v in g("alerts").items()}, "scans": {k: v["doc"] for k, v in g("scans").items()},
               "learning": {k: v["doc"] for k, v in learn.items()}, "research": {k: v["doc"] for k, v in g("research").items()},
               "results": {k: v["doc"] for k, v in g("research_results").items()}, "journal": {k: v["doc"] for k, v in g("journal").items()},
               "params": strat.get("params", {}).get("doc"),
               "versions": {**{f"strategy/{k}": v["version"] for k, v in strat.items()}, **{f"learning/{k}": v["version"] for k, v in learn.items()},
                            **{f"config/{k}": v["version"] for k, v in cfg.items()}}})
    return st


def funds(state: dict) -> dict:
    s, led = state["settings"], list(state["ledger"].values())
    has_initial = any(d.get("type") == "initial" for d in led)
    total = 0.0 if has_initial else float(s.get("initial_balance", 1500))
    for d in led:
        total += float(d.get("amount") or 0)
    open_tr = [t for t in state["trades"].values() if t.get("status") == "open"]
    cost = sum(float(t.get("entry_price") or 0) * float(t.get("qty") or 0) for t in open_tr)
    return {"capital": round(total, 2), "open_cost": round(cost, 2), "cash": round(total - cost, 2), "open_count": len(open_tr)}


def closed_trades(state: dict) -> list:
    out = [t for t in state["trades"].values() if t.get("status") == "closed" and t.get("exit_price") and t.get("entry_price")]
    return sorted(out, key=lambda t: str(t.get("exit_at") or t.get("exit_date") or ""))


def risk_governor(state: dict, capital: float) -> dict:
    s = state["settings"]
    full = float(s.get("risk_pct", DEFAULT_SETTINGS["risk_pct"]))
    g, floor = GOVERNOR, min(full, GOVERNOR["floor_pct"])
    led = sorted(state["ledger"].values(), key=lambda d: str(d.get("date") or d.get("created_at") or ""))
    eq = peak = 0.0
    for d in led:
        eq += float(d.get("amount") or 0)
        peak = max(peak, eq)
    if not led:
        eq = peak = capital
    dd = (peak - eq) / peak if peak > 0 else 0.0
    recent = closed_trades(state)[-g["window"]:]
    wr = (sum(1 for t in recent if float(t.get("pnl") or 0) > 0) / len(recent)) if len(recent) >= g["window"] else None
    was = bool((state["learning"].get("governor") or {}).get("stepped_down"))
    why = []
    if dd >= g["dd_step_down"] or (was and dd > g["dd_restore"]):
        why.append(f"account {100 * dd:.1f}% below its peak")
    if wr is not None and (wr < g["win_step_down"] or (was and wr < g["win_restore"])):
        why.append(f"last {len(recent)} trades won {100 * wr:.0f}%")
    if why:
        return {"risk_pct": floor, "full_pct": full, "stepped_down": True, "drawdown": round(dd, 4), "trailing_win": wr,
                "reason": f"Risk stepped down to {floor:.2f}% a trade ({'; '.join(why)}). Back to {full:.2f}% once the drawdown is under 4% and the last {g['window']} trades win 60%+."}
    return {"risk_pct": full, "full_pct": full, "stepped_down": False, "drawdown": round(dd, 4), "trailing_win": wr,
            "reason": f"Full size ({full:.2f}% risk a trade): account {100 * dd:.1f}% below its peak."}


def _round_qty(q: float, step=None) -> float:
    """Order size the exchange accepts: floored to the coin's size step (never rounded up past the budget)."""
    if q <= 0:
        return 0.0
    try:
        step = float(step) if step else 0.0
    except (TypeError, ValueError):
        step = 0.0
    if step > 0:
        n = math.floor(q / step + 1e-9)
        dec = max(0, -int(math.floor(math.log10(step) + 1e-9)))
        return float(round(n * step, min(10, dec)))
    digits = max(0, 4 - int(math.floor(math.log10(q))) - 1)   # four significant figures
    f = 10 ** min(8, digits)
    return math.floor(q * f + 1e-9) / f


def size_position(entry: float, stop: float, capital: float, cash: float, risk_pct: float, settings: dict, min_order: float = 25.0, step=None) -> dict:
    """Coins to buy: risk_pct of capital lost at the stop (fees included), capped by the position limit and free cash."""
    maker, taker = float(settings.get("maker_fee_pct", 0.5)) / 100, float(settings.get("taker_fee_pct", 0.9)) / 100
    max_pct = float(settings.get("max_position_pct", 30.0))
    loss_per_usd = max(1e-6, 1 - stop / entry) + maker + taker         # stop distance plus fees in and out
    by_risk = capital * risk_pct / 100 / loss_per_usd
    cap_usd = capital * max_pct / 100
    note = None
    budget = min(by_risk, cap_usd, cash if cash > 0 else cap_usd)
    if cash <= 0:
        note = "No free cash right now: the size shown assumes a position closes first."
    elif cash < min(by_risk, cap_usd):
        note = "Sized down to the cash that is free."
    cost = max(0.0, budget)
    qty = _round_qty(cost / entry, step)
    cost = round(qty * entry, 2)
    if cost < min_order:
        note = f"Too small to place (under ${min_order:.0f}) with the cash that is free."
    return {"qty": qty, "total_cost": cost, "risk_usd": round(cost * loss_per_usd, 2), "pct_of_capital": round(100 * cost / capital, 1) if capital > 0 else None,
            "risk_pct_used": risk_pct, "limited_by": "risk" if by_risk <= min(cap_usd, cash if cash > 0 else cap_usd) else ("cash" if cash > 0 and cash < cap_usd else "position cap"),
            "fees_round_trip_usd": round(cost * (maker + maker), 2), "note": note}
