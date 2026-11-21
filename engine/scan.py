"""The scanner: new A+ setups, what to do with every open position, and whether resting orders filled.

  python -m engine.scan run  --mode close|check --data D [--state S] --out O
  python -m engine.scan news --out O "TICKER=clean" "TICKER=caution:reason" "TICKER=avoid:reason" [--issue "..."]

mode close  = right after the daily bar closes (00:00 UTC): tomorrow's setups, exits, fills
mode check  = between closes: open positions, fills and live prices only (no new setups)

Writes into O: scan.json, digest.md, headline.txt, docs/<collection>/<id>.json and writes.json
(a list of {"op","collection","doc_id","file_path"} for the dashboard database).
"""
from __future__ import annotations

import argparse
import hashlib
import datetime as dt
import json
import math
import pathlib
import traceback

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as I
from . import rules
from .calibrate import CONFLUENCE_BONUS
from . import strategies as S
from . import version
from .calibrate import clean
from .learn import apply_live
from .portfolio import funds, load_state, risk_governor, size_position

UTC = dt.timezone.utc
DYN_TEXT = {"sma5": "Sell at market after any daily close above the 5-day average.",
            "prev_high": "Sell at market after any daily close above the previous day's high.",
            "high7": "Sell at market after a daily close at a 7-day closing high.",
            "rsi2_70": "Sell at market after a daily close with RSI(2) above 70.",
            "ema10_lost": "Sell at market after a daily close below the 10-day EMA.",
            "none": "No signal exit: the target, the stop or the time limit ends it."}


def iso(t) -> str:
    return pd.Timestamp(t).tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tick(x: float, tick) -> float:
    try:
        tk = float(tick)
    except Exception:
        tk = 0.0
    if not tk or tk <= 0:
        mag = 10 ** (math.floor(math.log10(abs(x))) - 4) if x > 0 else 1e-8   # five significant figures
        tk = mag
    return float(round(math.floor(x / tk + 1e-9) * tk, 10))


def _write(out, writes, op, coll, did, data):
    p = out / "docs" / coll / f"{did}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = next((x for x in writes if x["collection"] == coll and x["doc_id"] == did), None)
    if prev is not None and op == "update":          # two updates to one doc in one run share a file: merge, never overwrite
        p.write_text(json.dumps(clean({**json.loads(p.read_text()), **data}), default=str))
        return
    p.write_text(json.dumps(clean(data), default=str))
    writes.append({"op": op, "collection": coll, "doc_id": did, "file_path": str(p.resolve())})


def live_prices(w, part, hourly) -> tuple[dict, str | None]:
    """Latest price per coin and when it is from: the hourly feed, else the forming daily bar, else the last close."""
    px, asof = w["close"].iloc[-1].dropna().to_dict(), iso(w["close"].index[-1] + pd.Timedelta(days=1))
    if part is not None:
        px.update(part["close"].to_dict())
    if hourly is not None and len(hourly["close"]):
        last = hourly["close"].ffill(limit=6).iloc[-1].dropna()
        px.update(last.to_dict())
        asof = iso(hourly["close"].index[-1] + pd.Timedelta(hours=1))
    return px, asof


_naive = D.naive_ts


def extremes_since(tk: str, since: pd.Timestamp, w, part, hourly) -> dict:
    """Lowest low and highest high since `since` (UTC).

    Hourly bars are used when they reach back to `since`. Otherwise daily bars fill in, and the high of the day
    that contains `since` is left out: on daily bars alone nobody can tell whether that high printed before or
    after the fill, and the backtest does not count a target on the fill day either."""
    lows, highs, lo_at = [], [], None
    h_ok = hourly is not None and tk in hourly["low"] and len(hourly["low"])
    covered = bool(h_ok) and hourly["low"].index[0] <= since
    if h_ok:
        sel = hourly["low"].index >= since.floor("h")
        seg_l, seg_h = hourly["low"][tk][sel].dropna(), hourly["high"][tk][sel].dropna()
        if len(seg_l):
            lows.append(float(seg_l.min())); highs.append(float(seg_h.max())); lo_at = iso(seg_l.idxmin())
    if not covered:
        d0 = since.normalize()
        h_low = lows[0] if lows else None
        if tk in w["low"]:
            seg_l, seg_h = w["low"][tk][w["low"].index >= d0].dropna(), w["high"][tk][w["high"].index > d0].dropna()
            lows += list(seg_l.values); highs += list(seg_h.values)
        if part is not None and tk in part["low"]:
            lows.append(float(part["low"][tk]))
            if part.get("day") is None or part["day"] > d0:
                highs.append(float(part["high"][tk]))
        if h_low is None or min(lows) < h_low:
            lo_at = None                      # the low came from a daily bar: its hour is unknown
    return {"low": float(min(lows)) if lows else None, "high": float(max(highs)) if highs else None, "low_at": lo_at}


def market_now(ind, px: dict | None = None) -> dict:
    mk = ind["market"].iloc[-1]
    cond = I.condition_series(ind["market"]).iloc[-1]
    f = lambda k, n=4: None if not np.isfinite(mk.get(k, np.nan)) else round(float(mk[k]), n)
    return {"condition": cond, "label": I.CONDITION_LABEL.get(cond, "Unknown"), "bull": bool(mk["bull"] >= 0.5), "stress": bool(mk["stress"] >= 0.5),
            "btc": f("btc", 2), "btc_now": None if not (px or {}).get("BTC") else round(float(px["BTC"]), 2),
            "washed_pct": None if f("washed") is None else round(100 * f("washed")), "flush7_pct": None if f("flush7") is None else round(100 * f("flush7")),
"zone": I.zone_of(mk.get("btc_dist200")), "zone_label": I.ZONE_LABEL[I.zone_of(mk.get("btc_dist200"))],
            "btc_vs_200d_pct": None if f("btc_dist200") is None else round(100 * f("btc_dist200"), 1),
            "btc_vs_50d_pct": None if f("btc_dist50") is None else round(100 * f("btc_dist50"), 1),
            "btc_7d_pct": None if f("btc_ret7") is None else round(100 * f("btc_ret7"), 1),
            "btc_30d_pct": None if f("btc_ret30") is None else round(100 * f("btc_ret30"), 1),
            "btc_vol_ratio": f("btc_vol_ratio", 2), "breadth_above_50d_pct": None if f("breadth50") is None else round(100 * f("breadth50")),
            "breadth_above_200d_pct": None if f("breadth200") is None else round(100 * f("breadth200")),
            "fear_greed": f("fng", 0), "alts_7d_pct": None if f("alt_ret7") is None else round(100 * f("alt_ret7"), 1), "rule": I.CONDITION_RULE}


def find_entries(ind, P, uni, cond, paused: dict) -> tuple[list, list]:
    """(setups for the next day from the last finished bar, notes). One per coin: its best-scoring strategy."""
    c, atr10 = ind["close"], ind["atr10"]
    day = c.index[-1]
    specs, table, table_w, costs, G = S.all_specs(P), P.get("grade_table") or {}, P.get("grade_table_wide") or {}, P["costs"], P["grading"]
    maker, taker = costs["maker_pct"] / 100, costs["taker_pct"] / 100
    best, notes, agree = {}, [], {}
    nest = P.get("confluence_groups") or {}
    zone = I.zone_of(ind["market"]["btc_dist200"].iloc[-1])
    zs = P.get("zones") or {}
    blocked_zones = set(zs.get("blocked") or [])
    for sid, plan in S.all_plans(ind, P).items():
        if sid in paused:
            notes.append(f"{specs[sid]['label']} is paused by the daily review: {paused[sid]}")
            continue
        row = plan["mask"].iloc[-1]
        for tk in row.index[row.values]:
            u = uni.get(tk) or {}
            if uni and (not u or u.get("status") != "online" or u.get("peg")):
                continue        # only coins that trade on Coinbase right now
            tier = "core" if bool(ind["is_core"][tk].iloc[-1]) else "wide"
            atr, last = float(atr10[tk].iloc[-1]), float(c[tk].iloc[-1])
            market = plan["entry_kind"] == "open_next"
            entry = last if market else float(plan["limit"][tk].iloc[-1])
            if not (np.isfinite(entry) and np.isfinite(atr) and entry > 0 and atr > 0):
                continue
            entry = _tick(entry, u.get("tick"))
            ss, ts = plan["stop"], plan["target"]
            stop = float(ss["frame"][tk].iloc[-1]) if ss["kind"] == "level" else entry - ss["mult"] * atr
            if not np.isfinite(stop) or stop >= entry:
                stop = entry - 2.0 * atr
            stop = max(stop, entry * 0.5)
            if ts["kind"] == "level":
                tg = float(ts["frame"][tk].iloc[-1])
                lo_t, hi_t = entry + ts.get("min_atr", 0.5) * atr, entry + ts.get("cap_atr", 5.0) * atr
                tg = lo_t if not np.isfinite(tg) else min(max(tg, lo_t), hi_t)
            else:
                tg = entry + ts["mult"] * atr
            stop, tg = _tick(stop, u.get("tick")), _tick(tg, u.get("tick"))
            cell = ((table_w if tier == "wide" else table).get(sid) or {}).get(cond) or {}
            p, grade = cell.get("p"), cell.get("grade", "B")
            if p is None:
                continue
            zone_blocked = zone_cell = False
            if zone in blocked_zones and grade == "A+":        # danger zone: only a strategy x tier with its own A+ record there trades it
                zc = ((zs.get("exempt") or {}).get(zone) or {}).get(f"{sid}|{tier}")
                if zc:
                    cell, p, zone_cell = {**cell, **zc}, zc["p"], True
                else:
                    grade, zone_blocked = "A", True
            fee_in = taker if market else maker
            net_win = (tg * (1 - maker)) / (entry * (1 + fee_in)) - 1
            net_loss = (stop * (1 - taker)) / (entry * (1 + fee_in)) - 1
            rr = net_win / abs(net_loss) if net_loss < 0 else 0.0
            rs = float(ind["rs_pct"][tk].iloc[-1]) if np.isfinite(ind["rs_pct"][tk].iloc[-1]) else 0.5
            score = round(100 * p + 2.0 * min(4.0, (tg - entry) / max(entry - stop, 1e-12)) + 3.0 * rs, 1)
            sp = specs[sid]
            e = {"ticker": tk, "product": u.get("product") or f"{tk}-USD", "strategy": sid, "strategy_label": sp["label"], "family": sp.get("family"),
                 "about": sp.get("about"), "grade": grade, "score": score, "win_pct": round(100 * p, 1), "cell_n": cell.get("n"), "cell_days": cell.get("days"),
                 "tier": tier, "min_size": u.get("min_size"), "zone": zone, "zone_blocked": zone_blocked, "zone_cell": zone_cell,
                 "cell_train": cell.get("train"), "cell_oos": cell.get("oos"), "condition": cond, "entry_kind": plan["entry_kind"],
                 "entry": entry, "stop": stop, "target": tg, "atr": round(atr, 10), "last_close": last,
                 "stop_pct": round(100 * (stop / entry - 1), 2), "target_pct": round(100 * (tg / entry - 1), 2),
                 "net_win_pct": round(100 * net_win, 2), "net_loss_pct": round(100 * net_loss, 2), "rr": round(rr, 2),
                 "hold_max_days": int(plan["max_hold"]), "valid_days": int(plan.get("valid_days", 1)), "dyn_exit": plan.get("dyn_exit", "none"),
                 "exit_signal": DYN_TEXT.get(plan.get("dyn_exit", "none"), ""), "signal_date": str(day.date()),
                 "rs_pct": round(100 * rs), "rsi2": round(float(ind["rsi2"][tk].iloc[-1]), 1), "rsi14": round(float(ind["rsi14"][tk].iloc[-1]), 1),
                 "vs_50d_pct": round(100 * float(ind["dist50"][tk].iloc[-1]), 1), "vs_200d_pct": round(100 * float(ind["dist200"][tk].iloc[-1]), 1) if np.isfinite(ind["dist200"][tk].iloc[-1]) else None,
                 "atr_pct": round(100 * float(ind["atr_pct"][tk].iloc[-1]), 1), "dollar_vol_30d": round(float(ind["dv30"][tk].iloc[-1])),
                 "is_major": bool(ind["is_major"][tk].iloc[-1])}
            if grade == "A+":
                agree.setdefault(tk, {}).setdefault(nest.get(sid, sid), []).append(sp["label"])
            if tk not in best or (e["grade"] == "A+", e["score"]) > (best[tk]["grade"] == "A+", best[tk]["score"]):
                best[tk] = e
    for tk, e in best.items():        # agreement between A+ strategies is the strongest ranking signal measured: it lifts the score
        k = len(agree.get(tk, {}))             # independent groups: strategies that take the same trades count once
        e["confluence"] = max(1, k)
        e["also_by"] = [x for v in agree.get(tk, {}).values() for x in v if x != e["strategy_label"]]
        if e["grade"] == "A+" and k > 1:
            e["score"] = round(e["score"] + CONFLUENCE_BONUS * (min(k, 3) - 1), 1)
    if zone in blocked_zones:
        held = sorted(e["ticker"] for e in best.values() if e.get("zone_blocked"))
        row = (zs.get("table") or {}).get(zone) or {}
        ex = sorted((zs.get("exempt") or {}).get(zone) or {})
        names = [f"{specs[k.split('|')[0]]['label']} on {'core' if k.endswith('|core') else 'wider-list'} coins" for k in ex if k.split("|")[0] in specs]
        notes.append(f"Danger zone: {I.ZONE_LABEL[zone]}. A+ setups in this zone won {100 * row['train']['win_rate']:.0f}% in the training years and {100 * row['unseen']['win_rate']:.0f}% on unseen data "
                     f"({row['all']['trades']} trades, {row['all']['days']} days), so only strategies with their own A+ record in this zone are sent"
                     + (f": {'; '.join(names)}" if names else " (none has one yet)") + (f". Moved to the watchlist: {', '.join(held)}." if held else "."))
    out = sorted(best.values(), key=lambda e: (e["grade"] != "A+", -e["score"]))
    return out, notes


def book_id(P: dict) -> str:
    """A stamp of everything that decides which setups are served: strategy settings, grades and the coin-list floor.
    A code release that changes none of these must not trigger a second look for setups."""
    grades = lambda t: {s: {c: v.get("grade") for c, v in (cells or {}).items()} for s, cells in (t or {}).items()}
    body = {"strategies": P.get("strategies"), "core": grades(P.get("grade_table")), "wide": grades(P.get("grade_table_wide")),
            "floor": (P.get("universe") or {}).get("min_dv"), "retired": sorted(P.get("retired") or {}),
            "zones": [sorted((P.get("zones") or {}).get("blocked") or []), {z: sorted(v) for z, v in ((P.get("zones") or {}).get("exempt") or {}).items()}]}
    # the rules themselves too: a revised rule (a new filter, a new entry price) changes what is served even when the
    # strategy's settings and grades stay the same
    rules_now = {}
    for sid, rec in (P.get("strategies") or {}).items():
        if (rec or {}).get("enabled"):
            c = ((P.get("custom") or {}).get(sid) or {})
            sp = c.get("spec") or c
            if sp:
                rules_now[sid] = {k: sp.get(k) for k in ("conditions", "entry", "exit", "stop", "target", "max_hold", "universe", "rev")}
    if rules_now:
        body["rules"] = rules_now
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]


def radar(ind, P, uni, cond, paused: dict, limit: int = 14) -> dict:
    """What is close: for every strategy with an A+ record in today's market condition, the coins one condition
    short of a signal, or the single market-wide condition it is waiting for."""
    specs, table, table_w, b = S.all_specs(P), P.get("grade_table") or {}, P.get("grade_table_wide") or {}, I.base_masks(ind)
    coins, waiting = [], []
    zs = P.get("zones") or {}
    zone, blocked = I.zone_of(ind["market"]["btc_dist200"].iloc[-1]), set(zs.get("blocked") or [])
    for sid, spec in specs.items():
        rec = (P.get("strategies") or {}).get(sid) or {}
        if not rec.get("enabled") or sid in paused or (rec.get("market_conditions") and cond not in rec["market_conditions"]):
            continue
        gc, gw = ((table.get(sid) or {}).get(cond) or {}), ((table_w.get(sid) or {}).get(cond) or {})
        if zone in blocked:        # in a danger zone the radar shows only what could actually be served there
            ez = (zs.get("exempt") or {}).get(zone) or {}
            gc = {**gc, **ez[f"{sid}|core"]} if f"{sid}|core" in ez else {}
            gw = {**gw, **ez[f"{sid}|wide"]} if f"{sid}|wide" in ez else {}
        if "A+" not in (gc.get("grade"), gw.get("grade")):
            continue
        try:
            nm = rules.near_misses(spec, ind, b, rec.get("params"))
        except Exception:
            continue
        if nm["market"]:
            ready = [tk for tk in nm["ready"] if not uni or ((uni.get(tk) or {}).get("status") == "online" and not (uni.get(tk) or {}).get("peg"))]
            ready = [tk for tk in ready if (gc if bool(ind["is_core"][tk].iloc[-1]) else gw).get("grade") == "A+"]
            if not ready and zone in blocked:
                continue
            waiting.append({"strategy": sid, "strategy_label": spec["label"], "needs": "; ".join(m["needs"] for m in nm["market"]), "now": "; ".join(m["now"] for m in nm["market"] if m["now"]),
                            "conditions_missing": len(nm["market"]), "coins_ready": len(ready), "examples": ready[:8]})
            continue
        for r in nm["coins"]:
            u = uni.get(r["ticker"]) or {}
            if uni and (not u or u.get("status") != "online" or u.get("peg")):
                continue
            tier = "core" if bool(ind["is_core"][r["ticker"]].iloc[-1]) else "wide"
            g = (gw if tier == "wide" else gc)
            if g.get("grade") != "A+":
                continue
            coins.append({"ticker": r["ticker"], "strategy": sid, "strategy_label": spec["label"], "needs": r["needs"], "now": r["now"], "tier": tier, "win_pct": round(100 * g["p"], 1)})
    coins.sort(key=lambda r: -r["win_pct"])
    waiting.sort(key=lambda x: (x["conditions_missing"], -x["coins_ready"]))
    return {"coins": coins[:limit], "more": max(0, len(coins) - limit), "waiting": waiting}


def check_position(tr: dict, ind, w, part, hourly, px: dict, P: dict, mode: str, now: pd.Timestamp) -> dict:
    """What to do now with one open position."""
    tk = tr.get("ticker")
    entry, qty = float(tr.get("entry_price") or 0), float(tr.get("qty") or 0)
    stop, tgt = float(tr.get("stop") or 0), float(tr.get("target") or 0)
    hold = int(tr.get("hold_max_days") or 7)
    at = _naive(tr.get("entry_at") or tr.get("entry_date"), now)
    last = px.get(tk)
    fill_day = at.normalize()
    last_hold_close = fill_day + pd.Timedelta(days=hold)          # 00:00 UTC that ends the last hold day
    hours_left = (last_hold_close - now).total_seconds() / 3600
    days_held = int((now.normalize() - fill_day).days) + 1
    ex = extremes_since(tk, at, w, part, hourly)
    out = {"trade_id": tr.get("id"), "ticker": tk, "last": last, "entry": entry, "qty": qty, "stop": stop, "target": tgt,
           "days_held": days_held, "hold_max_days": hold, "time_stop_at": iso(last_hold_close), "hours_left": round(hours_left, 1),
           "days_left": round(max(0.0, hours_left) / 24, 1), "low_since": ex["low"], "high_since": ex["high"]}
    if last is None or not entry:
        return {**out, "code": "NO_PRICE", "label": "NO PRICE", "tone": "act", "detail": "No current price for this coin in the feed. Check it on your exchange."}
    out["unrealized"] = round((last - entry) * qty, 2)
    out["pct"] = round(100 * (last / entry - 1), 2)
    dyn = tr.get("dyn_exit") or "none"
    c = ind["close"][tk] if tk in ind["close"] else None
    atr = float(tr.get("atr") or (ind["atr10"][tk].iloc[-1] if c is not None else 0) or 0)
    mg = P.get("manage") or {}
    done_bar = c is not None and c.index[-1] >= fill_day    # at least one finished daily bar since the fill
    sig = False
    if done_bar and dyn != "none":
        cl = float(c.iloc[-1])
        sig = {"sma5": cl > float(ind["sma5"][tk].iloc[-1]), "prev_high": len(c) > 1 and cl > float(ind["high"][tk].iloc[-2]),
               "high7": cl >= float(ind["max7c"][tk].iloc[-1]), "rsi2_70": float(ind["rsi2"][tk].iloc[-1]) > 70,
               "ema10_lost": c.index[-1] > fill_day and cl < float(ind["ema10"][tk].iloc[-1])}.get(dyn, False)
    if stop and ex["low"] is not None and ex["low"] <= stop:
        return {**out, "code": "STOP_HIT", "label": "STOP HIT: confirm the sale", "tone": "sell",
                "detail": f"Price traded down to {ex['low']:.6g}, through your stop at {stop:.6g}. If your stop order filled, record the sale. If you had no stop order resting, sell now at market."}
    if tgt and ex["high"] is not None and ex["high"] >= tgt:
        return {**out, "code": "TARGET_HIT", "label": "TARGET HIT: confirm the sale", "tone": "good",
                "detail": f"Price traded up to {ex['high']:.6g}, through your target at {tgt:.6g}. If your limit sell filled, record the sale. If you had none resting, sell now."}
    if sig:
        return {**out, "code": "EXIT_RULE", "label": "SELL NOW: exit signal", "tone": "sell",
                "detail": f"The strategy's exit signal fired on the last daily close. {DYN_TEXT.get(dyn, '')} Sell at market and cancel the stop and target orders."}
    zs = P.get("zones") or {}
    zone = I.zone_of(ind["market"]["btc_dist200"].iloc[-1])
    if zone in (zs.get("blocked") or []) and tr.get("strategy") and c is not None:
        tier = "core" if bool(ind["is_core"][tk].iloc[-1]) else "wide"
        if f"{tr['strategy']}|{tier}" not in ((zs.get("exempt") or {}).get(zone) or {}):
            row = (zs.get("table") or {}).get(zone) or {}
            fee = P["costs"]
            even = entry * (1 + fee["maker_pct"] / 100) / (1 - fee["taker_pct"] / 100)
            rec = (f"A+ setups in this zone won {100 * row['train']['win_rate']:.0f}% in the training years and {100 * row['unseen']['win_rate']:.0f}% on unseen data" if row.get("train") else "The A+ book loses in this zone")
            up = last >= even
            return {**out, "code": "ZONE_EXIT", "label": "SELL NOW: take the bounce" if up else "SELL: danger zone", "tone": "sell", "breakeven": round(even, 10),
                    "detail": (f"Danger zone ({I.ZONE_LABEL[zone]}): {rec}, and this strategy on {'core' if tier == 'core' else 'wider-list'} coins has no A+ record of its own here. "
                               "Tested on past trades like it, every way of holding lost on average and selling the first bounce did best. "
                               + (f"Price {last:.6g} is above your break-even after fees ({even:.6g}): sell now and cancel the stop and target orders." if up else
                                  f"Price {last:.6g} is under your break-even after fees ({even:.6g}): sell at market, or rest a sell limit at {even:.6g} for today and sell at market at the daily close if it has not filled. Keep the stop at {stop:.6g} until then."))}
    if hours_left <= 0:
        return {**out, "code": "TIME_STOP", "label": "SELL NOW: time is up", "tone": "sell",
                "detail": f"Held {days_held} days, the limit for this setup is {hold}. Sell at market and cancel the stop and target orders."}
    if mg.get("lock_atr") and atr and done_bar and float(c.iloc[-1]) - entry >= mg["lock_atr"] * atr and stop < entry + mg.get("lock_to", 0.5) * atr:
        ns = round(entry + mg.get("lock_to", 0.5) * atr, 10)
        return {**out, "code": "RAISE_STOP", "label": "RAISE YOUR STOP", "tone": "act", "new_stop": ns,
                "detail": f"The last daily close is {mg['lock_atr']:g} ATR above your entry. Move the stop up to {ns:.6g} to lock in a gain. Target unchanged at {tgt:.6g}."}
    last_day = hours_left <= 24
    return {**out, "code": "HOLD", "label": "HOLD" + (": last day" if last_day else ""), "tone": "hold",
            "detail": (f"Stop {stop:.6g}, target {tgt:.6g}. " + (f"Time limit ends {iso(last_hold_close)[:16].replace('T', ' ')} UTC: sell at market then if neither level is hit. " if last_day else
                       f"{out['days_left']:g} days left of {hold}. ") + (DYN_TEXT.get(dyn, "") if dyn != "none" else ""))}


def window_lows(tk: str, vf: pd.Timestamp, vu: pd.Timestamp, w, part, hourly) -> dict:
    """Lowest trade inside the order window [vf, vu) and, separately, after it.

    An order that is past its window is judged on the window alone: a dip that came after the window says nothing
    about whether the order filled in time, but it does matter if the order was left resting on the exchange."""
    win = after = win_at = after_at = None
    h_ok = hourly is not None and tk in hourly["low"] and len(hourly["low"])
    if h_ok and hourly["low"].index[0] <= vf:
        s = hourly["low"][tk].dropna()
        a, b = s[(s.index >= vf.floor("h")) & (s.index < vu)], s[s.index >= vu]
        if len(a):
            win, win_at = float(a.min()), iso(a.idxmin())
        if len(b):
            after, after_at = float(b.min()), iso(b.idxmin())
        return {"win": win, "win_at": win_at, "after": after, "after_at": after_at, "source": "hourly"}
    lows_w, lows_a = [], []                    # daily bars: whole UTC days that overlap the window, then the days after
    d0, d1 = vf.normalize(), vu.normalize()
    if tk in w["low"]:
        s = w["low"][tk].dropna()
        lows_w += list(s[(s.index >= d0) & (s.index < d1)].values); lows_a += list(s[s.index >= d1].values)
    if part is not None and tk in part["low"] and part.get("day") is not None and np.isfinite(part["low"][tk]):
        (lows_w if d0 <= part["day"] < d1 else lows_a if part["day"] >= d1 else []).append(float(part["low"][tk]))
    return {"win": float(min(lows_w)) if lows_w else None, "win_at": None, "after": float(min(lows_a)) if lows_a else None,
            "after_at": None, "source": "daily"}


def _when(ts) -> str:
    return "" if not ts else f" (around {str(ts)[:16].replace('T', ' ')} UTC)"


def check_order(a: dict, w, part, hourly, px: dict, now: pd.Timestamp, feed_at: pd.Timestamp | None = None) -> dict | None:
    """Did a resting entry order trade? Same rule as the backtest: the low must trade below the limit.

    Once the order's window is over the verdict is final and comes with an instruction (`action`):
      cancel       the low inside the window never reached the limit: the order did not fill, cancel it
      record_fill  the low inside the window was below the limit: it should have filled, record it
      check        not provable from prices (touched the limit exactly, traded below it only after the window,
                   or the feed has not covered the whole window yet): look at the exchange, cancel if still open"""
    try:
        vf, vu = _naive(a["valid_from"]), _naive(a["valid_until"])
    except Exception:
        return None
    if vf is None or vu is None or now < vf:
        return None
    tk, lim = a.get("ticker"), float(a.get("entry") or 0)
    over = now >= vu
    placed = (a.get("status") or "new") == "placed"
    out = {"alert_id": a.get("id"), "ticker": tk, "limit": lim, "window_over": bool(over), "last": px.get(tk),
           "status": a.get("status") or "new", "valid_until": iso(vu)}
    if not over:
        ex = extremes_since(tk, vf, w, part, hourly)
        out.update({"low": ex["low"], "high": ex["high"]})
        if a.get("entry_kind") == "open_next":
            out.update({"filled": None, "text": "Market order at the open: only your exchange knows the fill."})
        elif ex["low"] is None:
            out.update({"filled": None, "text": "No price data for the order's window yet."})
        elif ex["low"] < lim:
            out.update({"filled": True, "text": f"The low since the order went live is {ex['low']:.6g}, below the limit {lim:.6g}: it should have filled{_when(ex.get('low_at'))}."})
        else:
            out.update({"filled": False, "text": f"The low so far is {ex['low']:.6g}, never below the limit {lim:.6g}. Still working."})
        tgt = float(a.get("target") or 0)
        if out.get("filled") is False and tgt and ex["high"] is not None and ex["high"] >= tgt:
            out["withdraw"] = "Price already reached the target without filling the order: the move happened without you. Cancel the order."
        return out

    wl = window_lows(tk, vf, vu, w, part, hourly)
    out.update({"low": wl["win"], "low_after": wl["after"], "source": wl["source"]})
    ended = f"{str(iso(vu))[:16].replace('T', ' ')} UTC"
    complete = feed_at is None or feed_at >= vu          # a feed pass that started before the window ended can miss its last minutes
    if a.get("entry_kind") == "open_next":
        out.update({"filled": None, "action": "check", "text": "Market order at the open: only your exchange knows the fill."})
    elif wl["win"] is None:
        out.update({"filled": None, "action": "check", "text": f"The order's window ended {ended} and there is no price data for it. Check the order on Coinbase: if it is still open, cancel it."})
    elif wl["win"] < lim:
        out.update({"filled": True, "action": "record_fill",
                    "text": f"RECORD THE FILL: the low inside the order's window was {wl['win']:.6g}, below your limit {lim:.6g}{_when(wl['win_at'])}. It should have filled. Confirm on Coinbase and record it so the desk watches the position."})
    elif wl["win"] == lim:
        out.update({"filled": None, "action": "check",
                    "text": f"CHECK ON COINBASE: the low inside the order's window was exactly your limit {lim:.6g}, so it may have filled in part or not at all. Record whatever filled, then cancel what is still open."})
    elif not complete:
        out.update({"filled": False, "action": "check",
                    "text": f"The order's window ended {ended}. The low so far recorded for it is {wl['win']:.6g}, above your limit {lim:.6g}, but the price feed has not yet covered the window's last minutes. Open Coinbase now: if the order still shows as open it did not fill, so cancel it. The next scan confirms the low."})
    elif wl["after"] is not None and wl["after"] <= lim:
        out.update({"filled": False, "action": "check",
                    "text": f"CANCEL THIS ORDER NOW AND CHECK IT: inside its window (ended {ended}) the low was {wl['win']:.6g}, never down to your limit {lim:.6g}, so it did not fill in time. After the window price fell to {wl['after']:.6g}{_when(wl['after_at'])}, at or below your limit. If the order was still resting on Coinbase it may have filled late: if it did, record the fill and the desk manages it with the same stop and target; if not, cancel it."})
    else:
        out.update({"filled": False, "action": "cancel", "confirmed": True,
                    "text": (f"CANCEL THIS ORDER: confirmed unfilled. " if placed else "Expired unfilled. ")
                            + f"The lowest trade inside the order's window was {wl['win']:.6g}, never down to your limit {lim:.6g}"
                            + (f", and price has stayed above it since (low {wl['after']:.6g})" if wl["after"] is not None else "")
                            + f". The window ended {ended}. "
                            + ("Cancel the buy order on Coinbase so it cannot fill late outside the setup, then press \"I cancelled it\" on the dashboard."
                               if placed else "If you did place this order, cancel it on Coinbase.")})
    return out


def run(mode: str, data_dir, state_dir, out_dir, now=None) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(now).tz_localize(None) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    w, part, meta = D.load_daily(data_dir)
    hourly, uni = D.load_hourly(data_dir), D.load_universe(data_dir)
    health = D.health(w, meta, now)
    state = load_state(state_dir)
    settings = state["settings"]
    P = S.load_params(state["params"])
    ind = I.compute(w, D.load_fng(data_dir), min_dv=float((P.get("universe") or {}).get("min_dv") or I.MIN_DV))
    P["costs"]["maker_pct"], P["costs"]["taker_pct"] = float(settings.get("maker_fee_pct", P["costs"]["maker_pct"])), float(settings.get("taker_fee_pct", P["costs"]["taker_pct"]))
    P = apply_live(P, state["learning"].get("adjust"))
    paused = dict((state["learning"].get("adjust") or {}).get("paused") or {})
    fund = funds(state)
    gov = risk_governor(state, fund["capital"])
    px, asof = live_prices(w, part, hourly)
    mkt = market_now(ind, px)
    mkt["zone_blocked"] = mkt["zone"] in ((P.get("zones") or {}).get("blocked") or [])
    if mkt["zone_blocked"]:
        zr = ((P.get("zones") or {}).get("table") or {}).get(mkt["zone"]) or {}
        mkt["zone_note"] = (f"Danger zone: {mkt['zone_label']}. A+ setups here won {100 * zr['train']['win_rate']:.0f}% in training and {100 * zr['unseen']['win_rate']:.0f}% on unseen data, "
                            "so alerts are sent only for strategies with their own A+ record in this zone"
                            + (": " + ", ".join(f"{(S.all_specs(P).get(k.split('|')[0]) or {}).get('label', k.split('|')[0])} ({'core coins' if k.endswith('|core') else 'wider list'})" for k in sorted(((P.get("zones") or {}).get("exempt") or {}).get(mkt["zone"]) or {})) + "." if ((P.get("zones") or {}).get("exempt") or {}).get(mkt["zone"]) else " (none yet).")) if zr else "Danger zone."
    writes, notes, errors = [], list(health["notes"]), []
    scan_id = now.strftime("%Y%m%d-%H%M") + f"-{mode}"

    # ---- open positions
    positions, exit_alerts = [], []
    for tr in [t for t in state["trades"].values() if t.get("status") == "open"]:
        try:
            st = check_position(tr, ind, w, part, hourly, px, P, mode, now)
        except Exception as e:
            st = {"trade_id": tr.get("id"), "ticker": tr.get("ticker"), "code": "ERROR", "label": "CHECK BY HAND", "tone": "act", "detail": f"Could not evaluate this position ({e}). Check its stop and target on your exchange."}
            errors.append(f"position {tr.get('ticker')}: {e}"[:160])
        positions.append(st)
        if st["code"] in ("STOP_HIT", "TARGET_HIT", "EXIT_RULE", "TIME_STOP", "RAISE_STOP"):
            aid = f"{tr.get('id')}-{st['code']}-{now.strftime('%Y%m%d')}"
            if aid not in state["alerts"]:
                doc = {"id": aid, "type": "exit", "status": "new", "code": st["code"], "ticker": st["ticker"], "trade_id": tr.get("id"),
                       "label": st["label"], "detail": st["detail"], "last": st.get("last"), "new_stop": st.get("new_stop"),
                       "created_at": iso(now), "scan_id": scan_id}
                exit_alerts.append(doc)
                _write(out, writes, "set", "alerts", aid, doc)

    # ---- resting entry orders
    fill_checks, alert_updates = [], {}
    try:
        feed_at = pd.Timestamp(meta["generated_utc"]).tz_convert(None) if meta.get("generated_utc") else None
    except Exception:
        feed_at = None
    live_tickers = {t.get("ticker") for t in state["trades"].values() if t.get("status") == "open"}
    for a in state["alerts"].values():
        if a.get("type") != "entry" or a.get("status") not in ("new", "placed") or a.get("withdrawn"):
            continue
        fc = check_order(a, w, part, hourly, px, now, feed_at)
        if fc is None:
            continue
        if fc["window_over"]:
            # an order never marked as placed drops off a day after its window; a placed one stays until answered
            if a.get("status") != "placed" and now >= _naive(a["valid_until"]) + pd.Timedelta(hours=26):
                continue
            fill_checks.append(fc)
            exp = {"action": fc.get("action"), "confirmed": bool(fc.get("confirmed")), "low": fc.get("low"), "low_after": fc.get("low_after"),
                   "limit": fc["limit"], "text": fc["text"], "checked_at": iso(now)}
            if (a.get("expiry") or {}).get("text") != exp["text"]:
                _write(out, writes, "update", "alerts", a["id"], {"expiry": exp})
            alert_updates[a["id"]] = {"expiry": exp, "checked_at": iso(now)}
            continue
        fill_checks.append(fc)
        live_tickers.add(a.get("ticker"))
        last = px.get(a.get("ticker"))
        alert_updates[a["id"]] = {"last": last, "to_entry_pct": None if not last else round(100 * (float(a["entry"]) / last - 1), 2), "checked_at": iso(now),
                                  **({"withdrawn": True, "withdrawn_reason": fc["withdraw"]} if fc.get("withdraw") else {})}

    # ---- new setups (after the daily close only): every A+ setup is served, none held back
    entries, watch, near = [], [], {"coins": [], "more": 0, "waiting": []}
    day_bar = w["close"].index[-1]
    fresh = health["days_behind"] == 0
    liquid_now = ind["liquid"].iloc[-1]
    online = {tk for tk, u in uni.items() if u.get("status") == "online" and not u.get("peg")} if uni else set(liquid_now.index)
    scanned = [tk for tk in liquid_now.index[liquid_now.values] if tk in online]
    core_n = len([tk for tk in scanned if bool(ind["is_core"][tk].iloc[-1])])
    universe = {"scanned": len(scanned), "core": core_n, "wide": len(scanned) - core_n, "listed": len(online),
                "min_dollar_vol": float((P.get("universe") or {}).get("min_dv") or I.MIN_DV), "core_dollar_vol": I.CORE_DV}
    min_week = int((P.get("pace") or {}).get("min_week", 5))
    vf = day_bar + pd.Timedelta(days=1)
    wk0 = vf.normalize() - pd.Timedelta(days=vf.weekday())                      # Monday 00:00 UTC of the order's week
    served = {a.get("id") for a in state["alerts"].values() if a.get("type") == "entry" and a.get("grade") == "A+" and not a.get("topup")
              and (_naive(a.get("valid_from")) or pd.Timestamp(0)) >= wk0 and (_naive(a.get("valid_from")) or pd.Timestamp(0)) < wk0 + pd.Timedelta(days=7)}
    # a check run does the daily-close work too if the close scan for this bar never happened (missed run, late feed)
    book = book_id(P)
    legacy = f"|{P.get('rev')}|{P.get('calibrated_at')}"          # stamp of scans made before 2026-11-17 (code fingerprint|rev|calibrated)
    closed_for_bar = any(sc.get("mode") == "close" and sc.get("signal_date") == str(day_bar.date()) and not sc.get("skipped_setups")
                         and (sc.get("book") == book or str(sc.get("book") or "").endswith(legacy)) for sc in state["scans"].values())
    catch_up = mode == "check" and fresh and not closed_for_bar
    skipped_setups = False
    if catch_up:
        mode = "close"
        notes.append(f"Catch-up: no daily-close scan had run for {day_bar.date()} with the current strategy book, so this run looked for new setups too. Setups whose entry price already traded were left out.")
    if mode == "close":
        if not fresh:
            skipped_setups = True
            notes.append("Prices are behind, so no new setups were generated from stale bars. The next run catches up as soon as the feed does.")
        else:
            found, n2 = find_entries(ind, P, uni, mkt["condition"], paused)
            notes += n2
            for e in found:                                    # re-grade tickets that are still live under the book in force now
                aid0 = f"{(day_bar + pd.Timedelta(days=1)).strftime('%Y%m%d')}-{e['ticker']}-{e['strategy']}"
                old = state["alerts"].get(aid0)
                if not old or old.get("status") not in ("new", "placed") or old.get("withdrawn"):
                    continue
                was = float(old.get("win_pct") or 0)
                if e["grade"] != "A+":
                    why = ("Danger zone: this strategy has no A+ record of its own in the current zone." if e.get("zone_blocked") else "It no longer grades A+ under the current strategy book.")
                    upd = {"withdrawn": True, "withdrawn_reason": f"Downgraded from A+ on {iso(now)[:10]}: {why} Cancel the order if it has not filled.",
                           "win_pct": e["win_pct"], "regraded_at": iso(now)}
                elif abs(e["win_pct"] - was) >= 1:
                    upd = {"win_pct": e["win_pct"], "score": e["score"], "cell_n": e.get("cell_n"), "zone_cell": bool(e.get("zone_cell")), "regraded_at": iso(now),
                           "note": (f"Re-graded on {iso(now)[:10]}: win estimate {e['win_pct']:.0f}% (was {was:.0f}%)"
                                    + (f", from this strategy's own record in the danger zone ({e['cell_n']} trades on {e.get('cell_days')} days)." if e.get("zone_cell") else " under the current strategy book."))}
                else:
                    continue
                alert_updates.setdefault(aid0, {}).update(upd)
                _write(out, writes, "update", "alerts", aid0, upd)          # stored on the ticket itself, so later scans cannot bring the old odds back
                notes.append(f"{e['ticker']}: " + (upd.get("withdrawn_reason") or upd["note"]))
            # a live ticket from this bar that the book in force no longer flags at all (a revised rule, a new filter)
            # is withdrawn too: it must never stay open just because nothing re-issued it
            specs_now = S.all_specs(P)
            flagged = {(e["ticker"], e["strategy"]) for e in found if e["grade"] == "A+"}
            flagged |= {(e["ticker"], sid) for e in found if e["grade"] == "A+" for sid, sp in specs_now.items() if sp["label"] in (e.get("also_by") or [])}
            bar_tag = (day_bar + pd.Timedelta(days=1)).strftime('%Y%m%d')
            for aid0, old in state["alerts"].items():
                if (old.get("type") != "entry" or not str(aid0).startswith(bar_tag + "-") or old.get("status") not in ("new", "placed")
                        or old.get("withdrawn") or old.get("grade") != "A+"
                        or (alert_updates.get(aid0) or {}).get("withdrawn") or (alert_updates.get(aid0) or {}).get("regraded_at")):
                    continue
                if (old.get("ticker"), old.get("strategy")) in flagged:
                    continue
                upd = {"withdrawn": True, "regraded_at": iso(now),
                       "withdrawn_reason": f"Withdrawn on {iso(now)[:10]}: the revised {old.get('strategy_label') or old.get('strategy')} rule no longer flags {old.get('ticker')} (see the journal). Cancel the order if it has not filled; if it already filled, keep the stop and target and the desk manages it as usual."}
                alert_updates.setdefault(aid0, {}).update(upd)
                _write(out, writes, "update", "alerts", aid0, upd)
                notes.append(f"{old.get('ticker')}: " + upd["withdrawn_reason"])
            if catch_up or now >= day_bar + pd.Timedelta(days=1, hours=2):       # late: drop what the market already took
                late = []
                for e in found:
                    ex = extremes_since(e["ticker"], day_bar + pd.Timedelta(days=1), w, part, hourly)
                    gone = e["entry_kind"] == "open_next" or (ex["low"] is not None and ex["low"] < e["entry"]) or (ex["high"] is not None and ex["high"] >= e["target"])
                    if gone and e["grade"] == "A+":
                        late.append(e["ticker"])
                    e["_gone"] = gone
                found = [e for e in found if not e.pop("_gone")]
                if late:
                    notes.append("Too late for: " + ", ".join(late) + " (the entry or target already traded before this run).")
            try:
                near = radar(ind, P, uni, mkt["condition"], paused)
            except Exception as e:
                errors.append(f"radar: {e}"[:160])
            near["coins"] = [r for r in near["coins"] if r["ticker"] not in {e["ticker"] for e in found if e["grade"] == "A+"}]
            slots = max(0, int(settings.get("max_open_positions", P["risk"].get("max_open_positions", 4))) - fund["open_count"])
            min_order = float(P["risk"].get("min_order_usd", 25.0))
            cash, rank = fund["cash"], 0
            target_now = min(min_week, vf.weekday() + 1)
            aplus = [e for e in found if e["grade"] == "A+" and e["ticker"] not in live_tickers]
            tu = P.get("topup") or {}
            deficit = max(0, target_now - len(served | {f"{vf.strftime('%Y%m%d')}-{e['ticker']}-{e['strategy']}" for e in aplus}))
            topups = [e for e in found if e["grade"] == "A" and e["ticker"] not in live_tickers and e["win_pct"] >= 100 * float(tu.get("floor") or 1)][:deficit] if tu.get("ok") else []
            for e in found:
                if e["ticker"] in live_tickers or e in aplus or e in topups:
                    continue
                if e["grade"] == "A":
                    watch.append({k: e.get(k) for k in ("ticker", "strategy_label", "score", "win_pct", "entry", "stop", "target", "rr", "hold_max_days", "cell_n", "tier", "zone_blocked")})
            for e in aplus + topups:
                rank += 1
                is_top = e in topups
                funded = slots > 0 and cash >= min_order
                sz = size_position(e["entry"], e["stop"], fund["capital"], cash if funded else 0.0, gov["risk_pct"], settings, min_order, e.get("min_size"))
                funded = funded and sz["total_cost"] >= min_order
                if funded:
                    cash -= sz["total_cost"]
                    slots -= 1
                vu = vf + pd.Timedelta(days=e["valid_days"])
                aid = f"{vf.strftime('%Y%m%d')}-{e['ticker']}-{e['strategy']}"
                if catch_up and aid in state["alerts"]:
                    rank -= 1
                    continue                      # already served by an earlier scan: its ticket and order check carry on as they are
                order = (f"Buy {sz['qty']:g} {e['ticker']} at market now (about ${sz['total_cost']:,.2f})" if e["entry_kind"] == "open_next" else
                         f"Buy limit {sz['qty']:g} {e['ticker']} at {e['entry']:.6g} (about ${sz['total_cost']:,.2f}), post-only if offered; cancel if unfilled by {iso(vu)[:16].replace('T', ' ')} UTC")
                tier_txt = "" if e["tier"] == "core" else " This coin is on the wider list (thinner trading): the record shown is for wider-list coins, tested with harsher fills. Use a limit order only and expect a wider spread."
                why = [f"{e['strategy_label']}: {e['about']}",
                       f"Market now: {mkt['label']}. In this condition the setup won {e['win_pct']:.0f}% of {e['cell_n']} tested trades after fees"
                       + (f" ({100 * e['cell_train']:.0f}% in the training years, {100 * e['cell_oos']:.0f}% on unseen data)." if e.get("cell_train") is not None and e.get("cell_oos") is not None else ".") + tier_txt,
                       f"Coin: RSI(2) {e['rsi2']}, {e['vs_50d_pct']:+.1f}% vs its 50-day average, 90-day strength rank {e['rs_pct']}/100, daily range about {e['atr_pct']}%.",
                       f"After fees: +{e['net_win_pct']:.1f}% at the target, {e['net_loss_pct']:.1f}% at the stop (reward to risk {e['rr']:.2f})."]
                if e.get("zone_cell"):
                    why.insert(1, f"Danger zone ({I.ZONE_LABEL[e['zone']]}): most A+ setups lose here, but this strategy on {'core' if e['tier'] == 'core' else 'wider-list'} coins has its own record in the zone: "
                                  f"{e['win_pct']:.0f}% of {e['cell_n']} trades on {e.get('cell_days')} days. The win odds shown are for this zone.")
                if e.get("confluence", 1) > 1:
                    cf = {r["agree"]: r for r in (P.get("confluence") or [])}
                    one, many = cf.get("1 strategy"), cf.get("2 strategies" if e["confluence"] == 2 else "3 or more")
                    why.insert(1, f"Confirmed by {e['confluence']} A+ strategies (also {', '.join(e['also_by'])})."
                               + (f" On unseen data, setups with this much agreement won {100 * many['win_rate']:.0f}% of {many['trades']}, against {100 * one['win_rate']:.0f}% with one strategy alone." if one and many else ""))
                if is_top:
                    why.insert(1, f"Top-up: grade A, not A+. Sent because the week is behind its pace of {min_week} setups and this rule passed the top-up test on unseen data.")
                doc = {**e, **sz, "id": aid, "type": "entry", "status": "new", "rank": rank, "also_qualified": not funded, "funded": funded, "topup": is_top,
                       "capacity_note": None if funded else "A+ and fully valid, but the account has no free slot or cash for it right now. The size shown is what it would be with free cash: take it only if you skip a higher-ranked one or a position closes first.",
                       "valid_from": iso(vf), "valid_until": iso(vu), "created_at": iso(now), "scan_id": scan_id, "order": order, "why": why,
                       "after_fill": f"Once filled: place a stop-loss (stop-market) sell at {e['stop']:.6g} and a limit sell at {e['target']:.6g} for the full size. {e['exit_signal']} Out by the end of day {e['hold_max_days']} regardless.",
                       "risk_note": gov["reason"], "news": {"verdict": "unchecked"}}
                doc.pop("min_size", None)
                entries.append(doc)
                if aid not in state["alerts"]:
                    _write(out, writes, "set", "alerts", aid, doc)
    served |= {e["id"] for e in entries if not e["topup"]}
    dow = (now.normalize() - wk0).days if now >= wk0 else 0
    target_now = min(min_week, max(1, min(7, dow + 1)))
    pace = {"week_start": str(wk0.date()), "min_week": min_week, "served": len(served), "target_by_now": target_now,
            "behind": len(served) < target_now, "topups_on": bool((P.get("topup") or {}).get("ok")),
            "note": None if len(served) >= target_now else ("Behind the weekly pace. Top-ups are off because lower-grade setups lost money in testing, so the gap is closed only by finding more genuine A+ setups: "
                                                           "the daily research run is pointed at the current market condition.")}
    counts = {"aplus": len([e for e in entries if not e["topup"]]), "funded": len([e for e in entries if e["funded"]]), "also_qualified": len([e for e in entries if e["also_qualified"]]),
              "topups": len([e for e in entries if e["topup"]]), "watchlist": len(watch), "radar": len(near["coins"]) + near["more"], "exit_alerts": len(exit_alerts), "open_positions": len(positions),
              "orders_checked": len(fill_checks), "fills_likely": len([f for f in fill_checks if f.get("filled")]),
              "orders_to_cancel": len([f for f in fill_checks if f["status"] == "placed" and (f.get("action") in ("cancel", "check") and f.get("window_over") or f.get("withdraw"))])}
    active = [sid for sid, r in P["strategies"].items() if r.get("enabled") and sid not in paused and (not r.get("market_conditions") or mkt["condition"] in r["market_conditions"])]
    scan = {"id": scan_id, "mode": mode, "started_at": iso(now), "status": "ok" if not errors else "partial", "data": {**health, "prices_as_of": asof, "tradable": len(scanned)},
            "market": mkt, "counts": counts, "positions": positions, "fill_checks": fill_checks, "alert_updates": alert_updates, "watchlist": watch[:20],
            "radar": near, "pace": pace, "universe": universe, "catch_up": catch_up, "skipped_setups": skipped_setups, "book": book,
            "notes": notes, "errors": errors, "risk": gov, "funds": fund, "strategies_active": active, "strategies_paused": paused,
            "signal_date": str(day_bar.date()), "engine": version.info(), "params_version": P.get("rev")}
    scan["headline"] = headline(scan, entries)
    scan["digest"] = digest(scan, entries, exit_alerts)
    _write(out, writes, "set", "scans", scan_id, scan)
    (out / "scan.json").write_text(json.dumps(clean({"scan": scan, "entries": entries, "exit_alerts": exit_alerts}), indent=1, default=str))
    (out / "writes.json").write_text(json.dumps(writes, indent=1))
    (out / "digest.md").write_text(scan["digest"])
    (out / "headline.txt").write_text(scan["headline"])
    return {"scan_id": scan_id, "status": scan["status"], "headline": scan["headline"], "writes": len(writes), "entries": len(entries)}


def headline(scan, entries) -> str:
    c, m = scan["counts"], scan["market"]
    bits = []
    if scan["data"]["status"] != "ok":
        bits.append("⚠ prices behind")
    acts = [p for p in scan["positions"] if p["code"] not in ("HOLD",)]
    if c["aplus"]:
        names = [e["ticker"] for e in entries if not e["topup"]]
        bits.append(f"{c['aplus']} A+ setup{'s' if c['aplus'] != 1 else ''} ({', '.join(names[:8])}{' +' + str(len(names) - 8) + ' more' if len(names) > 8 else ''})")
    if c.get("topups"):
        bits.append(f"{c['topups']} top-up")
    if acts:
        bits.append(", ".join(f"{p['ticker']} {p['label'].split(':')[0]}" for p in acts))
    canc = [f["ticker"] for f in scan["fill_checks"] if f["status"] == "placed" and (f.get("action") == "cancel" or f.get("withdraw"))]
    chk = [f["ticker"] for f in scan["fill_checks"] if f["status"] == "placed" and f.get("action") == "check" and f.get("window_over")]
    if canc:
        bits.append(f"CANCEL the {', '.join(canc)} buy order{'s' if len(canc) != 1 else ''} (unfilled)")
    if chk:
        bits.append(f"check the {', '.join(chk)} order on Coinbase")
    if c["fills_likely"]:
        bits.append(f"{c['fills_likely']} order{'s' if c['fills_likely'] != 1 else ''} likely filled")
    if not [b for b in bits if not b.startswith("⚠")]:
        if scan["mode"] == "close":
            bits.append(("Danger zone: no new A+ setups" if scan["market"].get("zone_blocked") else f"No new A+ setups from {scan['universe']['scanned']} coins") + (f", {c['radar']} close" if c.get("radar") and not scan["market"].get("zone_blocked") else ""))
        else:
            bits.append("No action needed")
    return " · ".join(b for b in bits if b) + f" · market {m['label']}"


def digest(scan, entries, exit_alerts) -> str:
    m, c = scan["market"], scan["counts"]
    L = [f"CRYPTO SWING DESK | {'DAILY-CLOSE SCAN' if scan['mode'] == 'close' else 'POSITION CHECK'} | {scan['started_at'][:16].replace('T', ' ')} UTC", ""]
    L.append(f"Market: {m['label']}. BTC ${m['btc']:,.0f} ({m['btc_vs_200d_pct']:+.1f}% vs its 200-day, {m['btc_7d_pct']:+.1f}% in 7 days); "
             f"{m['breadth_above_50d_pct']}% of liquid coins above their 50-day" + (f"; Fear & Greed {m['fear_greed']:.0f}" if m.get("fear_greed") is not None else "") + ".")
    L.append(scan["risk"]["reason"])
    for n in scan["notes"]:
        L.append("Note: " + n)
    acts = [p for p in scan["positions"] if p["code"] != "HOLD"]
    if acts:
        L += ["", "ACTION ON OPEN POSITIONS"]
        for p in acts:
            L.append(f"- {p['ticker']}: {p['label']}. {p['detail']}")
    fl = scan["fill_checks"]                      # every resting order, filled or not: an order still working is part of the picture
    if fl:
        todo = [f for f in fl if f["status"] == "placed" and (f.get("withdraw") or (f.get("window_over") and f.get("action") in ("cancel", "check")))]
        if todo:
            L += ["", "ORDERS TO CANCEL OR CHECK ON COINBASE"]
            for f in todo:
                L.append(f"- {f['ticker']} buy limit {f['limit']:.6g}: {f.get('withdraw') or f['text']}")
        rest = [f for f in fl if f not in todo]
        if rest:
            L += ["", "YOUR OTHER ORDERS"]
            for f in rest:
                L.append(f"- {f['ticker']}: {f.get('withdraw') or f['text']}")
    if scan["mode"] == "close":
        u, pc, rd = scan["universe"], scan["pace"], scan["radar"]
        L += ["", f"NEW A+ SETUPS ({len(entries)}) from {u['scanned']} Coinbase coins scanned ({u['core']} core, {u['wide']} wider list)"]
        if not entries:
            L.append(scan["market"].get("zone_note") or "No new ones. The filter only passes setups whose record in this market condition is strong in training and on unseen data.")
        for e in entries:
            L += [f"- {e['ticker']} | {e['strategy_label']} | {'TOP-UP (grade A)' if e['topup'] else 'A+'}{'' if e['tier'] == 'core' else ' | wider list'}{'' if e.get('confluence', 1) < 2 else ' | ' + str(e['confluence']) + ' strategies agree'} | score {e['score']} | win odds {e['win_pct']:.0f}%"
                  + ("" if e["funded"] else " | no free slot or cash right now"),
                  f"    Order: {e['order']}",
                  f"    Entry {e['entry']:.6g} | Stop {e['stop']:.6g} ({e['stop_pct']:+.1f}%) | Target {e['target']:.6g} ({e['target_pct']:+.1f}%) | reward:risk {e['rr']:.2f} after fees",
                  f"    Size {e['qty']:g} {e['ticker']} = ${e['total_cost']:,.2f} ({e['pct_of_capital']}% of capital), risk ${e['risk_usd']:,.2f}",
                  f"    Hold up to {e['hold_max_days']} days. Order good from {e['valid_from'][:16].replace('T', ' ')} to {e['valid_until'][:16].replace('T', ' ')} UTC.",
                  f"    {e['after_fill']}"]
            if (e.get("news") or {}).get("verdict") not in (None, "unchecked", "clean"):
                L.append(f"    News: {e['news']['verdict']}: {e['news'].get('reason', '')}")
        L.append(f"Week so far: {pc['served']} A+ setup{'s' if pc['served'] != 1 else ''} served, pace {pc['target_by_now']} by today, minimum {pc['min_week']} a week." + (" " + pc["note"] if pc.get("note") else ""))
        if rd["coins"]:
            L.append("On the radar (one condition short of an A+ signal): " + "; ".join(f"{r['ticker']} ({r['strategy_label']}: needs {r['needs']}{', ' + r['now'] if r['now'] else ''})" for r in rd["coins"][:8])
                     + (f" and {rd['more'] + max(0, len(rd['coins']) - 8)} more" if rd["more"] or len(rd["coins"]) > 8 else ""))
        for x in rd["waiting"][:6]:
            L.append(f"Waiting on the market: {x['strategy_label']} needs {x['needs']}" + (f" ({x['now']})" if x["now"] else "") + (f"; {x['coins_ready']} coin{'s' if x['coins_ready'] != 1 else ''} otherwise ready" if x.get("coins_ready") else ""))
        if scan["watchlist"]:
            L.append("Watchlist (grade A, not sent as alerts): " + ", ".join(f"{x['ticker']} ({x['win_pct']:.0f}%)" for x in scan["watchlist"]))
    hold = [p for p in scan["positions"] if p["code"] == "HOLD"]
    if hold:
        L += ["", "HOLDING"]
        for p in hold:
            L.append(f"- {p['ticker']}: {p.get('pct', 0):+.1f}% (${p.get('unrealized', 0):+,.2f}). {p['detail']}")
    L += ["", "Model signals from backtested rules, not financial advice. Win odds are estimates and can be wrong. You place every order yourself."]
    return "\n".join(L)


NEWS_VERDICTS = ("clean", "caution", "avoid", "unclear")


def apply_news(out_dir, verdicts: list[str], issues: list[str], notes: list[str] | None = None):
    """Fold the news check into the scan's documents, digest and headline.

    issues = real run problems (something is wrong, late, missing or unsaved): red banner, headline flag, top of digest.
    notes  = how the run went (an extra search, a retry that worked): kept at the foot of the digest, never an alarm."""
    out = pathlib.Path(out_dir)
    js = json.loads((out / "scan.json").read_text())
    scan, entries, exits = js["scan"], js["entries"], js.get("exit_alerts", [])
    v = {}
    for s in verdicts:
        tk, _, rest = s.partition("=")
        verdict, _, reason = rest.partition(":")
        verdict = verdict.strip().lower()
        if verdict not in NEWS_VERDICTS:
            verdict, reason = "unclear", (rest.strip() or "no verdict given")
        v[tk.strip().upper()] = {"verdict": verdict, "reason": reason.strip(), "checked_at": iso(pd.Timestamp.now(tz="UTC"))}
    for e in entries:
        n = v.get(e["ticker"])
        if not n:
            continue
        e["news"] = n
        if n["verdict"] == "avoid":
            e.update({"withdrawn": True, "withdrawn_reason": "News check: " + (n["reason"] or "avoid"), "also_qualified": True, "funded": False})
        elif n["verdict"] == "caution":
            e["score"] = round(e["score"] - 8, 1)
    live = sorted([e for e in entries if not e.get("withdrawn")], key=lambda e: -e["score"])
    for i, e in enumerate(live):
        e["rank"] = i + 1
    scan["counts"]["aplus"] = len([e for e in entries if not e.get("topup") and not e.get("withdrawn")])
    scan["news_checked"] = sorted(set(scan.get("news_checked") or []) | set(v))
    if issues:
        scan["run_issues"] = list(scan.get("run_issues", [])) + [i for i in issues if i not in scan.get("run_issues", [])]
    if notes:
        scan["run_notes"] = list(scan.get("run_notes", [])) + [n for n in notes if n not in scan.get("run_notes", [])]
    scan["headline"] = ("⚠ run problem · " if scan.get("run_issues") else "") + headline(scan, [e for e in entries if not e.get("withdrawn")])
    scan["digest"] = ("\n".join("⚠ RUN PROBLEM: " + i for i in scan.get("run_issues", [])) + "\n\n" if scan.get("run_issues") else "") + digest(scan, [e for e in entries if not e.get("withdrawn")], exits)
    wd = [e for e in entries if e.get("withdrawn")]
    if wd:
        scan["digest"] += "\n\nWithdrawn after the news check: " + "; ".join(f"{e['ticker']} ({e['withdrawn_reason']})" for e in wd)
    if scan.get("run_notes"):
        scan["digest"] += "\n\nHow this run went (nothing to act on): " + " ".join(scan["run_notes"])
    for e in entries:
        f = out / "docs" / "alerts" / f"{e['id']}.json"
        if f.exists():
            f.write_text(json.dumps(clean(e), default=str))
    (out / "docs" / "scans" / f"{scan['id']}.json").write_text(json.dumps(clean(scan), default=str))
    (out / "scan.json").write_text(json.dumps(clean({"scan": scan, "entries": entries, "exit_alerts": exits}), indent=1, default=str))
    (out / "digest.md").write_text(scan["digest"])
    (out / "headline.txt").write_text(scan["headline"])
    return scan["headline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "news"])
    ap.add_argument("--mode", default="close", choices=["close", "check"])
    ap.add_argument("--data")
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--now", default=None)
    ap.add_argument("--issue", action="append", default=[], help="a real run problem: the result is wrong, late, missing or unsaved")
    ap.add_argument("--note", action="append", default=[], help="how the run went (extra searches, a retry that worked): never shown as a problem")
    ap.add_argument("verdicts", nargs="*")
    a = ap.parse_args()
    if a.cmd == "news":
        print(apply_news(a.out, a.verdicts, a.issue, a.note))
        return
    try:
        print(json.dumps(run(a.mode, a.data, a.state, a.out, a.now)))
    except Exception as e:
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "failure.json").write_text(json.dumps({"error": str(e), "trace": traceback.format_exc()}, indent=1))
        print(json.dumps({"status": "failed", "error": str(e)[:300]}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
