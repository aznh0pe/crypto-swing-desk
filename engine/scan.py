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
import datetime as dt
import json
import math
import pathlib
import traceback

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as I
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


def extremes_since(tk: str, since: pd.Timestamp, w, part, hourly) -> dict:
    """Lowest low and highest high since `since` (UTC), from hourly bars where they reach back far enough."""
    lo = hi = None
    lo_at = None
    if hourly is not None and tk in hourly["low"] and hourly["low"].index[0] <= since:
        seg_l, seg_h = hourly["low"][tk][hourly["low"].index >= since.floor("h")].dropna(), hourly["high"][tk][hourly["high"].index >= since.floor("h")].dropna()
        if len(seg_l):
            lo, hi, lo_at = float(seg_l.min()), float(seg_h.max()), iso(seg_l.idxmin())
    else:
        d0 = since.normalize()
        seg_l, seg_h = w["low"][tk][w["low"].index >= d0].dropna() if tk in w["low"] else pd.Series(dtype=float), w["high"][tk][w["high"].index >= d0].dropna() if tk in w["high"] else pd.Series(dtype=float)
        vals_l, vals_h = list(seg_l.values), list(seg_h.values)
        if part is not None and tk in part["low"]:
            vals_l.append(float(part["low"][tk]))
            vals_h.append(float(part["high"][tk]))
        if vals_l:
            lo, hi = float(min(vals_l)), float(max(vals_h))
    return {"low": lo, "high": hi, "low_at": lo_at}


def market_now(ind) -> dict:
    mk = ind["market"].iloc[-1]
    cond = I.condition_series(ind["market"]).iloc[-1]
    f = lambda k, n=4: None if not np.isfinite(mk.get(k, np.nan)) else round(float(mk[k]), n)
    return {"condition": cond, "label": I.CONDITION_LABEL.get(cond, "Unknown"), "bull": bool(mk["bull"] >= 0.5), "stress": bool(mk["stress"] >= 0.5),
            "btc": f("btc", 2), "btc_vs_200d_pct": None if f("btc_dist200") is None else round(100 * f("btc_dist200"), 1),
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
    specs, table, costs, G = S.all_specs(P), P.get("grade_table") or {}, P["costs"], P["grading"]
    maker, taker = costs["maker_pct"] / 100, costs["taker_pct"] / 100
    best, notes = {}, []
    for sid, plan in S.all_plans(ind, P).items():
        if sid in paused:
            notes.append(f"{specs[sid]['label']} is paused by the daily review: {paused[sid]}")
            continue
        row = plan["mask"].iloc[-1]
        for tk in row.index[row.values]:
            u = uni.get(tk) or {}
            if uni and not u.get("tradable"):
                continue
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
            cell = (table.get(sid) or {}).get(cond) or {}
            p, grade = cell.get("p"), cell.get("grade", "B")
            if p is None:
                continue
            fee_in = taker if market else maker
            net_win = (tg * (1 - maker)) / (entry * (1 + fee_in)) - 1
            net_loss = (stop * (1 - taker)) / (entry * (1 + fee_in)) - 1
            rr = net_win / abs(net_loss) if net_loss < 0 else 0.0
            rs = float(ind["rs_pct"][tk].iloc[-1]) if np.isfinite(ind["rs_pct"][tk].iloc[-1]) else 0.5
            score = round(100 * p + 2.0 * min(4.0, (tg - entry) / max(entry - stop, 1e-12)) + 3.0 * rs, 1)
            sp = specs[sid]
            e = {"ticker": tk, "product": u.get("product") or f"{tk}-USD", "strategy": sid, "strategy_label": sp["label"], "family": sp.get("family"),
                 "about": sp.get("about"), "grade": grade, "score": score, "win_pct": round(100 * p, 1), "cell_n": cell.get("n"),
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
            if tk not in best or (e["grade"] == "A+", e["score"]) > (best[tk]["grade"] == "A+", best[tk]["score"]):
                best[tk] = e
    out = sorted(best.values(), key=lambda e: (e["grade"] != "A+", -e["score"]))
    return out, notes


def check_position(tr: dict, ind, w, part, hourly, px: dict, P: dict, mode: str, now: pd.Timestamp) -> dict:
    """What to do now with one open position."""
    tk = tr.get("ticker")
    entry, qty = float(tr.get("entry_price") or 0), float(tr.get("qty") or 0)
    stop, tgt = float(tr.get("stop") or 0), float(tr.get("target") or 0)
    hold = int(tr.get("hold_max_days") or 7)
    at = pd.Timestamp(tr.get("entry_at") or tr.get("entry_date") or now).tz_localize(None) if pd.Timestamp(tr.get("entry_at") or tr.get("entry_date") or now).tzinfo is None else pd.Timestamp(tr.get("entry_at")).tz_convert(None)
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


def check_order(a: dict, w, part, hourly, px: dict, now: pd.Timestamp) -> dict | None:
    """Did a resting entry order trade? Same rule as the backtest: the low must trade below the limit."""
    try:
        vf, vu = pd.Timestamp(a["valid_from"]).tz_convert(None), pd.Timestamp(a["valid_until"]).tz_convert(None)
    except Exception:
        return None
    if now < vf:
        return None
    tk, lim = a.get("ticker"), float(a.get("entry") or 0)
    ex = extremes_since(tk, vf, w, part, hourly)
    over = now >= vu
    out = {"alert_id": a.get("id"), "ticker": tk, "limit": lim, "low": ex["low"], "high": ex["high"], "window_over": bool(over),
           "last": px.get(tk), "status": a.get("status") or "new"}
    if a.get("entry_kind") == "open_next":
        out.update({"filled": None, "text": "Market order at the open: only your exchange knows the fill."})
    elif ex["low"] is None:
        out.update({"filled": None, "text": "No price data for the order's window yet."})
    elif ex["low"] < lim:
        out.update({"filled": True, "text": f"The low since the order went live is {ex['low']:.6g}, below the limit {lim:.6g}: it should have filled" + (f" (around {ex['low_at'][:16].replace('T', ' ')} UTC)." if ex.get("low_at") else ".")})
    else:
        out.update({"filled": False, "text": (f"The low was {ex['low']:.6g}, never below the limit {lim:.6g}. " + ("The order's day is over: it expired unfilled." if over else "Still working."))})
    tgt, stp = float(a.get("target") or 0), float(a.get("stop") or 0)
    if not over and out.get("filled") is False and tgt and ex["high"] is not None and ex["high"] >= tgt:
        out["withdraw"] = "Price already reached the target without filling the order: the move happened without you. Cancel the order."
    return out


def run(mode: str, data_dir, state_dir, out_dir, now=None) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(now).tz_localize(None) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    w, part, meta = D.load_daily(data_dir)
    hourly, uni = D.load_hourly(data_dir), D.load_universe(data_dir)
    ind = I.compute(w, D.load_fng(data_dir))
    health = D.health(w, meta, now)
    state = load_state(state_dir)
    settings = state["settings"]
    P = S.load_params(state["params"])
    P["costs"]["maker_pct"], P["costs"]["taker_pct"] = float(settings.get("maker_fee_pct", P["costs"]["maker_pct"])), float(settings.get("taker_fee_pct", P["costs"]["taker_pct"]))
    P = apply_live(P, state["learning"].get("adjust"))
    paused = dict((state["learning"].get("adjust") or {}).get("paused") or {})
    fund = funds(state)
    gov = risk_governor(state, fund["capital"])
    mkt = market_now(ind)
    px, asof = live_prices(w, part, hourly)
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
    live_tickers = {t.get("ticker") for t in state["trades"].values() if t.get("status") == "open"}
    for a in state["alerts"].values():
        if a.get("type") != "entry" or a.get("status") not in ("new", "placed") or a.get("withdrawn"):
            continue
        fc = check_order(a, w, part, hourly, px, now)
        if fc is None:
            continue
        fill_checks.append(fc)
        if not fc["window_over"]:
            live_tickers.add(a.get("ticker"))
            last = px.get(a.get("ticker"))
            alert_updates[a["id"]] = {"last": last, "to_entry_pct": None if not last else round(100 * (float(a["entry"]) / last - 1), 2), "checked_at": iso(now),
                                      **({"withdrawn": True, "withdrawn_reason": fc["withdraw"]} if fc.get("withdraw") else {})}

    # ---- new setups (after the daily close only)
    entries, watch = [], []
    day_bar = w["close"].index[-1]
    fresh = health["days_behind"] == 0
    if mode == "close":
        if not fresh:
            notes.append("Prices are behind, so no new setups were generated from stale bars.")
        else:
            found, n2 = find_entries(ind, P, uni, mkt["condition"], paused)
            notes += n2
            vf = day_bar + pd.Timedelta(days=1)
            slots = max(0, int(settings.get("max_open_positions", 4)) - fund["open_count"])
            cash, rank = fund["cash"], 0
            for e in found:
                if e["ticker"] in live_tickers:
                    continue
                if e["grade"] != "A+":
                    if e["grade"] == "A":
                        watch.append({k: e[k] for k in ("ticker", "strategy_label", "score", "win_pct", "entry", "stop", "target", "rr", "hold_max_days", "cell_n")})
                    continue
                rank += 1
                sz = size_position(e["entry"], e["stop"], fund["capital"], cash, gov["risk_pct"], settings, P["risk"].get("min_order_usd", 25.0))
                featured = rank <= int(P["risk"].get("max_per_day", 3)) and rank <= max(slots, 0) and sz["total_cost"] >= P["risk"].get("min_order_usd", 25.0)
                if featured:
                    cash -= sz["total_cost"]
                vu = vf + pd.Timedelta(days=e["valid_days"])
                aid = f"{vf.strftime('%Y%m%d')}-{e['ticker']}-{e['strategy']}"
                order = (f"Buy {sz['qty']:g} {e['ticker']} at market now (about ${sz['total_cost']:,.2f})" if e["entry_kind"] == "open_next" else
                         f"Buy limit {sz['qty']:g} {e['ticker']} at {e['entry']:.6g} (about ${sz['total_cost']:,.2f}), post-only if offered; cancel if unfilled by {iso(vu)[:16].replace('T', ' ')} UTC")
                why = [f"{e['strategy_label']}: {e['about']}",
                       f"Market now: {mkt['label']}. In this condition the setup won {e['win_pct']:.0f}% of {e['cell_n']} tested trades after fees"
                       + (f" ({100 * e['cell_train']:.0f}% in the training years, {100 * e['cell_oos']:.0f}% on unseen data)." if e.get("cell_train") is not None and e.get("cell_oos") is not None else "."),
                       f"Coin: RSI(2) {e['rsi2']}, {e['vs_50d_pct']:+.1f}% vs its 50-day average, 90-day strength rank {e['rs_pct']}/100, daily range about {e['atr_pct']}%.",
                       f"After fees: +{e['net_win_pct']:.1f}% at the target, {e['net_loss_pct']:.1f}% at the stop (reward to risk {e['rr']:.2f})."]
                doc = {**e, **sz, "id": aid, "type": "entry", "status": "new", "rank": rank, "also_qualified": not featured,
                       "capacity_note": None if featured else ("No free slot or cash for it today; listed so you can swap into it." if rank > slots or sz["total_cost"] < 25 else "Beyond today's top setups."),
                       "valid_from": iso(vf), "valid_until": iso(vu), "created_at": iso(now), "scan_id": scan_id, "order": order, "why": why,
                       "after_fill": f"Once filled: place a stop-loss (stop-market) sell at {e['stop']:.6g} and a limit sell at {e['target']:.6g} for the full size. {e['exit_signal']} Out by the end of day {e['hold_max_days']} regardless.",
                       "risk_note": gov["reason"], "news": {"verdict": "unchecked"}}
                entries.append(doc)
                if aid not in state["alerts"]:
                    _write(out, writes, "set", "alerts", aid, doc)
    counts = {"aplus": len([e for e in entries if not e["also_qualified"]]), "also_qualified": len([e for e in entries if e["also_qualified"]]),
              "watchlist": len(watch), "exit_alerts": len(exit_alerts), "open_positions": len(positions),
              "orders_checked": len(fill_checks), "fills_likely": len([f for f in fill_checks if f.get("filled")])}
    active = [sid for sid, r in P["strategies"].items() if r.get("enabled") and sid not in paused and (not r.get("market_conditions") or mkt["condition"] in r["market_conditions"])]
    scan = {"id": scan_id, "mode": mode, "started_at": iso(now), "status": "ok" if not errors else "partial", "data": {**health, "prices_as_of": asof, "tradable": len([1 for u in uni.values() if u.get("tradable")])},
            "market": mkt, "counts": counts, "positions": positions, "fill_checks": fill_checks, "alert_updates": alert_updates, "watchlist": watch[:12],
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
        bits.append(f"{c['aplus']} A+ setup{'s' if c['aplus'] != 1 else ''} ({', '.join(e['ticker'] for e in entries if not e['also_qualified'])})")
    if acts:
        bits.append(", ".join(f"{p['ticker']} {p['label'].split(':')[0]}" for p in acts))
    if c["fills_likely"]:
        bits.append(f"{c['fills_likely']} order{'s' if c['fills_likely'] != 1 else ''} likely filled")
    if not bits or bits == ["⚠ prices behind"]:
        bits.append("No action needed" if scan["mode"] == "check" or not c["aplus"] else "")
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
    fl = [f for f in scan["fill_checks"] if f.get("filled") or f.get("withdraw")]
    if fl:
        L += ["", "YOUR RESTING ORDERS"]
        for f in fl:
            L.append(f"- {f['ticker']}: {f.get('withdraw') or f['text']}")
    if scan["mode"] == "close":
        top = [e for e in entries if not e["also_qualified"]]
        L += ["", f"NEW A+ SETUPS ({len(top)})"]
        if not top:
            L.append("None today. The filter only passes setups whose record in this market condition is strong in training and on unseen data. No trade is a position too.")
        for e in top:
            L += [f"- {e['ticker']} | {e['strategy_label']} | score {e['score']} | win odds {e['win_pct']:.0f}%",
                  f"    Order: {e['order']}",
                  f"    Entry {e['entry']:.6g} | Stop {e['stop']:.6g} ({e['stop_pct']:+.1f}%) | Target {e['target']:.6g} ({e['target_pct']:+.1f}%) | reward:risk {e['rr']:.2f} after fees",
                  f"    Size {e['qty']:g} {e['ticker']} = ${e['total_cost']:,.2f} ({e['pct_of_capital']}% of capital), risk ${e['risk_usd']:,.2f}",
                  f"    Hold up to {e['hold_max_days']} days. Order good from {e['valid_from'][:16].replace('T', ' ')} to {e['valid_until'][:16].replace('T', ' ')} UTC.",
                  f"    {e['after_fill']}"]
            if (e.get("news") or {}).get("verdict") not in (None, "unchecked", "clean"):
                L.append(f"    News: {e['news']['verdict']}: {e['news'].get('reason', '')}")
        more = [e for e in entries if e["also_qualified"]]
        if more:
            L.append("Also qualified (no slot or cash today): " + ", ".join(f"{e['ticker']} ({e['strategy_label']}, {e['win_pct']:.0f}%)" for e in more))
        if scan["watchlist"]:
            L.append("Watchlist (grade A, not sent as alerts): " + ", ".join(f"{x['ticker']} ({x['win_pct']:.0f}%)" for x in scan["watchlist"]))
    hold = [p for p in scan["positions"] if p["code"] == "HOLD"]
    if hold:
        L += ["", "HOLDING"]
        for p in hold:
            L.append(f"- {p['ticker']}: {p.get('pct', 0):+.1f}% (${p.get('unrealized', 0):+,.2f}). {p['detail']}")
    L += ["", "Model signals from backtested rules, not financial advice. Win odds are estimates and can be wrong. You place every order yourself."]
    return "\n".join(L)


def apply_news(out_dir, verdicts: list[str], issues: list[str]):
    """Fold the news check (and any run problems) into the scan's documents, digest and headline."""
    out = pathlib.Path(out_dir)
    js = json.loads((out / "scan.json").read_text())
    scan, entries, exits = js["scan"], js["entries"], js.get("exit_alerts", [])
    v = {}
    for s in verdicts:
        tk, _, rest = s.partition("=")
        verdict, _, reason = rest.partition(":")
        v[tk.strip().upper()] = {"verdict": verdict.strip().lower(), "reason": reason.strip(), "checked_at": iso(pd.Timestamp.now(tz="UTC"))}
    for e in entries:
        n = v.get(e["ticker"])
        if not n:
            continue
        e["news"] = n
        if n["verdict"] == "avoid":
            e.update({"withdrawn": True, "withdrawn_reason": "News check: " + (n["reason"] or "avoid"), "also_qualified": True})
        elif n["verdict"] == "caution":
            e["score"] = round(e["score"] - 8, 1)
    live = sorted([e for e in entries if not e.get("withdrawn")], key=lambda e: -e["score"])
    for i, e in enumerate(live):
        e["rank"] = i + 1
    scan["counts"]["aplus"] = len([e for e in entries if not e["also_qualified"] and not e.get("withdrawn")])
    scan["news_checked"] = sorted(v)
    if issues:
        scan["run_issues"] = list(scan.get("run_issues", [])) + issues
    scan["headline"] = ("⚠ run problem · " if scan.get("run_issues") else "") + headline(scan, [e for e in entries if not e.get("withdrawn")])
    scan["digest"] = ("\n".join("⚠ RUN PROBLEM: " + i for i in scan.get("run_issues", [])) + "\n\n" if scan.get("run_issues") else "") + digest(scan, [e for e in entries if not e.get("withdrawn")], exits)
    wd = [e for e in entries if e.get("withdrawn")]
    if wd:
        scan["digest"] += "\n\nWithdrawn after the news check: " + "; ".join(f"{e['ticker']} ({e['withdrawn_reason']})" for e in wd)
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
    ap.add_argument("--issue", action="append", default=[])
    ap.add_argument("verdicts", nargs="*")
    a = ap.parse_args()
    if a.cmd == "news":
        print(apply_news(a.out, a.verdicts, a.issue))
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
