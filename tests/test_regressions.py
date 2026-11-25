"""Regression tests: one for every fault the 2026-11-16 audit found, so none of them can come back unnoticed.
Run: python3 tests/test_regressions.py"""
import json, pathlib, shutil, sys, tempfile
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.make_synth import make  # noqa: E402
from engine import backtest as bt, data as D, indicators as I, learn as L, portfolio as PF, rules, scan as SC, strategies as S  # noqa: E402


def entry(tk, score, grade="A+", price=10.0):
    return {"ticker": tk, "product": f"{tk}-USD", "strategy": "DIP_UPTREND", "strategy_label": "Dip", "family": "x", "about": "x", "grade": grade, "score": score,
            "win_pct": 75.0, "cell_n": 100, "cell_days": 30, "tier": "core", "min_size": "0.01", "cell_train": 0.7, "cell_oos": 0.7, "condition": "bull_calm",
            "entry_kind": "limit_next", "entry": price, "stop": price * 0.9, "target": price * 5, "atr": 0.5, "last_close": price * 1.02, "stop_pct": -10.0,
            "target_pct": 10.0, "net_win_pct": 8.9, "net_loss_pct": -11.3, "rr": 0.79, "hold_max_days": 5, "valid_days": 1, "dyn_exit": "none", "exit_signal": "",
            "signal_date": "x", "rs_pct": 50, "rsi2": 5.0, "rsi14": 30.0, "vs_50d_pct": -5.0, "vs_200d_pct": 1.0, "atr_pct": 5.0, "dollar_vol_30d": 1e7, "is_major": False}


def main():
    w = pathlib.Path(tempfile.mkdtemp())
    make(w / "data")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    # 1. timestamps in any form parse to naive UTC (the old one-liner crashed on some forms)
    assert D.naive_ts("2026-11-16T00:00:00Z") == pd.Timestamp("2026-11-16") == D.naive_ts("2026-11-16 00:00:00+00:00") == D.naive_ts("2026-11-16")
    assert D.naive_ts(None, "d") == "d" and D.naive_ts("garbage", "d") == "d"

    # 2. order size is floored to the exchange's step and never costs more than the budget
    assert PF._round_qty(1.23456789, "0.01") == 1.23 and PF._round_qty(0.999, "1") == 0.0 and PF._round_qty(177.979, "0.1") == 177.9
    sz = PF.size_position(0.469, 0.307, 1500, 1500, 2.0, {"maker_fee_pct": 0.5, "taker_fee_pct": 0.9, "max_position_pct": 30}, 25, "0.01")
    assert sz["qty"] * 0.469 <= 450.001 and abs(sz["qty"] * 100 - round(sz["qty"] * 100)) < 1e-6

    # 3. on daily bars alone, the high of the fill day never counts as a target hit (it may have printed before the fill)
    d0 = pd.Timestamp("2026-01-10")
    idx = pd.date_range("2026-01-08", "2026-01-11", freq="D")
    wd = {"low": pd.DataFrame({"X": [9, 9, 8.0, 9.5]}, index=idx), "high": pd.DataFrame({"X": [10, 10, 20.0, 11.0]}, index=idx)}
    ex = SC.extremes_since("X", d0 + pd.Timedelta(hours=9), wd, None, None)
    assert ex["high"] == 11.0 and ex["low"] == 8.0, ex
    hr = pd.date_range("2026-01-09", periods=72, freq="h")                     # with hourly bars covering the fill, they decide
    hd = {"low": pd.DataFrame({"X": np.r_[np.full(33, 9.0), np.full(39, 9.4)]}, index=hr), "high": pd.DataFrame({"X": np.r_[np.full(33, 20.0), np.full(39, 10.5)]}, index=hr)}
    ex = SC.extremes_since("X", d0 + pd.Timedelta(hours=9), wd, None, hd)
    assert ex["high"] == 10.5 and ex["low"] == 9.4, ex

    # 4. a trade still open when its coin stops trading is closed at the last price, not silently dropped
    P = S.load_params(None)
    wdl, part, meta = D.load_daily(w / "data")
    for f in wdl:                                                                # delist C05 forty days ago
        wdl[f].loc[wdl[f].index[-40]:, "C05"] = np.nan
    ind = I.compute(wdl, None, min_dv=P["universe"]["min_dv"])
    b = I.base_masks(ind)
    spec = {"id": "T_ALWAYS", "label": "t", "family": "t", "about": "t", "universe": "all", "conditions": ["close > 0"], "entry": {"kind": "open_next"},
            "stop": {"atr": 50}, "target": {"atr": 50}, "exit": "none", "max_hold": 30, "valid_days": 1, "params": {}, "grid": {}}
    assert not rules.validate(spec), rules.validate(spec)
    tr = bt.simulate(rules.build_plan(spec, ind, P, b, {}, "T_ALWAYS"), ind, P["costs"])
    x = tr[(tr["ticker"] == "C05") & tr["exit_date"].notna()] if "exit_date" in tr else tr[tr["ticker"] == "C05"]
    assert (x["reason"] == "delisted").any(), x["reason"].value_counts().to_dict()

    # 5. the engine lays the feed's small recent file over the history file
    d2 = w / "data2"
    shutil.copytree(w / "data", d2)
    df = pd.read_csv(d2 / "daily.csv.gz", dtype={"ticker": str})
    last = df["date"].max()
    df[df["date"] < last].to_csv(d2 / "daily.csv.gz", index=False, compression="gzip")
    rec = df[df["date"] >= sorted(df["date"].unique())[-5]].copy()
    rec.loc[(rec["ticker"] == "BTC") & (rec["date"] == sorted(df["date"].unique())[-2]), "close"] = 12345.0
    rec.to_csv(d2 / "recent.csv.gz", index=False, compression="gzip")
    w2, part2, _ = D.load_daily(d2)
    assert part2 is not None and str(part2["day"].date()) == last and float(w2["close"]["BTC"].iloc[-1]) == 12345.0

    # 6. every A+ setup is served: nothing is cut at a daily cap, and what the account cannot hold is still a full ticket
    st = w / "state"
    (st / "strategy").mkdir(parents=True)
    real_find = SC.find_entries
    px = lambda ind, tk: 0.6 * float(ind["close"][tk].iloc[-1])                  # a bid far below the market: cannot have traded yet
    SC.find_entries = lambda ind, P, uni, cond, paused: ([entry(f"C{i:02d}", 90 - i, price=px(ind, f"C{i:02d}")) for i in range(9)] + [entry("C15", 50, "A", px(ind, "C15"))], [])
    out = SC.run("close", w / "data", st, w / "scan1", now.normalize() + pd.Timedelta(minutes=20))
    js = json.loads((w / "scan1" / "scan.json").read_text())
    ents = js["entries"]
    assert len(ents) == 9 and js["scan"]["counts"]["aplus"] == 9, len(ents)
    assert 1 <= sum(e["funded"] for e in ents) <= 8 and any(not e["funded"] for e in ents) and all(e["qty"] > 0 and e["order"] for e in ents)
    assert len([x for x in json.loads((w / "scan1" / "writes.json").read_text()) if x["collection"] == "alerts"]) == 9
    assert js["scan"]["pace"]["served"] == 9 and js["scan"]["universe"]["scanned"] > 0 and len(js["scan"]["watchlist"]) == 1

    # 7. a check run catches up when no close scan ran for the bar, and does not repeat it once one has
    out = SC.run("check", w / "data", st, w / "scan2")
    s2 = json.loads((w / "scan2" / "scan.json").read_text())
    assert s2["scan"]["catch_up"] and len(s2["entries"]) >= 1
    (st / "scans").mkdir()
    (st / "scans" / "a.json").write_text(json.dumps(s2["scan"]))
    SC.run("check", w / "data", st, w / "scan3")
    s3 = json.loads((w / "scan3" / "scan.json").read_text())
    assert not s3["scan"]["catch_up"] and not s3["entries"]
    # 7b. a catch-up run never re-announces a setup an earlier scan already served
    shutil.rmtree(st / "scans")
    (st / "alerts").mkdir()
    for f in (w / "scan1" / "docs" / "alerts").glob("*.json"):
        shutil.copy(f, st / "alerts" / f.name)
    SC.run("check", w / "data", st, w / "scan3b")
    s3b = json.loads((w / "scan3b" / "scan.json").read_text())
    assert s3b["scan"]["catch_up"] and not s3b["entries"] and s3b["scan"]["counts"]["aplus"] == 0, len(s3b["entries"])
    shutil.rmtree(st / "alerts")
    # 7c. the book stamp follows what decides setups (settings, grades, floor), not the code version or the clock
    P0 = S.load_params(None)
    P0["grade_table"] = {"DIP_UPTREND": {"bull_calm": {"p": 0.7, "n": 60, "grade": "A+"}}}
    b0 = SC.book_id(P0)
    assert SC.book_id({**P0, "calibrated_at": "later", "rev": 99}) == b0
    assert SC.book_id({**P0, "grade_table": {"DIP_UPTREND": {"bull_calm": {"p": 0.6, "n": 60, "grade": "A"}}}}) != b0
    # 7d. run notes are not run problems: a second news search must never raise the alarm (2026-11-17, USELESS)
    h0 = json.loads((w / "scan1" / "scan.json").read_text())["scan"]["headline"]
    tk = json.loads((w / "scan1" / "scan.json").read_text())["entries"][0]["ticker"]
    SC.apply_news(w / "scan1", [f"{tk}=clean", "C01=maybe:could not tell"], [], ["News check for X took a second search."])
    j = json.loads((w / "scan1" / "scan.json").read_text())
    assert not j["scan"].get("run_issues") and "run problem" not in j["scan"]["headline"] and "RUN PROBLEM" not in j["scan"]["digest"]
    assert j["scan"]["headline"] == h0 and "How this run went" in j["scan"]["digest"] and j["scan"]["run_notes"]
    assert [e for e in j["entries"] if e["ticker"] == "C01"][0]["news"]["verdict"] == "unclear"
    SC.apply_news(w / "scan1", [], ["The results did not save."], [])
    j = json.loads((w / "scan1" / "scan.json").read_text())
    assert j["scan"]["run_issues"] == ["The results did not save."] and j["scan"]["headline"].startswith("⚠ run problem")
    SC.find_entries = real_find

    # 8. stale prices: no setups from old bars, and the scan says so (so the next run catches up)
    d3 = w / "data3"
    shutil.copytree(w / "data", d3)
    df = pd.read_csv(d3 / "daily.csv.gz", dtype={"ticker": str})
    df[df["date"] < sorted(df["date"].unique())[-3]].to_csv(d3 / "daily.csv.gz", index=False, compression="gzip")
    SC.run("close", d3, st, w / "scan4")
    s4 = json.loads((w / "scan4" / "scan.json").read_text())["scan"]
    assert s4["skipped_setups"] and s4["data"]["status"] == "stale" and "prices behind" in s4["headline"]

    # 9. live results feed the right tier's table; an old two-part key still means the core list
    P = S.load_params(None)
    P["grade_table"] = {"DIP_UPTREND": {"bull_calm": {"p": 0.70, "n": 60, "grade": "A+"}}}
    P["grade_table_wide"] = {"DIP_UPTREND": {"bull_calm": {"p": 0.70, "n": 60, "grade": "A+"}}}
    L.apply_live(P, {"cells": {"DIP_UPTREND|bull_calm|wide": {"n": 10, "wins": 1}}})
    assert P["grade_table"]["DIP_UPTREND"]["bull_calm"]["grade"] == "A+" and P["grade_table_wide"]["DIP_UPTREND"]["bull_calm"]["grade"] != "A+"
    L.apply_live(P, {"cells": {"DIP_UPTREND|bull_calm": {"n": 10, "wins": 1}}})
    assert P["grade_table"]["DIP_UPTREND"]["bull_calm"]["grade"] != "A+"

    # 10. the daily journal is filed under the UTC trading day that just finished, and reports run health, pace and the coin list
    r = L.review(w / "data", st, w / "learn")
    assert r["journal"] == (now.normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), r["journal"]
    j = json.loads((w / "learn" / "journal.json").read_text())
    assert {"opportunities", "pace", "universe", "runs"} <= set(j) and j["universe"]["scanned"] > 0

    # 11. the radar explains what is missing in plain words
    assert rules.plain_condition("ret7 <= -0.3")[0] == "7-day change at or below -30.0%" and rules.plain_condition("rsi2 < 8")[0] == "RSI(2) below 8"
    # 12. danger zones: the middle of a bear market is its own zone, and a blocked zone sends no A+ alert (2026-11-17)
    assert [I.zone_of(x) for x in (0.05, -0.05, -0.11, -0.34, -0.36, float("nan"))] == ["bull", "shallow", "slide", "slide", "deep", "bull"]
    Pz = S.load_params(None)
    indz = I.compute(D.load_daily(w / "data")[0], None, min_dv=Pz["universe"]["min_dv"])
    zone = I.zone_of(indz["market"]["btc_dist200"].iloc[-1])
    sid = next(k for k, v in Pz["strategies"].items() if v.get("enabled"))
    all_a = {c: {"p": 0.9, "n": 99, "days": 30, "grade": "A+"} for c in I.CONDITIONS}
    Pz["grade_table"] = {k: dict(all_a) for k in Pz["strategies"]}
    Pz["grade_table_wide"] = {k: dict(all_a) for k in Pz["strategies"]}
    open_found, _ = SC.find_entries(indz, Pz, {}, I.condition_series(indz["market"]).iloc[-1] if hasattr(I, "condition_series") else "bull_calm", {})
    Pz["zones"] = {"blocked": [zone], "table": {zone: {"train": {"win_rate": 0.46}, "unseen": {"win_rate": 0.58}, "all": {"trades": 755, "days": 70}}}}
    blocked_found, notes = SC.find_entries(indz, Pz, {}, I.condition_series(indz["market"]).iloc[-1] if hasattr(I, "condition_series") else "bull_calm", {})
    assert not [e for e in blocked_found if e["grade"] == "A+"] and len(blocked_found) == len(open_found)
    assert any("Danger zone" in n for n in notes)
    # 12b. ...except for a strategy x coin tier with its own A+ record in that zone, shown with the zone's odds
    Pz["zones"]["exempt"] = {zone: {f"{k}|{t}": {"p": 0.67, "n": 49, "days": 17, "train": 0.75, "oos": 0.67} for k in Pz["strategies"] for t in ("core", "wide")}}
    ex_found, _ = SC.find_entries(indz, Pz, {}, I.condition_series(indz["market"]).iloc[-1] if hasattr(I, "condition_series") else "bull_calm", {})
    assert len([e for e in ex_found if e["grade"] == "A+"]) == len([e for e in open_found if e["grade"] == "A+"])
    assert all(e["win_pct"] == 67.0 and e["zone_cell"] for e in ex_found if e["grade"] == "A+")
    # 13. an open position whose strategy has no record of its own in a blocked zone gets a sell call; an exempt one holds
    wd0, part0, _ = D.load_daily(w / "data")
    tk0 = [c for c in indz["close"].columns if c.startswith("C")][0]
    px0 = {tk0: float(indz["close"][tk0].iloc[-1])}
    trade = {"id": "t1", "ticker": tk0, "entry_price": px0[tk0] * 0.98, "qty": 10, "stop": px0[tk0] * 0.5, "target": px0[tk0] * 3, "hold_max_days": 6, "dyn_exit": "none",
             "entry_at": (now - pd.Timedelta(hours=3)).isoformat(), "strategy": sid}
    Pz["zones"]["exempt"] = {}
    st1 = SC.check_position(trade, indz, wd0, part0, None, px0, Pz, "check", now)
    assert st1["code"] == "ZONE_EXIT" and st1["tone"] == "sell", st1["code"]
    tier0 = "core" if bool(indz["is_core"][tk0].iloc[-1]) else "wide"
    Pz["zones"]["exempt"] = {zone: {f"{sid}|{tier0}": {"p": 0.67, "n": 49, "days": 17}}}
    assert SC.check_position(trade, indz, wd0, part0, None, px0, Pz, "check", now)["code"] == "HOLD"
    # 13b. the exemption follows the tier the trade was bought on, not today's volume (ZRO, 25 Nov)
    other = "wide" if tier0 == "core" else "core"
    Pz["zones"]["exempt"] = {zone: {f"{sid}|{other}": {"p": 0.67, "n": 49, "days": 17}}}
    assert SC.check_position({**trade, "tier": other}, indz, wd0, part0, None, px0, Pz, "check", now)["code"] == "HOLD"
    assert SC.check_position(trade, indz, wd0, part0, None, px0, Pz, "check", now)["code"] == "ZONE_EXIT"
    # 14. the book stamp changes when a zone opens or closes, so the next scan re-grades live tickets
    assert SC.book_id({**P0, "zones": {"blocked": ["slide"], "exempt": {}}}) != b0
    # 15. an order past its window is judged on the window alone, and the verdict is an instruction
    T = pd.Timestamp
    hx = pd.date_range("2026-03-01", periods=96, freq="h")                     # window = all of 2026-03-02
    lo = np.full(96, 10.5); lo[30] = 10.2                                      # lowest trade inside the window: 10.2
    hq = {"low": pd.DataFrame({"X": lo}, index=hx), "high": pd.DataFrame({"X": lo + 0.3}, index=hx)}
    al = {"id": "a1", "ticker": "X", "entry": 10.0, "target": 12.0, "stop": 9.0, "status": "placed",
          "valid_from": "2026-03-02T00:00:00Z", "valid_until": "2026-03-03T00:00:00Z"}
    late = T("2026-03-03 00:20")
    fc = SC.check_order(al, wd, None, hq, {"X": 10.5}, late, feed_at=T("2026-03-03 00:11"))
    assert fc["action"] == "cancel" and fc["confirmed"] and fc["filled"] is False and fc["text"].startswith("CANCEL THIS ORDER") and "10.2" in fc["text"], fc
    fc = SC.check_order(al, wd, None, hq, {"X": 10.5}, late, feed_at=T("2026-03-02 23:55"))   # feed has not covered the last minutes: no verdict yet
    assert fc["action"] == "check" and not fc.get("confirmed") and "CANCEL" not in fc["text"], fc
    lo2 = lo.copy(); lo2[60] = 9.8                                             # dipped under the limit only after the window
    hq2 = {"low": pd.DataFrame({"X": lo2}, index=hx), "high": hq["high"]}
    fc = SC.check_order(al, wd, None, hq2, {"X": 10.5}, T("2026-03-04 13:20"), feed_at=T("2026-03-04 13:11"))
    assert fc["action"] == "check" and fc["filled"] is False and "CANCEL THIS ORDER NOW" in fc["text"] and "late" in fc["text"], fc
    lo3 = lo.copy(); lo3[30] = 9.9                                             # traded under the limit inside the window
    fc = SC.check_order(al, wd, None, {"low": pd.DataFrame({"X": lo3}, index=hx), "high": hq["high"]}, {"X": 10.5}, late, feed_at=late)
    assert fc["action"] == "record_fill" and fc["filled"] is True, fc
    lo4 = lo.copy(); lo4[30] = 10.0                                            # touched the limit exactly: nobody can tell from prices
    fc = SC.check_order(al, wd, None, {"low": pd.DataFrame({"X": lo4}, index=hx), "high": hq["high"]}, {"X": 10.5}, late, feed_at=late)
    assert fc["action"] == "check" and fc["filled"] is None, fc
    fc = SC.check_order({**al, "status": "new"}, wd, None, hq, {"X": 10.5}, late, feed_at=late)   # never marked placed: no shouting
    assert fc["action"] == "cancel" and fc["text"].startswith("Expired unfilled"), fc
    di = pd.date_range("2026-03-01", "2026-03-03", freq="D")                   # daily bars only: the window's own day decides
    wdd = {"low": pd.DataFrame({"X": [9.0, 10.2, 9.5]}, index=di), "high": pd.DataFrame({"X": [11.0, 11.0, 11.0]}, index=di)}
    wl = SC.window_lows("X", T("2026-03-02"), T("2026-03-03"), wdd, None, None)
    assert wl["win"] == 10.2 and wl["after"] == 9.5, wl
    fc = SC.check_order(al, wd, None, hq, {"X": 10.5}, T("2026-03-02 13:00"))  # still inside the window: no instruction
    assert fc["window_over"] is False and "action" not in fc and fc["text"].endswith("Still working."), fc
    # 16. strategies that take the same trades are one voice: no agreement bonus between them
    import engine.calibrate as CB
    tr = pd.DataFrame({"strategy": ["A"] * 10 + ["B"] * 9 + ["C"] * 5,
                       "ticker": [f"T{i}" for i in range(10)] + [f"T{i}" for i in range(9)] + ["T0", "X1", "X2", "X3", "X4"],
                       "signal_date": [T("2026-01-01")] * 24})
    gr = CB.nested_groups(tr)
    assert gr["B"] == gr["A"] and gr["C"] == "C", gr
    gg = tr.assign(grade="A+", score=50.0)
    cf = CB.add_confluence(gg, {k: v for k, v in gr.items() if k != v})
    assert int(cf[(cf["ticker"] == "T0")]["confluence"].max()) == 2 and int(cf[(cf["ticker"] == "T5")]["confluence"].max()) == 1
    # 17. a zone whose average trade loses stays blocked even if its win rate edges over 58%
    src = pathlib.Path(CB.__file__).read_text()
    assert 's["avg_ret"] < R["avg"] and (s["win_rate"] < R["win"] or s["avg_ret"] < 0)' in src
    # 18. zone judgements are made on settled trades too: fresh one-day marks can never reopen a zone on their own
    src = pathlib.Path(CB.__file__).read_text()
    assert "settled_mask" in src and "passes(x[settled_mask(x)])" in src and "any(verdicts)" in src
    assert "aside.add((zone" in src and "in aside for z_" in src      # near-miss winners never lift the rest of the book
    # 19. the regime is read from the newest bar, and cover respects a blocked zone
    import engine.indicators as II
    dd = pd.date_range("2026-01-01", periods=6, freq="D")
    mkt = pd.DataFrame({"bull": [1, 1, 0, 0, 0, 0], "stress": [0, 1, 1, 1, 0, 0], "btc_dist200": [0.05, 0.02, -0.12, -0.15, -0.16, -0.17]}, index=dd)
    rg = II.regime_now({"market": mkt})
    assert rg["condition"] == "bear_calm" and rg["day_of_stretch"] == 2 and rg["zone"] == "slide" and rg["days_in_zone"] == 4 and rg["previous_condition"] == "Bear-market selloff", rg
    Pc = {"strategies": {"S1": {"enabled": True}, "S2": {"enabled": True}},
          "grade_table": {"S1": {c: {"grade": "A+", "p": 0.7} for c in II.CONDITIONS}, "S2": {"bear_calm": {"grade": "A+", "p": 0.8}}}, "grade_table_wide": {},
          "zones": {"blocked": ["slide"], "exempt": {"slide": {"S2|core": {}}}}}
    cv = II.alert_cover(Pc, "bear_calm", "slide")
    assert set(cv["can_alert"]) == {"S2"} and set(cv["benched_by_zone"]) == {"S1"} and cv["all_weather"] == ["S1"], cv
    assert set(II.alert_cover(Pc, "bear_calm", "bull")["can_alert"]) == {"S1", "S2"}
    # a run that updates one alert twice keeps both updates
    wr, od = [], pathlib.Path(tempfile.mkdtemp())
    SC._write(od, wr, "update", "alerts", "a1", {"win_pct": 67}); SC._write(od, wr, "update", "alerts", "a1", {"expiry": {"action": "cancel"}})
    assert len(wr) == 1 and json.loads(pathlib.Path(wr[0]["file_path"]).read_text()) == {"win_pct": 67, "expiry": {"action": "cancel"}}
    print("regression tests passed")


if __name__ == "__main__":
    main()
