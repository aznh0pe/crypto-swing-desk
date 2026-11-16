"""Crypto price feed for the Crypto Swing Desk.

Pulls public market data from the Coinbase Exchange API (no key needed) and writes the files the
engine reads. Runs in a GitHub Action around the clock; the result is published as the `data` branch.

  daily.csv.gz   date,ticker,open,high,low,close,volume   every USD pair, full history (UTC daily bars). Rewritten
                                                          only when a day finishes, a coin lists or history is rebuilt.
  recent.csv.gz  same columns                             the last days of every pair incl. the bar still forming;
                                                          rewritten every pass. The engine lays it over daily.csv.gz.
  hourly.csv.gz  time,ticker,open,high,low,close,volume   last ~200 hours for every coin the desk may scan
  universe.json  one item per pair: status, liquidity, tier, first bar, order increments
  fng.json       Crypto Fear & Greed index history (optional extra; the engine works without it)
  meta.json      when the feed ran, the last finished daily bar, counts and problems

Daily bars close at 00:00 UTC. The bar for the current UTC day is still forming: meta.partial_day names it.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

BASE = "https://api.exchange.coinbase.com"
UA = {"User-Agent": "crypto-swing-desk-feed/1.0", "Accept": "application/json"}
OUT = pathlib.Path(os.environ.get("OUT_DIR", "out"))
START = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
# not swing candidates: dollar/euro stablecoins and wrapped or staked copies of another coin
STABLE = {"USDT", "USDC", "DAI", "PYUSD", "GUSD", "USDS", "EURC", "PAX", "BUSD", "UST", "USDP", "TUSD", "FDUSD",
          "GYEN", "EURT", "USD1", "USDE", "RLUSD", "USDG", "FRAX", "LUSD", "SUSD", "MUSD", "CUSD", "UXD"}
WRAPPED = {"WBTC", "CBBTC", "CBETH", "LSETH", "MSOL", "WAXL", "WCFG", "WLUNA", "WAMPL", "RETH", "STETH", "WSTETH",
           "JITOSOL", "BSOL", "LBTC", "CBXRP", "CBDOGE", "CBADA", "CBLTC", "PAXG"}
CORE_DOLLAR_VOL = 2_000_000     # median daily dollar volume over 30 days: the core tier
MIN_DOLLAR_VOL = 100_000        # floor for the feed to carry hourly bars (the engine applies its own, stricter floor)
MIN_HISTORY_DAYS = 120
RECENT_DAYS = 12
HISTORY_EVERY_DAYS = 7          # full rebuild (revisions, relistings) once a week, in a quiet hour
HOURS = 200
_last = [0.0]


def get(path: str, tries: int = 6):
    """GET with a polite rate (about 6 a second) and back-off on 429 / 5xx."""
    url = path if path.startswith("http") else BASE + path
    for i in range(tries):
        wait = 0.17 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(30, 1.5 * 2 ** i))
        except Exception:
            time.sleep(min(30, 1.5 * 2 ** i))
    raise RuntimeError(f"no answer from {url}")


def candles(product: str, gran: int, start: dt.datetime, end: dt.datetime) -> list:
    """All candles in [start, end], oldest first. Coinbase returns at most 300 per call."""
    out, step = [], dt.timedelta(seconds=gran * 300)
    a = start
    while a < end:
        b = min(a + step, end)
        rows = get(f"/products/{product}/candles?granularity={gran}&start={a.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                   f"&end={b.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if rows:
            out.extend(rows)
        a = b
    return sorted({r[0]: r for r in out if isinstance(r, list) and len(r) >= 6}.values())


def backfill(product: str, gran: int, start: dt.datetime, end: dt.datetime) -> list:
    """Full history, newest window first, stopping once two windows in a row come back empty after data was
    seen (the coin was not listed yet). Saves thousands of empty calls on a full rebuild."""
    out, step, b, empty = [], dt.timedelta(seconds=gran * 300), end, 0
    while b > start:
        a = max(b - step, start)
        rows = get(f"/products/{product}/candles?granularity={gran}&start={a.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                   f"&end={b.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if rows:
            out.extend(rows)
            empty = 0
        else:
            empty += 1
            if empty >= 2:
                break
        b = a
    return sorted({r[0]: r for r in out if isinstance(r, list) and len(r) >= 6}.values())


def frame(rows: list, ticker: str, col: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[col, "ticker", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    ts = pd.to_datetime(df["t"], unit="s", utc=True)
    df[col] = ts.dt.strftime("%Y-%m-%d" if col == "date" else "%Y-%m-%dT%H:%MZ")
    df["ticker"] = ticker
    return df[[col, "ticker", "open", "high", "low", "close", "volume"]]


def read_gz(name: str, col: str) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        return pd.DataFrame(columns=[col, "ticker", "open", "high", "low", "close", "volume"])
    return pd.read_csv(p, dtype={col: str, "ticker": str})


def write_gz(df: pd.DataFrame, name: str):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
        g.write(df.to_csv(index=False, float_format="%.10g").encode())
    (OUT / name).write_bytes(buf.getvalue())


def restricted(p: dict) -> str | None:
    """Why a listed pair cannot be traded normally right now (None = it can)."""
    if p.get("status") != "online":
        return p.get("status") or "gone"
    for flag in ("trading_disabled", "cancel_only", "post_only", "limit_only", "auction_mode"):
        if p.get(flag):
            return flag
    return None


def main(mode: str = "auto"):
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    today = now.strftime("%Y-%m-%d")
    problems: list[str] = []
    prods = get("/products") or []
    usd = [p for p in prods if p.get("quote_currency") == "USD" and p.get("base_currency") not in STABLE
           and p.get("base_currency") not in WRAPPED]
    if len(usd) < 20:
        raise RuntimeError(f"Coinbase returned only {len(usd)} USD pairs; not publishing")
    try:
        old_meta = json.loads((OUT / "meta.json").read_text())
    except Exception:
        old_meta = {}
    old = read_gz("daily.csv.gz", "date")
    have = set(old["ticker"].unique())
    built = old_meta.get("history_built")
    age = (now - dt.datetime.fromisoformat(built)).days if built else 999
    full = mode == "history" or old.empty or (mode == "auto" and age >= HISTORY_EVERY_DAYS and 3 <= now.hour < 11)
    # how far back the quick top-up must reach to join up with what is already stored
    old_max = old["date"].max() if len(old) else None
    gap = (now.date() - dt.date.fromisoformat(old_max)).days if old_max else 0
    reach = min(299, max(RECENT_DAYS, gap + 3))
    parts, n_err, new_coins = [], 0, []
    for k, p in enumerate(sorted(usd, key=lambda x: x["id"])):
        tk, pid = p["base_currency"], p["id"]
        try:
            if full or tk not in have:
                rows = backfill(pid, 86400, START, now)
                if not full and rows:
                    new_coins.append(tk)
            elif p.get("status") != "online":
                continue                                   # delisted: nothing new to fetch
            else:
                rows = candles(pid, 86400, now - dt.timedelta(days=reach), now)
            parts.append(frame(rows, tk, "date"))
        except Exception as e:
            n_err += 1
            if n_err <= 8:
                problems.append(f"{pid} daily: {e}"[:160])
        if k % 100 == 0:
            print(f"daily {k}/{len(usd)}", flush=True)
    if n_err > 0.3 * len(usd):
        raise RuntimeError(f"{n_err} of {len(usd)} pairs failed; not publishing")
    new = pd.concat([f for f in parts if len(f)], ignore_index=True) if any(len(f) for f in parts) else old.iloc[0:0]
    if full:   # keep stored history only for pairs the rebuild could not reach (delisted long ago)
        daily = pd.concat([old[~old["ticker"].isin(set(new["ticker"]))], new], ignore_index=True)
    else:
        daily = pd.concat([old, new], ignore_index=True)
    daily = daily.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    daily = daily[(daily["close"] > 0) & (daily["high"] >= daily["low"])]

    # universe: what trades on Coinbase today, how liquid it is, and the order increments
    done = daily[daily["date"] < today]
    items, active = [], []
    info = {p["base_currency"]: p for p in usd}
    fresh_cut = (now - dt.timedelta(days=3)).strftime("%Y-%m-%d")
    for tk, g in done.groupby("ticker"):
        p = info.get(tk, {})
        tail = g.tail(30)
        dv = float((tail["close"] * tail["volume"]).median()) if len(tail) else 0.0
        c = g["close"].tail(200)
        peg = len(c) >= 60 and float(((c > 0.97) & (c < 1.03)).mean()) > 0.95   # behaves like a stablecoin
        why = restricted(p) if p else "gone"
        if why is None and g["date"].iloc[-1] < fresh_cut:
            why = "no recent trades"
        ok = bool(why is None and not peg and dv >= MIN_DOLLAR_VOL and len(g) >= MIN_HISTORY_DAYS)
        items.append({"ticker": tk, "product": p.get("id") or f"{tk}-USD", "status": "online" if why is None else ("delisted" if why in ("delisted", "gone") else "restricted"),
                      "restriction": why, "first": g["date"].iloc[0], "last": g["date"].iloc[-1], "days": int(len(g)),
                      "dollar_vol_30d": round(dv), "peg": bool(peg), "tradable": ok,
                      "tier": None if not ok else ("core" if dv >= CORE_DOLLAR_VOL else "wide"),
                      "tick": p.get("quote_increment"), "min_size": p.get("base_increment")})
        if ok:
            active.append(tk)
    items.sort(key=lambda x: -x["dollar_vol_30d"])
    pegged = {i["ticker"] for i in items if i["peg"]}
    daily = daily[~daily["ticker"].isin(pegged)]
    # the big history file changes only when a day has finished, a coin was added or history was rebuilt
    # (it holds finished days only; the forming bar lives in recent.csv.gz)
    stored_done = old["date"].max() if len(old) else None
    last_done = done["date"].max() if len(done) else None
    rewrite = full or bool(new_coins) or stored_done != last_done or not (OUT / "daily.csv.gz").exists()
    if rewrite:
        write_gz(daily[daily["date"] < today], "daily.csv.gz")
    cut = (now - dt.timedelta(days=RECENT_DAYS + 1)).strftime("%Y-%m-%d")
    write_gz(daily[daily["date"] >= cut], "recent.csv.gz")
    (OUT / "universe.json").write_text(json.dumps({"built": now.isoformat(), "min_dollar_vol": MIN_DOLLAR_VOL, "core_dollar_vol": CORE_DOLLAR_VOL,
                                                   "tradable": len(active), "core": len([i for i in items if i["tier"] == "core"]),
                                                   "items": items}, indent=0))

    # hourly bars for every coin the desk may scan: live marks, fill checks and position checks
    hp, h_err = [], 0
    for tk in active:
        try:
            hp.append(frame(candles(info[tk]["id"], 3600, now - dt.timedelta(hours=HOURS), now), tk, "time"))
        except Exception as e:
            h_err += 1
            if h_err <= 5:
                problems.append(f"{tk} hourly: {e}"[:160])
    hp = [f for f in hp if len(f)]
    if hp and h_err <= 0.3 * len(active):
        write_gz(pd.concat(hp, ignore_index=True).sort_values(["ticker", "time"]), "hourly.csv.gz")
    elif active:
        problems.append("hourly bars not refreshed this pass")

    fng_file = OUT / "fng.json"
    if not fng_file.exists() or full or now.minute < 12:      # the index changes once a day: ask about hourly
        try:
            fng = get("https://api.alternative.me/fng/?limit=0&format=json", tries=2)
            rows = [{"date": dt.datetime.fromtimestamp(int(d["timestamp"]), dt.timezone.utc).strftime("%Y-%m-%d"),
                     "value": int(d["value"])} for d in (fng or {}).get("data", [])]
            if len(rows) > 100:
                fng_file.write_text(json.dumps(sorted(rows, key=lambda r: r["date"])))
        except Exception as e:
            problems.append(f"fear & greed: {e}"[:120])

    meta = {"generated_utc": now.isoformat(), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source": "Coinbase Exchange public API",
            "mode": "history" if full else "quick", "history_built": now.isoformat() if full else built,
            "last_daily_bar": last_done, "partial_day": today, "pairs": int(daily["ticker"].nunique()), "listed": len([i for i in items if i["status"] == "online"]),
            "tradable": len(active), "core": len([i for i in items if i["tier"] == "core"]), "new_coins": new_coins,
            "rows": int(len(daily)), "first_bar": daily["date"].min(), "daily_rewritten": bool(rewrite),
            "hourly_last": (max(f["time"].max() for f in hp) if hp else old_meta.get("hourly_last")), "problems": problems}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODE", "auto"))
