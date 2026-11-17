# Crypto Swing Desk

A rules-based crypto swing-trading desk for a small spot account (started at $1,500, long only, no leverage).
It scans the liquid USD pairs on Coinbase after every daily close, sends only A+ setups (limit entry, stop,
target, size, hold window, time left, exit rule), watches open positions, grades every alert against what the
market did, and researches and tests new strategies every day.

**Not financial advice.** These are model-generated signals from backtested rules. Win percentages are
estimates and can be wrong. You place and manage every order yourself.

Separate from the stocks Swing Desk (`aznh0pe/swing-desk`): its own repo, data feed, dashboard, database and
scheduled tasks. Nothing here reads or writes anything of the stocks desk.

## How it fits together

| Piece | Where | What it does |
|---|---|---|
| Price feed | `.github/workflows/crypto-data.yml`, `crypto-watchdog.yml`, `pipeline/` | A GitHub Action that loops around the clock: one pass every 10 minutes for about 5.5 hours, then the run starts its successor (GitHub's own schedule proved far too sparse: 3 runs in 7 hours). `crypto-watchdog` checks every 15 minutes that the feed is fresh and restarts the loop if it stopped. Coinbase Exchange public API, no keys. `daily.csv.gz` holds finished daily bars for every USD pair incl. delisted ones (rewritten once a day), `recent.csv.gz` the last days incl. the forming bar (every pass), `hourly.csv.gz` ~200 hours for every coin the desk may scan, `universe.json` status, liquidity tier and order increments. |
| Engine | `engine/` | Indicators, strategies as JSON rule specs, a fee-aware backtester, calibration and grading by market condition, the scanner, the position monitor, the daily learning review, strategy research and the lessons registry. |
| Scheduled runs | Claude scheduled tasks (`ops/schedule.json`, prompts in `ops/generated/`) | Daily-close scan (00:20 UTC), two position checks a day, the daily learning review + research, the Sunday recalibration, and Scan now from the dashboard. Each reads the engine copy published with the dashboard, clones the `data` branch, exports the dashboard state, runs the engine, writes the results and sends a phone alert. |
| Dashboard | Claude artifact (`dashboard/`) | Overview, Alerts (order tickets with countdowns, open positions with what to do now), Trade log, Funds and settings, Scans, Strategy lab (backtests, the market-condition map, research log, lessons, daily journal). |

## Method

* **Daily bars close at 00:00 UTC.** A signal on day D gives a limit order that works during day D+1 (UTC).
* **Honest testing.** Each strategy's small parameter grid is searched on prices before 2023 only, then judged
  once on everything after. A strategy trades live only while it passes the bar in `engine/strategies.py`
  (`ENABLE_RULE`), overall or as a specialist in named market conditions.
* **After fees.** Limit orders pay the maker fee (0.50%), stops and market exits the taker fee (0.90%) plus
  0.10% slippage. A limit counts as filled only when the day's low trades below it. If a day touches both the
  stop and the target, the stop is assumed first.
* **Market conditions.** Bull or bear by Bitcoin against its 200-day average; calm or stressed by its 7-day
  move and volatility. Every strategy is measured in each of the four, and graded there.
* **A+ = evidence.** A setup is A+ when its strategy's record in today's market condition shows 66%+ wins
  after fees on 25+ trades over 10+ separate days, and it held up on unseen data. The daily review blends the
  desk's own graded alerts into those odds, so grading keeps learning.
* **Same-day signals move together.** Results are also counted by signal day, and the A+ rule needs separate days.

## Commands

```bash
git clone --depth 1 -b data https://github.com/aznh0pe/crypto-swing-desk data
python3 -m engine.calibrate backtest --data data --out bt            # full calibration
python3 -m engine.scan run --mode close --data data --state S --out out
python3 -m engine.learn review --data data --state S --out out
python3 -m engine.research next --state S --out rs                   # what to test next + lessons
python3 -m engine.research test --spec idea.json --data data --state S --out rs
python3 -m engine.research states --spec base.json --data data --out st   # where does a simple trade win?
python3 -m engine.research promote --data data --state S --out promo
python3 tests/test_smoke.py
```

`S` is a folder of dashboard documents, one file per document: `S/<collection>/<doc id>.json`.
New ideas are plain JSON rule specs; `engine/rules.py` documents the format.

## After changing the engine or the dashboard

1. `python3 ops/publish_engine.py` prints the files to publish with the dashboard (the engine copy the
   scheduled runs read) and rewrites `ops/engine_manifest.json`. Republish the dashboard with them.
2. After editing a prompt in `ops/build_prompts.py`, run it and update the scheduled task with the generated file.


## What the 2026-11-16 audit changed (and the test that keeps each fix in place)

`python3 tests/test_regressions.py` holds one test per fault found.

- **Coin list**: from 32 coins ($2M+ a day) to every normally trading Coinbase USD pair above $150K a day (143 on the day of the audit): a core tier ($2M+) and a wider tier graded on its own record with harsher fills and slippage. The list is rebuilt from fresh volumes at every scan; the weekly calibration re-tests the floor (`calibrate.floor_probe`) and moves it one step when the evidence says so.
- **No cap**: every A+ setup is served as a full ticket. What the account cannot fund is still shown, marked, in rank order.
- **Ranking**: when several A+ strategies flag the same coin on the same day it ranks first (72% wins alone, 87% with two, 91% with three or more on unseen data).
- **Grading**: an A+ cell must also win on most of its signal days (same-day setups move together).
- **Radar and pace**: each close scan lists coins one condition short of an A+ signal, the market conditions strategies wait on, and the week's A+ count against the minimum of 5.
- **Reliability**: feed loop + watchdog; a check run catches up on new setups when the close scan was missed, prices were late or the strategy book changed; the daily review and the dashboard list every scheduled run that did not happen.
- **Fixed faults**: target hits read from the fill day's high on daily bars; order sizes not floored to the exchange's step; backtest trades dropped when a coin was delisted; the journal filed under the run date instead of the trading day; calibration ignoring the account's real fee settings.
- **Tested and rejected** (kept in the research log so nobody repeats them): top-ups from grade-A setups (won 43 to 45%), flushes of a coin falling alone, resting deep bids, range lows, streak and rebound-confirmation entries in quiet markets.

A push that touches `pipeline/` restarts the feed loop with the new code within about ten minutes.

## Alarms are for real problems only (2026-11-17)

A scan raised a red "Problem in the last scan" banner because a routine news check needed a second web search. The run was fine. Since then:
- Scheduled runs keep two lists. ISSUES = something ended up wrong, late, missing or unsaved, or needs repair: red banner, "run problem" in the headline, phone alert. NOTES = how the run went when the result is unaffected (a retry that worked, extra searches): filed at the foot of the digest via `engine.scan news --note`, never an alarm.
- The news check allows up to three searches and three page reads per coin as normal procedure, and has an `unclear` verdict for when it still cannot tell.
- A code release no longer makes the next check run look for setups again: the catch-up stamp (`scan.book_id`) follows strategy settings, grades and the coin-list floor only, and a catch-up never re-announces a setup that was already served.
- Tests 7b to 7d in `tests/test_regressions.py` hold all of this in place.

