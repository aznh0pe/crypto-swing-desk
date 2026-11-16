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
| Price feed | `.github/workflows/crypto-data.yml`, `pipeline/` | GitHub Action, asked every 15 minutes around the clock (GitHub lets about one an hour through at worst). Coinbase Exchange public API, no keys: daily bars since 2017 for every USD pair (delisted ones included), hourly bars for the tradable ones, the Fear & Greed index. Publishes the `data` branch as one squashed commit. Sunday: full history rebuild. |
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
