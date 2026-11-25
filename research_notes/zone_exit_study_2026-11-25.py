import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/crypto-swing-desk")
from engine import indicators as I, backtest as bt
cap = pickle.load(open("cap.pkl", "rb"))
g, zones, ind = cap["g_in"], cap["zones"], cap["ind"]
C, H, L = (ind[k].values for k in ("close", "high", "low"))
dates = ind["close"].index
zday = np.array([I.zone_of(x) for x in ind["market"]["btc_dist200"].reindex(dates).values])
exempt = set((zones["exempt"].get("slide") or {}).keys())
aside = {s.split(": ")[1] for s in zones["set_aside"] if s.startswith("slide: ")}
ap = bt.done(g[g["grade"] == "A+"]).copy()
ap = ap[ap["reason"] != "mark"]
maker, taker = 0.005, 0.009
rows = []
for r in ap.itertuples():
    key = f"{r.strategy}|{r.tier}"
    if key in exempt:
        continue
    slip = 0.001 if r.tier == "core" else 0.0035
    f, x, j = int(r.fill_i), int(r.exit_i), int(r.j)
    ev = [d for d in range(f, x) if zday[d] == "slide"]      # scans after a close with the trade still open
    if not ev:
        continue
    e = ev[0]
    fee_in = maker
    cost = r.fill * (1 + fee_in)
    even = r.fill * (1 + maker) / (1 - taker)
    now = C[e, j] * (1 - slip) * (1 - taker) / cost - 1                      # sell at market after the scan
    if C[e, j] >= even:
        rule = now
    elif e + 1 <= x and L[e + 1, j] <= r.stop:
        rule = r.stop * (1 - slip) * (1 - taker) / cost - 1
    elif H[e + 1, j] >= even:
        rule = even * (1 - maker) / cost - 1                                  # resting limit at break-even fills
    else:
        rule = C[e + 1, j] * (1 - slip) * (1 - taker) / cost - 1
    sigz = zday[int(r.t)]
    rows.append({"key": key, "grp": "set-aside (wins in zone)" if key in aside else "rest (no own edge)", "how": "signalled in zone" if sigz == "slide" else "slid in mid-hold",
                 "period": "train" if r.signal_date < pd.Timestamp("2023-01-01") else "unseen", "day": r.signal_date, "hold": r.ret, "now": now, "rule": rule, "at_fill_day": e == f})
d = pd.DataFrame(rows)
pd.set_option("display.width", 250)
def summ(x):
    return pd.Series({"n": len(x), "days": x["day"].nunique(), "hold_win": (x["hold"] > 0).mean(), "hold_avg": x["hold"].mean(), "now_avg": x["now"].mean(), "rule_win": (x["rule"] > 0).mean(), "rule_avg": x["rule"].mean(),
                      "hold_minus_rule": (x["hold"] - x["rule"]).mean(), "day_hold_better": (x.groupby("day").apply(lambda y: (y["hold"] - y["rule"]).mean()) > 0).mean()})
print(d.groupby(["grp"]).apply(summ).round(4))
print(d.groupby(["grp", "how"]).apply(summ).round(4))
print(d.groupby(["grp", "period"]).apply(summ).round(4))
print(d.groupby(["grp", "key"]).apply(summ).round(4).to_string())
d.to_pickle("d.pkl")
