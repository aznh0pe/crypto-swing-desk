"""Data-driven strategy specs: how every strategy (built-in or researched) becomes an order plan.

A spec is plain JSON, so a new idea can be written, tested and promoted without code changes:

{
  "id": "R_DIP_IN_UPTREND",                   capitals/digits/underscores (research ideas start with R_)
  "label": "Dip in an uptrend",
  "family": "Mean reversion",
  "about": "One or two plain-English sentences.",
  "source": {"title": "...", "url": "https://..."},
  "universe": "all",                          all | majors (top 12 by dollar volume) | alts (the rest)
  "conditions": ["close > sma200", "rsi2 < 10", "btc_above200 > 0.5"],
  "entry": {"kind": "limit_next", "price": "close - {k}*atr10"},   or {"kind": "open_next"} (market order at the next bar's open)
  "stop": {"atr": "{stop}"},                  or {"price": "ll20 - 0.25*atr10"}
  "target": {"atr": 2.0},                     or {"price": "sma20", "min_atr": 0.5, "cap_atr": 5}
  "exit": "sma5",                             sma5 | prev_high | high7 | rsi2_70 | ema10_lost | none
  "max_hold": "{hold}", "valid_days": 1,
  "regime": [],                               optional gates: btc200, breadth, calm
  "market_conditions": [],                    optional: trade only in these market conditions (bull_calm,
                                              bull_stress, bear_calm, bear_stress)
  "params": {"k": 0.3, "stop": 3.0, "hold": 7},
  "grid": {"k": [0, 0.3, 0.6], "hold": [5, 8]}
}

Bars are daily and close at 00:00 UTC; crypto trades every day, so "next session" is the next UTC day.
Expressions may use indicator frames (FRAME_NAMES), market series (MARKET_NAMES), calendar fields
(dow 0=Mon..6=Sun, dom = day of the month), numbers, + - * / and parentheses, lags x[n] (n days ago),
rmin / rmax / rmean / rsum(x, n), min(a, b), max(a, b) and abs(x). Conditions are comparisons
joined with and / or / not; all conditions must hold. {name} placeholders are filled from
params (defaults) or the grid (variants tried on the training years only).
"""
from __future__ import annotations

import ast
import itertools
import json
import re

import numpy as np
import pandas as pd


FRAME_NAMES = {"open", "high", "low", "close", "volume", "sma5", "sma10", "sma20", "sma50", "sma100", "sma200",
               "ema10", "ema20", "ema50", "rsi2", "rsi3", "rsi14", "atr10", "atr14", "atr_pct", "ibs", "pctb", "min7c",
               "max7c", "hh10", "hh20", "hh55", "hh90", "ll10", "ll20", "ll55", "streak", "upstreak", "slope200", "slope50",
               "dist200", "dist50", "dist20", "v20", "v5", "dv30", "vol_ratio", "ret1", "ret3", "ret7", "ret14", "ret30",
               "ret90", "adx14", "vwap20", "vstd20", "rs_pct", "rs30_pct", "dd90", "vol_rank", "is_major", "range",
               "nr7", "inside", "body", "upper_wick", "lower_wick", "age", "is_core"}
MARKET_NAMES = {"btc", "btc_sma200", "btc_sma50", "btc_above200", "btc_above50", "btc_dist200", "btc_dist50", "btc_ret1",
                "btc_ret7", "btc_ret30", "btc_rsi2", "btc_rsi14", "btc_vol_ratio", "btc_dd90", "breadth50", "breadth200",
                "breadth20", "alt_ret7", "eth_btc_up", "fng", "stress", "bull", "washed", "flush7", "up7"}
CAL_NAMES = {"dow", "dom"}
FUNCS = {"rmin": 2, "rmax": 2, "rmean": 2, "rsum": 2, "min": 2, "max": 2, "abs": 1}
EXITS = {"sma5", "prev_high", "high7", "rsi2_70", "ema10_lost", "none"}
UNIVERSES = {"all", "majors", "alts"}
REGIMES = {"btc200", "breadth", "calm"}
ID_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,40}$")
PH_RE = re.compile(r"\{(\w+)\}")
MAX_COMBOS = 36


# ----------------------------------------------------------------------------- params
def subst(x, params: dict):
    """Fill {name} placeholders. Numbers stay numbers."""
    if isinstance(x, str):
        def rep(m):
            k = m.group(1)
            if k not in params:
                raise ValueError(f"placeholder {{{k}}} has no value in params")
            return repr(params[k]) if isinstance(params[k], str) else str(params[k])
        out = PH_RE.sub(rep, x)
        return out
    return x


def _num(x, params, cast=float):
    v = subst(x, params)
    return cast(float(v))


def combos(spec: dict) -> list[dict]:
    base = dict(spec.get("params") or {})
    grid = spec.get("grid") or {}
    keys = [k for k, v in grid.items() if isinstance(v, list) and v]
    if not keys:
        return [base]
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        out.append({**base, **dict(zip(keys, vals))})
    return out[:MAX_COMBOS]


# ----------------------------------------------------------------------------- evaluator
class Evaluator:
    """Evaluates rule expressions over the indicator frames, caching every sub-result."""

    def __init__(self, ind: dict):
        self.ind = ind
        self.c = ind["close"]
        self.cache: dict = {}
        self._cal = None

    # fields
    def _bcast(self, s: pd.Series) -> pd.DataFrame:
        v = s.reindex(self.c.index).astype(float).values
        return pd.DataFrame(np.repeat(v[:, None], self.c.shape[1], axis=1), index=self.c.index, columns=self.c.columns)

    def _calendar(self) -> pd.DataFrame:
        if self._cal is None:
            idx = self.c.index
            self._cal = pd.DataFrame({"dow": idx.dayofweek.astype(float), "dom": idx.day.astype(float)}, index=idx)
        return self._cal

    def field(self, name: str) -> pd.DataFrame:
        key = "f:" + name
        if key in self.cache:
            return self.cache[key]
        ind = self.ind
        if name == "range":
            f = ind["high"] - ind["low"]
        elif name in ("is_major", "is_core"):
            f = ind[name].astype(float)
        elif name in FRAME_NAMES and name in ind:
            f = ind[name]
        elif name in MARKET_NAMES and name in ind["market"].columns:
            f = self._bcast(ind["market"][name])
        elif name in CAL_NAMES:
            f = self._bcast(self._calendar()[name])
        else:
            raise ValueError(f"unknown field '{name}'")
        self.cache[key] = f
        return f

    # expressions
    def value(self, expr: str):
        expr = expr.strip()
        if expr in self.cache:
            return self.cache[expr]
        tree = ast.parse(expr, mode="eval")
        v = self._eval(tree.body)
        if len(self.cache) > 80:
            for k in [k for k in self.cache if not k.startswith("f:")][:40]:
                self.cache.pop(k, None)
        self.cache[expr] = v
        return v

    def mask(self, expr: str) -> pd.DataFrame:
        v = self.value(expr)
        if isinstance(v, (bool, np.bool_)):
            return pd.DataFrame(bool(v), index=self.c.index, columns=self.c.columns)
        if isinstance(v, pd.DataFrame):
            if v.dtypes.apply(lambda d: d == bool).all():
                return v
            return v.fillna(0) != 0
        raise ValueError(f"'{expr}' is not a condition")

    def frame(self, expr: str) -> pd.DataFrame:
        v = self.value(expr)
        if isinstance(v, pd.DataFrame):
            return v.astype(float)
        return pd.DataFrame(float(v), index=self.c.index, columns=self.c.columns)

    def _eval(self, n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name):
            return self.field(n.id)
        if isinstance(n, ast.Subscript):
            sl = n.slice
            if not (isinstance(sl, ast.Constant) and isinstance(sl.value, int) and 0 <= sl.value <= 260):
                raise ValueError("a lag must be a whole number of days, e.g. close[1]")
            base = self._eval(n.value)
            return base.shift(sl.value) if isinstance(base, pd.DataFrame) else base
        if isinstance(n, ast.UnaryOp):
            v = self._eval(n.operand)
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.UAdd):
                return v
            if isinstance(n.op, ast.Not):
                return ~self._as_bool(v)
        if isinstance(n, ast.BinOp):
            a, b = self._eval(n.left), self._eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                if isinstance(b, pd.DataFrame):
                    b = b.replace(0, np.nan)
                elif b == 0:
                    return a * np.nan
                return a / b
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS:
            fn = n.func.id
            if len(n.args) != FUNCS[fn] or n.keywords:
                raise ValueError(f"{fn}() takes {FUNCS[fn]} argument(s)")
            args = [self._eval(a) for a in n.args]
            if fn in ("rmin", "rmax", "rmean", "rsum"):
                x, w = args
                w = int(w)
                if not 1 <= w <= 260 or not isinstance(x, pd.DataFrame):
                    raise ValueError(f"{fn}(x, n) needs a field and 1 <= n <= 260")
                r = x.rolling(w, min_periods=w)
                return {"rmin": r.min, "rmax": r.max, "rmean": r.mean, "rsum": r.sum}[fn]()
            if fn == "abs":
                return abs(args[0])
            a, b = args
            if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
                fa = a if isinstance(a, pd.DataFrame) else pd.DataFrame(a, index=self.c.index, columns=self.c.columns)
                return np.minimum(fa, b) if fn == "min" else np.maximum(fa, b)
            return min(a, b) if fn == "min" else max(a, b)
        if isinstance(n, ast.Compare):
            left = self._eval(n.left)
            res = None
            for op, comp in zip(n.ops, n.comparators):
                right = self._eval(comp)
                if isinstance(op, ast.Lt):
                    r = left < right
                elif isinstance(op, ast.LtE):
                    r = left <= right
                elif isinstance(op, ast.Gt):
                    r = left > right
                elif isinstance(op, ast.GtE):
                    r = left >= right
                elif isinstance(op, ast.Eq):
                    r = left == right
                elif isinstance(op, ast.NotEq):
                    r = left != right
                else:
                    raise ValueError("unsupported comparison")
                r = self._as_bool(r)
                res = r if res is None else (res & r)
                left = right
            return res
        if isinstance(n, ast.BoolOp):
            vals = [self._as_bool(self._eval(v)) for v in n.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(n.op, ast.And) else (out | v)
            return out
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    def _as_bool(self, v):
        if isinstance(v, pd.DataFrame):
            return v if v.dtypes.apply(lambda d: d == bool).all() else (v.fillna(0) != 0)
        return pd.DataFrame(bool(v), index=self.c.index, columns=self.c.columns)


_EVALS: dict = {}


def evaluator(ind: dict) -> Evaluator:
    key = id(ind)
    ev = _EVALS.get(key)
    if ev is None or ev.ind is not ind:
        _EVALS.clear()
        ev = _EVALS[key] = Evaluator(ind)
    return ev


# ----------------------------------------------------------------------------- plans
def build_plan(spec: dict, ind: dict, P: dict, b: dict, params: dict | None = None, name: str | None = None) -> dict:
    pr = {**(spec.get("params") or {}), **(params or {})}
    ev = evaluator(ind)
    c = ind["close"]
    uni = spec.get("universe", "all")
    m = b["liquid"].copy()
    if uni == "majors":
        m = m & b["is_major"]
    elif uni == "alts":
        m = m & ~b["is_major"]
    for cond in spec["conditions"]:
        m = m & ev.mask(subst(cond, pr))
    reg = spec.get("regime") or []
    if isinstance(reg, str):
        reg = [reg]
    if reg:
        mk = ind["market"]
        gate = pd.Series(True, index=c.index)
        if "btc200" in reg:
            gate &= mk["btc_above200"] >= 0.5
        if "breadth" in reg:
            gate &= mk["breadth50"].fillna(0.5) >= 0.35
        if "calm" in reg:
            gate &= mk["stress"].fillna(0) < 0.5
        m = m & pd.DataFrame(np.repeat(gate.values[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    if spec.get("market_conditions"):  # a specialist: signals only in these market conditions
        from .indicators import condition_gate
        m = m & condition_gate(ind, spec["market_conditions"], c)
    en = spec.get("entry") or {"kind": "limit_next", "price": "close"}
    kind = en.get("kind", "limit_next")
    limit = c.copy() if kind == "open_next" else ev.frame(subst(en.get("price", "close"), pr))
    st = spec.get("stop") or {"atr": 3.0}
    stop = ({"kind": "atr", "mult": _num(st["atr"], pr)} if "atr" in st
            else {"kind": "level", "frame": ev.frame(subst(st["price"], pr))})
    tg = spec.get("target") or {"atr": 1.0}
    target = ({"kind": "atr", "mult": _num(tg["atr"], pr)} if "atr" in tg
              else {"kind": "level", "frame": ev.frame(subst(tg["price"], pr)),
                    "min_atr": _num(tg.get("min_atr", 0.25), pr), "cap_atr": _num(tg.get("cap_atr", 4.0), pr)})
    dyn = str(subst(spec.get("exit", "none"), pr)).strip("'\"")
    return {"name": name or spec["id"], "entry_kind": "open_next" if kind == "open_next" else "limit_next",
            "valid_days": _num(spec.get("valid_days", 1), pr, int), "max_hold": _num(spec.get("max_hold", 7), pr, int),
            "dyn_exit": dyn if dyn in EXITS else "none", "stop": stop, "target": target,
            "mask": m.fillna(False).astype(bool), "limit": limit}


FIELD_WORDS = {
    "alt_ret7": "alt coins' 7-day change", "btc_ret7": "Bitcoin's 7-day change", "btc_ret1": "Bitcoin's 1-day change", "btc_ret30": "Bitcoin's 30-day change",
    "flush7": "share of coins down 15%+ in a week", "washed": "share of coins washed out", "up7": "share of coins up on the week",
    "breadth20": "share of coins above their 20-day average", "breadth50": "share of coins above their 50-day average", "breadth200": "share of coins above their 200-day average",
    "btc_dist200": "Bitcoin vs its 200-day average", "btc_dist50": "Bitcoin vs its 50-day average", "btc_dd90": "Bitcoin's drop from its 90-day high",
    "btc_rsi2": "Bitcoin's RSI(2)", "btc_rsi14": "Bitcoin's RSI(14)", "btc_vol_ratio": "Bitcoin's volatility vs normal", "fng": "Fear & Greed",
    "rsi2": "RSI(2)", "rsi3": "RSI(3)", "rsi14": "RSI(14)", "ret1": "1-day change", "ret3": "3-day change", "ret5": "5-day change", "ret7": "7-day change",
    "ret14": "14-day change", "ret30": "30-day change", "ret90": "90-day change", "dd90": "drop from its 90-day high", "dist20": "distance from its 20-day average",
    "dist50": "distance from its 50-day average", "dist200": "distance from its 200-day average", "ibs": "close within the day's range", "streak": "streak of up/down days",
    "atr_pct": "daily range", "vol_ratio": "volume vs its 20-day average", "rs_pct": "90-day strength rank", "rs30_pct": "30-day strength rank", "adx14": "trend strength (ADX)",
    "pctb": "position in its Bollinger band", "close": "close", "low": "low", "high": "high",
}
PCT_FIELDS = {k for k in FIELD_WORDS if k.startswith(("ret", "dist", "dd", "alt_ret", "btc_ret", "btc_dist", "btc_dd", "breadth")) or k in ("flush7", "washed", "up7", "atr_pct")}
OP_WORDS = {ast.Lt: "below", ast.LtE: "at or below", ast.Gt: "above", ast.GtE: "at or above"}


def _fmt_field(name: str, v: float) -> str:
    if name in PCT_FIELDS:
        return f"{100 * v:+.1f}%" if name.startswith(("ret", "dist", "alt_ret", "btc_ret", "btc_dist", "dd", "btc_dd")) else f"{100 * v:.0f}%"
    return f"{v:.3g}"


def plain_condition(txt: str) -> tuple[str, str | None]:
    """('7-day change at or below -30.0%', 'ret7') for a simple one-field comparison; (the raw text, None) otherwise."""
    try:
        tree = ast.parse(txt, mode="eval").body
        if isinstance(tree, ast.Compare) and len(tree.ops) == 1 and isinstance(tree.left, ast.Name) and tree.left.id in FIELD_WORDS:
            rhs = ast.literal_eval(tree.comparators[0])
            if isinstance(rhs, (int, float)) and type(tree.ops[0]) in OP_WORDS:
                return f"{FIELD_WORDS[tree.left.id]} {OP_WORDS[type(tree.ops[0])]} {_fmt_field(tree.left.id, float(rhs))}", tree.left.id
    except Exception:
        pass
    return txt, None


def near_misses(spec: dict, ind: dict, b: dict, params: dict | None = None) -> dict:
    """What the last bar is missing for this strategy, for the dashboard's radar:
    "coins"  coins that pass every condition but one (with what is missing and where it stands now)
    "market" the market-wide conditions that are not met (false for every coin)
    "ready"  coins that already pass every coin-level condition and wait only on those market conditions"""
    pr = {**(spec.get("params") or {}), **(params or {})}
    ev = evaluator(ind)
    base = b["liquid"].iloc[-1].copy()
    uni = spec.get("universe", "all")
    if uni == "majors":
        base &= b["is_major"].iloc[-1]
    elif uni == "alts":
        base &= ~b["is_major"].iloc[-1]
    names = base.index[base.values]
    if not len(names):
        return {"coins": [], "market": [], "ready": []}

    def now_text(field, tk):
        if not field:
            return ""
        try:
            v = float(ev.frame(field).iloc[-1][tk])
            return f"now {_fmt_field(field, v)}" if np.isfinite(v) else ""
        except Exception:
            return ""
    market, per, fails = [], [], pd.Series(0, index=names)
    for cond in spec["conditions"]:
        txt = subst(cond, pr)
        m = ev.mask(txt).iloc[-1].reindex(names).fillna(False).astype(bool)
        words, field = plain_condition(txt)
        if not m.any() and not (_names(ast.parse(txt, mode="eval"))[0] & FRAME_NAMES):
            market.append({"needs": words, "now": now_text(field, names[0])})      # market-wide, false for every coin
            continue
        per.append((words, field, m))
        fails = fails + (~m).astype(int)
    if market:
        return {"coins": [], "market": market, "ready": list(names[(fails == 0).values])}
    rows = []
    for tk in names[(fails == 1).values]:
        words, field, _ = next(x for x in per if not x[2][tk])
        rows.append({"ticker": tk, "needs": words, "now": now_text(field, tk)})
    return {"coins": rows, "market": [], "ready": []}


def strategy_entry(spec: dict, params: dict, enabled: bool) -> dict:
    """The P['strategies'][id] record the scanner reads for a promoted research strategy."""
    pr = {**(spec.get("params") or {}), **(params or {})}
    en = spec.get("entry") or {}
    return {"enabled": bool(enabled), "custom": True, "universe": spec.get("universe", "all"),
            "entry_kind": "open_next" if en.get("kind") == "open_next" else "limit_next",
            "max_hold": _num(spec.get("max_hold", 7), pr, int), "valid_days": _num(spec.get("valid_days", 1), pr, int),
            "dyn_exit": str(subst(spec.get("exit", "none"), pr)).strip("'\"")}


# ----------------------------------------------------------------------------- checks
def _names(tree) -> tuple[set, set]:
    names, funcs = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            funcs.add(n.func.id)
        elif isinstance(n, ast.Name):
            names.add(n.id)
    return names - funcs, funcs


def validate(spec: dict) -> list[str]:
    errs = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    if not ID_RE.match(str(spec.get("id", ""))):
        errs.append("id must look like SOME_NAME (capitals, digits, underscores)")
    for k in ("label", "family", "about"):
        if not isinstance(spec.get(k), str) or not spec.get(k).strip():
            errs.append(f"'{k}' is required")
    if spec.get("universe", "all") not in UNIVERSES:
        errs.append("universe must be all, majors or alts")
    conds = spec.get("conditions")
    if not isinstance(conds, list) or not conds or len(conds) > 14:
        errs.append("conditions must be a list of 1-14 strings")
        conds = []
    try:
        cs = combos(spec)
    except Exception as e:
        cs, _ = [{}], errs.append(f"grid: {e}")
    n_combos = 1
    for v in (spec.get("grid") or {}).values():
        n_combos *= max(1, len(v) if isinstance(v, list) else 1)
    if n_combos > MAX_COMBOS:
        errs.append(f"grid has {n_combos} combinations; keep it to {MAX_COMBOS} or fewer")
    pr = cs[0] if cs else {}
    exprs = list(conds)
    en = spec.get("entry") or {}
    if en.get("kind", "limit_next") not in ("limit_next", "open_next"):
        errs.append("entry.kind must be limit_next or open_next")
    if en.get("kind", "limit_next") == "limit_next":
        exprs.append(en.get("price", "close"))
    st, tg = spec.get("stop") or {}, spec.get("target") or {}
    if not ("atr" in st or "price" in st):
        errs.append("stop needs 'atr' or 'price'")
    elif "price" in st:
        exprs.append(st["price"])
    if not ("atr" in tg or "price" in tg):
        errs.append("target needs 'atr' or 'price'")
    elif "price" in tg:
        exprs.append(tg["price"])
    reg = spec.get("regime") or []
    if isinstance(reg, str):
        reg = [reg]
    if any(r not in REGIMES for r in reg):
        errs.append("regime gates must be among btc200, breadth, calm")
    mc = spec.get("market_conditions") or []
    if mc and (not isinstance(mc, list) or any(x not in ("bull_calm", "bull_stress", "bear_calm", "bear_stress") for x in mc)):
        errs.append("market_conditions must list among bull_calm, bull_stress, bear_calm, bear_stress")
    allowed = FRAME_NAMES | MARKET_NAMES | CAL_NAMES
    for e in exprs:
        try:
            tree = ast.parse(str(subst(e, pr)), mode="eval")
        except Exception as ex:
            errs.append(f"can't read '{e}': {ex}")
            continue
        names, funcs = _names(tree)
        bad = names - allowed
        if bad:
            errs.append(f"unknown field(s) {sorted(bad)} in '{e}'")
        badf = funcs - set(FUNCS)
        if badf:
            errs.append(f"unknown function(s) {sorted(badf)} in '{e}'")
    try:
        for c in cs:
            dyn = str(subst(spec.get("exit", "none"), c)).strip("'\"")
            if dyn not in EXITS:
                errs.append(f"exit must be one of {sorted(EXITS)}")
                break
            _num(spec.get("max_hold", 7), c, int)
            if "atr" in st:
                _num(st["atr"], c)
            if "atr" in tg:
                _num(tg["atr"], c)
    except Exception as e:
        errs.append(f"params: {e}")
    return errs


def signature(spec: dict) -> str:
    """Rule fingerprint used to spot a duplicate idea under a different name."""
    pr = (combos(spec) or [{}])[0]
    conds = sorted(re.sub(r"\s+", "", str(subst(c, pr))) for c in spec.get("conditions") or [])
    en = spec.get("entry") or {}
    return json.dumps([spec.get("universe", "all"), en.get("kind", "limit_next"), conds])


def describe(spec: dict) -> str:
    """One plain-English line of the rules, for the research log."""
    pr = (combos(spec) or [{}])[0]
    en = spec.get("entry") or {}
    ent = "buy at the next open (market)" if en.get("kind") == "open_next" else f"buy limit at {subst(en.get('price', 'close'), pr)} for the next day"
    st, tg = spec.get("stop") or {}, spec.get("target") or {}
    stop = f"stop {subst(st['atr'], pr)} ATR" if "atr" in st else f"stop at {subst(st.get('price'), pr)}"
    tgt = f"target {subst(tg['atr'], pr)} ATR" if "atr" in tg else f"target {subst(tg.get('price'), pr)}"
    return (f"{spec.get('universe', 'all')} | when " + " and ".join(str(subst(c, pr)) for c in spec.get("conditions") or [])
            + f" | {ent} | {stop}, {tgt}, exit {subst(spec.get('exit', 'none'), pr)}, max {subst(spec.get('max_hold', 7), pr)} days")
