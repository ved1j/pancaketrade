"""python -m stocksim.run [--refresh] [--cost 5]"""
import argparse

import pandas as pd

from . import backtest, checks, data
from .strategies import STRATEGIES

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "SHY"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    p.add_argument("--cost", type=float, default=5.0, help="cost in bps per unit turnover")
    p.add_argument("--start", default="2005-01-01")
    args = p.parse_args()

    opens, close = data.load(UNIVERSE, start=args.start, refresh=args.refresh)
    print(f"Data: {close.index[0].date()} -> {close.index[-1].date()}, {len(close)} days, {', '.join(UNIVERSE)}")
    print(f"Fills at next open, {args.cost:g} bps cost per trade, cash earns 0%\n")

    perf, sanity = {}, {}
    for name, fn in STRATEGIES.items():
        w = fn(close)
        r = backtest.run(w, opens, args.cost)
        perf[name] = backtest.stats(r)
        # Always-invested strategies have no timing to compare against luck.
        luck = None if perf[name]["Exposure"] > 0.99 else checks.random_sharpes(
            close, opens, perf[name]["Exposure"], cost_bps=args.cost)
        sanity[name] = checks.run_all(w, close, opens, args.cost, luck)

    pct = ["CAGR", "Vol", "MaxDD", "Total", "Exposure"]
    t = pd.DataFrame(perf).T
    for c in pct:
        t[c] = t[c].map("{:.1%}".format)
    t["Sharpe"] = t["Sharpe"].map("{:.2f}".format)
    t["Turnover/yr"] = t["Turnover/yr"].map("{:.1f}x".format)
    print("PERFORMANCE")
    print(t.to_string(), "\n")

    s = pd.DataFrame(sanity).T
    print("SANITY CHECKS (Sharpe under each condition)")
    print(s.round(2).to_string())
    print("\nbeats luck % = share of 300 random in/out strategies (same exposure) with a lower Sharpe.")


if __name__ == "__main__":
    main()
