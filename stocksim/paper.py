"""Run a strategy against an Alpaca PAPER account.

    python -m stocksim.paper                 # dry run: show the plan, place nothing
    python -m stocksim.paper --submit        # actually place paper orders

Run it once a day after the 4pm ET close. DAY market orders sent after the close
fill at the next open, which is exactly what the backtester assumes.
Keys come from .env (ALPACA_API_KEY / ALPACA_SECRET_KEY). Always paper=True.
"""
import argparse
import math
import os

import pandas as pd
from dotenv import load_dotenv

from . import data
from .run import UNIVERSE
from .strategies import STRATEGIES


def target_weights(strategy):
    _, close = data.load(UNIVERSE, refresh=True)
    w = STRATEGIES[strategy](close).iloc[-1]
    return w[w > 0], close.iloc[-1], close.index[-1].date()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="momentum_rotation", choices=list(STRATEGIES))
    p.add_argument("--submit", action="store_true", help="place orders (paper only)")
    p.add_argument("--equity", type=float, help="dry run without keys: pretend account size")
    args = p.parse_args()

    weights, last, asof = target_weights(args.strategy)
    print(f"{args.strategy} signal as of close {asof}: "
          + (", ".join(f"{t} {x:.0%}" for t, x in weights.items()) or "all cash"))
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    if pd.Timestamp(asof) < today - pd.offsets.BDay(1):
        print(f"WARNING: latest usable close is {asof}; Yahoo may not have today's bar yet.")

    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    client = None
    if key and secret:
        from alpaca.trading.client import TradingClient
        client = TradingClient(key, secret, paper=True)  # never live
        equity = float(client.get_account().equity)
        held = {p.symbol: float(p.qty) for p in client.get_all_positions()}
    elif args.equity:
        equity, held = args.equity, {}
    else:
        raise SystemExit("No keys in .env. Add them, or pass --equity 100000 for a dry run.")

    # Use 98% of equity so price moves overnight don't cause a rejected buy.
    target = {t: math.floor(equity * 0.98 * weights.get(t, 0) / last[t]) for t in UNIVERSE}
    orders = [(t, target[t] - int(held.get(t, 0))) for t in UNIVERSE if target[t] != int(held.get(t, 0))]
    orders.sort(key=lambda o: o[1])  # sells first, frees cash for buys

    print(f"Equity ${equity:,.0f}")
    for t in UNIVERSE:
        if target[t] or held.get(t):
            print(f"  {t}: hold {int(held.get(t, 0))} -> target {target[t]} (~${target[t] * last[t]:,.0f})")
    if not orders:
        print("Nothing to trade.")
        return
    for t, q in orders:
        print(f"  {'BUY ' if q > 0 else 'SELL'} {abs(q)} {t}")

    if not args.submit or client is None:
        print("Dry run, no orders sent. Add --submit to place them on the paper account.")
        return
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    for t, q in orders:
        o = client.submit_order(MarketOrderRequest(
            symbol=t, qty=abs(q), side=OrderSide.BUY if q > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY))
        print(f"  sent {o.side.value} {o.qty} {o.symbol}: {o.status.value}")


if __name__ == "__main__":
    main()
