"""Run a strategy against an Alpaca PAPER account.

    python -m stocksim.paper                 # dry run: show the plan, place nothing
    python -m stocksim.paper --submit        # actually place paper orders

Run it once a day after the 4pm ET close. DAY market orders sent after the close
fill at the next open, which is exactly what the backtester assumes.
Keys come from .env (ALPACA_API_KEY / ALPACA_SECRET_KEY). Always paper=True.

Safety checks before anything is sent:
  - market must be closed (intraday runs would trade on a half-finished bar
    and fill immediately instead of at the next open)
  - the signal must use the most recent completed session's close
  - no unfilled orders may be pending (otherwise a rerun double-buys)
  - positions outside the strategy's universe stop the run
"""
import argparse
import csv
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from . import data
from .run import UNIVERSE
from .strategies import STRATEGIES

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "paper_trades.csv"
BUFFER = 0.98      # invest 98% of equity so an overnight gap can't overspend
MIN_TRADE = 0.01   # skip rebalances smaller than 1% of equity (avoids daily 1-share churn)


def target_weights(strategy):
    _, close = data.load(UNIVERSE, refresh=True)
    w = STRATEGIES[strategy](close).iloc[-1]
    return w[w > 0], close.iloc[-1], close.index[-1].date()


def last_session_date(client):
    """Date of the most recent trading session that has already closed."""
    from alpaca.trading.requests import GetCalendarRequest
    now = client.get_clock().timestamp
    days = client.get_calendar(GetCalendarRequest(start=(now - timedelta(days=10)).date(), end=now.date()))
    closed = [d for d in days if d.close.replace(tzinfo=now.tzinfo) <= now]
    return closed[-1].date


def plan_orders(weights, last, equity, held):
    target = {t: math.floor(equity * BUFFER * weights.get(t, 0) / last[t]) for t in UNIVERSE}
    orders = []
    for t in UNIVERSE:
        q = target[t] - int(held.get(t, 0))
        # Always fully exit; otherwise ignore tiny drift trades.
        if q and (target[t] == 0 or abs(q) * last[t] >= MIN_TRADE * equity):
            orders.append((t, q))
    orders.sort(key=lambda o: o[1])  # sells first, frees cash for buys
    return target, orders


def log_orders(rows):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "strategy", "signal_date", "equity", "symbol", "side", "qty", "ref_price", "order_id", "status"])
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="momentum_rotation", choices=list(STRATEGIES))
    p.add_argument("--submit", action="store_true", help="place orders (paper only)")
    p.add_argument("--equity", type=float, help="dry run without keys: pretend account size")
    args = p.parse_args()

    weights, last, asof = target_weights(args.strategy)
    print(f"{args.strategy} signal as of close {asof}: "
          + (", ".join(f"{t} {x:.0%}" for t, x in weights.items()) or "all cash"))

    load_dotenv(ROOT / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not (key and secret):
        if not args.equity:
            raise SystemExit("No keys in .env. Add them, or pass --equity 100000 for a dry run.")
        target, orders = plan_orders(weights, last, args.equity, {})
        for t, q in orders:
            print(f"  {'BUY ' if q > 0 else 'SELL'} {abs(q)} {t}")
        print("Dry run without an account, nothing sent.")
        return

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
    client = TradingClient(key, secret, paper=True)  # never live

    problems = []
    if client.get_clock().is_open:
        problems.append("market is open; run after the 4pm ET close")
    expected = last_session_date(client)
    if asof != expected:
        problems.append(f"signal uses the {asof} close but the latest finished session is {expected} (Yahoo lagging?)")
    pending = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    if pending:
        problems.append("unfilled orders already queued: " + ", ".join(f"{o.side.value} {o.qty} {o.symbol}" for o in pending))
    positions = client.get_all_positions()
    stray = [p.symbol for p in positions if p.symbol not in UNIVERSE]
    if stray:
        problems.append(f"account holds symbols outside the strategy: {', '.join(stray)}")

    equity = float(client.get_account().equity)
    held = {p.symbol: float(p.qty) for p in positions}
    target, orders = plan_orders(weights, last, equity, held)

    print(f"Equity ${equity:,.0f}")
    for t in UNIVERSE:
        if target[t] or held.get(t):
            print(f"  {t}: hold {int(held.get(t, 0))} -> target {target[t]} (~${target[t] * last[t]:,.0f})")
    for t, q in orders:
        print(f"  {'BUY ' if q > 0 else 'SELL'} {abs(q)} {t}")
    if not orders:
        print("Nothing to trade.")
    for msg in problems:
        print(f"BLOCKED: {msg}")
    if not orders or problems:
        return
    if not args.submit:
        print("Dry run, no orders sent. Add --submit to place them on the paper account.")
        return

    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for t, q in orders:
        o = client.submit_order(MarketOrderRequest(
            symbol=t, qty=abs(q), side=OrderSide.BUY if q > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY))
        print(f"  sent {o.side.value} {o.qty} {o.symbol}: {o.status.value}")
        rows.append([now, args.strategy, asof, round(equity, 2), t, o.side.value, o.qty, round(last[t], 2), o.id, o.status.value])
    log_orders(rows)
    print(f"Logged to {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
