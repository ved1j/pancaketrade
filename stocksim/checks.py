"""Sanity checks: ways a backtest can fool you, and a test for each."""
import numpy as np

from . import backtest
from .strategies import random_timing

SPLIT = "2016-01-01"


def sharpe(r):
    return backtest.stats(r)["Sharpe"]


def random_sharpes(close, opens, exposure, n=300, cost_bps=5.0):
    """Sharpe of n coin-flip strategies with the same average exposure: what luck looks like."""
    exposure = float(np.clip(exposure, 0.05, 1.0))
    return np.array([
        sharpe(backtest.run(random_timing(close, exposure=exposure, seed=s), opens, cost_bps))
        for s in range(n)
    ])


def run_all(weights, close, opens, cost_bps=5.0, luck=None):
    base = backtest.run(weights, opens, cost_bps)
    s = sharpe(base)
    out = {
        "Sharpe": s,
        # Costs: an edge that only lives at low costs isn't an edge.
        "2x cost": sharpe(backtest.run(weights, opens, cost_bps * 2)),
        "5x cost": sharpe(backtest.run(weights, opens, cost_bps * 5)),
        # Timing: filling one day later shouldn't destroy a real signal.
        "1d late": sharpe(backtest.run(weights, opens, cost_bps, delay=2)),
        # Look-ahead: what you'd see with the most common backtest bug. Big jump = fragile.
        "CHEAT": sharpe(backtest.run(weights, opens, cost_bps, delay=0)),
        # Overfitting: does it work in both halves of history?
        "pre-2016": sharpe(base[base.index < SPLIT]),
        "2016+": sharpe(base[base.index >= SPLIT]),
    }
    if luck is not None:
        out["beats luck %"] = (luck < s).mean() * 100
    return out
