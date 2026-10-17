"""The simulator.

The one rule that keeps it honest:
    A strategy decides target weights using data up to the CLOSE of day t.
    Those trades fill at the OPEN of day t+1. Nothing else is allowed.

So a position decided on day t earns the open(t+1) -> open(t+2) return.
Weights are fractions of equity per ticker (0..1, long-only); whatever is
left over sits in cash earning 0%.
"""
import numpy as np
import pandas as pd


def run(weights, opens, cost_bps=5.0, delay=1):
    """Simulate a weights DataFrame. Returns a daily Series of net returns.

    cost_bps: cost per unit of turnover (commission + spread + slippage), each way.
    delay:    days between decision and fill. 1 is the honest default;
              2 is a stress test; 0 is CHEATING (trades at a price you couldn't get).
    """
    weights = weights.reindex(opens.index).fillna(0.0)
    if (weights < -1e-9).any().any() or (weights.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("weights must be long-only and sum to <= 1")

    # Return from today's open to tomorrow's open.
    open_to_open = opens.shift(-1) / opens - 1
    # Position held from today's open = what was decided `delay` closes ago.
    # With delay=0 you hold from this morning's open a position based on tonight's close: look-ahead.
    held = weights.shift(delay).fillna(0.0)

    gross = (held * open_to_open).sum(axis=1)
    turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    net = net.iloc[:-1]  # last day has no next open
    net.attrs["turnover"] = turnover.iloc[:-1]
    net.attrs["exposure"] = held.sum(axis=1).iloc[:-1]
    return net


def stats(r):
    """Headline metrics for a daily return Series."""
    r = r.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / 252
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min()
    out = {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": max_dd, "Total": equity.iloc[-1] - 1}
    if "turnover" in r.attrs:
        out["Turnover/yr"] = r.attrs["turnover"].loc[r.index].sum() / years
        out["Exposure"] = r.attrs["exposure"].loc[r.index].mean()
    return out
