"""Strategies. Each takes the close-price DataFrame and returns target weights
(same shape) using only data up to and including each row's close."""
import numpy as np
import pandas as pd


def buy_and_hold(close, ticker="SPY"):
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[ticker] = 1.0
    return w


def trend_filter(close, ticker="SPY", window=200):
    """Hold the ticker while it's above its N-day average, otherwise sit in cash."""
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sma = close[ticker].rolling(window).mean()
    w[ticker] = (close[ticker] > sma).astype(float)
    return w


def sma_cross(close, ticker="SPY", fast=50, slow=200):
    """Classic golden cross: long while fast average > slow average."""
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[ticker] = (close[ticker].rolling(fast).mean() > close[ticker].rolling(slow).mean()).astype(float)
    return w


def momentum_rotation(close, lookback=126, top=2):
    """Once a month: rank assets by their past `lookback`-day return, hold the
    top `top` equally, but only the ones whose return is positive (rest -> cash)."""
    ret = close / close.shift(lookback) - 1
    month_end = close.index.to_series().groupby(close.index.to_period("M")).transform("max") == close.index
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for d in close.index[month_end]:
        r = ret.loc[d].dropna()
        if len(r) < len(close.columns):
            continue
        picks = r.nlargest(top)
        row = pd.Series(0.0, index=close.columns)
        row[picks[picks > 0].index] = 1.0 / top
        w.loc[d] = row
    return w.ffill().fillna(0.0)


def random_timing(close, ticker="SPY", exposure=0.7, hold_days=21, seed=0):
    """Coin-flip strategy: every ~month, randomly be in or out. A baseline for luck."""
    rng = np.random.default_rng(seed)
    n = len(close)
    blocks = rng.random(n // hold_days + 1) < exposure
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[ticker] = np.repeat(blocks, hold_days)[:n].astype(float)
    return w


STRATEGIES = {
    "buy_hold_SPY": buy_and_hold,
    "trend_200d_SPY": trend_filter,
    "sma_50_200_SPY": sma_cross,
    "momentum_rotation": momentum_rotation,
}
