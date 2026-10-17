"""Strategies. Each takes the close-price DataFrame and returns target weights
(same shape) using only data up to and including each row's close."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

_NEXT_TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def is_month_end(index):
    """True on the last trading day of each month, decided without peeking at
    future rows: a day is month-end if the next trading day is in a new month.
    (Grouping by month and taking the max date marks the latest row of an
    unfinished month as month-end, which made live runs re-rank every day.)"""
    nxt = index[1:].append(pd.DatetimeIndex([index[-1] + _NEXT_TRADING_DAY]))
    return pd.Series(nxt.month != index.month, index=index)


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


CASH = "SHY"  # 1-3y Treasury ETF: where "cash" can sit and still earn interest


def momentum(close, lookbacks=(126,), top=2, cash=None, inv_vol=False, vol_window=63):
    """Once a month: score each asset by its average return over `lookbacks`,
    hold the top `top` whose score is positive; empty slots go to `cash`
    (a ticker) or real cash (None).

    inv_vol: within the picks, size by 1/volatility instead of equally, so a
    jumpy asset doesn't dominate the risk. Total invested share is unchanged.
    """
    assets = [c for c in close.columns if c != CASH]
    px = close[assets]
    score = sum(px / px.shift(lb) - 1 for lb in lookbacks) / len(lookbacks)
    vol = px.pct_change().rolling(vol_window).std()
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for d in close.index[is_month_end(close.index)]:
        s = score.loc[d].dropna()
        if len(s) < len(assets) or vol.loc[d].isna().any():
            continue
        picks = s.nlargest(top)
        picks = picks[picks > 0].index
        row = pd.Series(0.0, index=close.columns)
        if len(picks):
            raw = 1 / vol.loc[d, picks] if inv_vol else pd.Series(1.0, index=picks)
            row[picks] = raw / raw.sum() * len(picks) / top
        if cash:
            row[cash] = 1 - row.sum()
        w.loc[d] = row
    return w.ffill().fillna(0.0)


def momentum_rotation(close, lookback=126, top=2):
    """The original: 6-month momentum, top 2 equal weight, leftovers in cash."""
    return momentum(close, lookbacks=(lookback,), top=top)


BLEND = (21, 63, 126, 252)  # 1, 3, 6, 12 months


def momentum_v2(close):
    """Top 3 by blended 1/3/6/12-month momentum, sized by inverse volatility,
    unused slots parked in SHY. Each piece was checked across nearby settings:
    holding 3 instead of 2 and inverse-vol sizing helped everywhere; the blend
    mainly removes dependence on one lucky lookback."""
    return momentum(close, lookbacks=BLEND, top=3, cash=CASH, inv_vol=True)


def inverse_vol_all(close, vol_window=63):
    """Benchmark: hold every risky asset, sized by 1/volatility. No prediction at all.
    A momentum strategy only earns its complexity if it beats this."""
    assets = [c for c in close.columns if c != CASH]
    iv = 1 / close[assets].pct_change().rolling(vol_window).std()
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[assets] = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    return w


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
    "inverse_vol_all": lambda c: inverse_vol_all(c),
    "momentum_v2": lambda c: momentum_v2(c),
}
