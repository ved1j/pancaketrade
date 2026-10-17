"""Download daily bars from Yahoo and cache them as CSV so reruns are offline and identical."""
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent.parent / "data"


def load(tickers, start="2005-01-01", refresh=False):
    """Return (open, close) DataFrames: rows = dates, columns = tickers. Prices are split/dividend adjusted."""
    CACHE.mkdir(exist_ok=True)
    opens, closes = {}, {}
    for t in tickers:
        path = CACHE / f"{t}.csv"
        if refresh or not path.exists():
            df = yf.download(t, start=start, auto_adjust=True, progress=False, multi_level_index=False)
            if df.empty:
                raise RuntimeError(f"no data for {t}")
            df.to_csv(path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[df.index >= start]
        opens[t], closes[t] = df["Open"], df["Close"]
    # Only keep dates where every ticker traded, so nothing is silently forward-filled.
    o = pd.DataFrame(opens).dropna()
    c = pd.DataFrame(closes).loc[o.index]
    return o, c
