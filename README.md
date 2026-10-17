# stocksim

A small, honest daily-bar stock backtester.

```
python3 -m venv .venv && .venv/bin/pip install pandas numpy yfinance
.venv/bin/python -m stocksim.run            # --refresh re-downloads, --cost 10 for 10 bps
```

| File | What |
|---|---|
| `stocksim/data.py` | Yahoo daily bars (adjusted), cached in `data/` |
| `stocksim/backtest.py` | The simulator: decide at close, fill at next open, pay costs on turnover |
| `stocksim/strategies.py` | Strategies return target weights; add yours to `STRATEGIES` |
| `stocksim/checks.py` | Cost stress, late fills, look-ahead "cheat", pre/post-2016 split, vs random luck |

Rule of thumb: a strategy is interesting only if it beats `buy_hold_SPY` on Sharpe
**and** survives every column of the sanity table.
