# LEAPS Stock Screener

A Streamlit dashboard that screens US-listed stocks for long-call LEAPS candidates using free market data.

The first version focuses on:

- US stock universe from free Nasdaq Trader symbol directories
- Historical prices from Yahoo Finance through `yfinance`
- RSI, MACD, Bollinger Bands, bounce-history, and optional fundamentals signals
- Score out of 100 for LEAPS-style call ideas
- Local watchlist saved to `data/watchlist.json`

This is research software, not financial advice. Free data can be delayed, incomplete, rate-limited, or revised.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Practical Usage

Screening every US stock can take a while and may hit free-data rate limits. Start with a smaller limit in the sidebar, tune the filters, then increase the universe size once things look right.

For LEAPS, the score intentionally favors beaten-down but improving setups:

- RSI below 40
- Price near the lower Bollinger Band
- MACD line close to crossing above signal
- Prior bounces after lower-band touches
- Positive revenue or earnings growth when available

## Later: Signal Alerts

Signal alerts are not wired in yet. A likely implementation is `signal-cli`, called from an alert module after the scanner identifies new high-score names.
"# stock_screener" 
