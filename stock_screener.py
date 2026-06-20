from __future__ import annotations

import json
import tempfile
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from yfinance import EquityQuery


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
WATCHLIST_PATH = Path("data/watchlist.json")
YFINANCE_CACHE_PATH = Path(tempfile.gettempdir()) / "stock_screener_yfinance_cache"

YFINANCE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_PATH))
try:
    yf.cache.set_cache_location(str(YFINANCE_CACHE_PATH))
except Exception:
    pass


@dataclass(frozen=True)
class ScanConfig:
    period: str = "2y"
    interval: str = "1d"
    chunk_size: int = 80
    min_price: float = 5.0
    min_avg_volume: int = 300_000
    include_etfs: bool = False
    max_symbols: int | None = 500
    universe: str = "largest_market_cap"
    add_fundamentals: bool = False


def load_largest_market_cap_symbols(limit: int = 200) -> pd.DataFrame:
    """Load the largest US equities by market cap from Yahoo's free screener."""
    size = max(1, min(int(limit), 250))
    fetch_size = min(250, size + 50)
    query = EquityQuery(
        "and",
        [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery(
                "or",
                [
                    EquityQuery("eq", ["exchange", "NYQ"]),
                    EquityQuery("eq", ["exchange", "NMS"]),
                    EquityQuery("eq", ["exchange", "NGM"]),
                    EquityQuery("eq", ["exchange", "NCM"]),
                    EquityQuery("eq", ["exchange", "ASE"]),
                ],
            ),
            EquityQuery("gt", ["intradaymarketcap", 0]),
        ],
    )
    response = yf.screen(query, size=fetch_size, sortField="intradaymarketcap", sortAsc=False)
    quotes = response.get("quotes", [])
    listed_symbols = set(load_us_symbols(include_etfs=False)["symbol"])
    rows = [
        {
            "symbol": quote.get("symbol", "").replace(".", "-"),
            "name": quote.get("shortName") or quote.get("longName") or quote.get("symbol", ""),
            "is_etf": "N",
            "is_test": "N",
            "market_cap": quote.get("marketCap") or quote.get("intradaymarketcap"),
        }
        for quote in quotes
        if quote.get("symbol", "").replace(".", "-") in listed_symbols
    ]
    return pd.DataFrame(rows).drop_duplicates("symbol").head(size).reset_index(drop=True)


def load_us_symbols(include_etfs: bool = False) -> pd.DataFrame:
    """Load US-listed symbols from Nasdaq Trader free symbol directories."""
    nasdaq = _read_pipe_file(NASDAQ_LISTED_URL)
    other = _read_pipe_file(OTHER_LISTED_URL)

    nasdaq = nasdaq.rename(
        columns={
            "Symbol": "symbol",
            "Security Name": "name",
            "ETF": "is_etf",
            "Test Issue": "is_test",
        }
    )
    nasdaq = nasdaq[["symbol", "name", "is_etf", "is_test"]]

    other = other.rename(
        columns={
            "ACT Symbol": "symbol",
            "Security Name": "name",
            "ETF": "is_etf",
            "Test Issue": "is_test",
        }
    )
    other = other[["symbol", "name", "is_etf", "is_test"]]

    symbols = pd.concat([nasdaq, other], ignore_index=True)
    symbols = symbols[symbols["is_test"].eq("N")]
    if not include_etfs:
        symbols = symbols[symbols["is_etf"].eq("N")]

    symbols["symbol"] = symbols["symbol"].astype(str).str.replace(".", "-", regex=False)
    symbols = symbols.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    return symbols


def _read_pipe_file(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = [line for line in response.text.splitlines() if line and not line.startswith("File Creation Time")]
    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def scan_market(config: ScanConfig) -> pd.DataFrame:
    if config.universe == "largest_market_cap":
        symbols_df = load_largest_market_cap_symbols(config.max_symbols or 200)
    else:
        symbols_df = load_us_symbols(config.include_etfs)

    if config.max_symbols:
        symbols_df = symbols_df.head(config.max_symbols)

    frames: list[pd.DataFrame] = []
    symbols = symbols_df["symbol"].tolist()
    name_map = dict(zip(symbols_df["symbol"], symbols_df["name"]))

    for chunk in _chunks(symbols, config.chunk_size):
        prices = yf.download(
            tickers=" ".join(chunk),
            period=config.period,
            interval=config.interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        frames.extend(_score_downloaded_prices(prices, chunk, name_map, config))

    results = pd.DataFrame(frames)
    if results.empty:
        return results

    results = results[
        (results["price"] >= config.min_price)
        & (results["avg_volume_20d"] >= config.min_avg_volume)
    ]

    if config.add_fundamentals and not results.empty:
        results = add_fundamental_growth(results)
        results["score"] = results.apply(_recompute_score_with_fundamentals, axis=1)

    return results.sort_values(["score", "macd_cross_setup", "rsi"], ascending=[False, False, True]).reset_index(drop=True)


def _score_downloaded_prices(
    prices: pd.DataFrame,
    symbols: list[str],
    name_map: dict[str, str],
    config: ScanConfig,
) -> list[dict]:
    rows: list[dict] = []
    for symbol in symbols:
        try:
            history = _extract_symbol_frame(prices, symbol, len(symbols))
            scored = score_symbol(history)
        except Exception:
            continue

        if scored is None:
            continue

        scored["symbol"] = symbol
        scored["name"] = name_map.get(symbol, "")
        rows.append(scored)
    return rows


def _extract_symbol_frame(prices: pd.DataFrame, symbol: str, symbol_count: int) -> pd.DataFrame:
    if symbol_count == 1 and not isinstance(prices.columns, pd.MultiIndex):
        frame = prices.copy()
    elif isinstance(prices.columns, pd.MultiIndex) and symbol in prices.columns.get_level_values(0):
        frame = prices[symbol].copy()
    else:
        return pd.DataFrame()

    frame = frame.rename(columns=str.title)
    needed = {"Close", "High", "Low", "Volume"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    return frame.dropna(subset=["Close"])


def score_symbol(history: pd.DataFrame) -> dict | None:
    if history.empty or len(history) < 120:
        return None

    df = add_indicators(history)
    df["avg_volume_20d"] = df["Volume"].rolling(20).mean()

    indicator_rows = df.dropna(subset=["Close", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "avg_volume_20d"])
    if len(indicator_rows) < 2:
        return None

    latest = indicator_rows.iloc[-1]
    prev = indicator_rows.iloc[-2]

    bb_width = max(latest["bb_upper"] - latest["bb_lower"], 0.01)
    bb_position = float((latest["Close"] - latest["bb_lower"]) / bb_width)
    macd_gap = float(latest["macd"] - latest["macd_signal"])
    prev_gap = float(prev["macd"] - prev["macd_signal"])
    macd_cross_setup = macd_gap < 0 and macd_gap > prev_gap
    macd_gap_pct = abs(macd_gap) / max(float(latest["Close"]), 0.01)
    bounce_rate, bounce_count = lower_band_bounce_rate(df)

    rsi_score = np.interp(float(latest["rsi"]), [20, 40, 55], [30, 25, 0])
    bb_score = np.interp(bb_position, [-0.05, 0.25, 0.60, 1.0], [25, 25, 10, 0])
    macd_score = 25 if macd_cross_setup and macd_gap_pct < 0.015 else np.interp(macd_gap, [-2.0, 0.0, 2.0], [5, 15, 18])
    bounce_score = min(15, bounce_rate * 15)

    score = float(np.clip(rsi_score + bb_score + macd_score + bounce_score, 0, 85))

    return {
        "score": round(score, 1),
        "price": round(float(latest["Close"]), 2),
        "rsi": round(float(latest["rsi"]), 1),
        "bb_position": round(bb_position, 3),
        "near_lower_band": bool(bb_position <= 0.25),
        "macd": round(float(latest["macd"]), 4),
        "macd_signal": round(float(latest["macd_signal"]), 4),
        "macd_gap": round(macd_gap, 4),
        "macd_cross_setup": bool(macd_cross_setup),
        "bounce_rate": round(float(bounce_rate), 2),
        "bounce_count": int(bounce_count),
        "avg_volume_20d": int(latest["avg_volume_20d"]),
        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def add_indicators(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df["rsi"] = rsi(df["Close"])
    macd_line, signal_line, hist = macd(df["Close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(df["Close"])
    return df


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.mask((avg_loss == 0) & (avg_gain > 0), 100)
    value = value.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return value


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal, macd_line - signal


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid, mid + num_std * std, mid - num_std * std


def lower_band_bounce_rate(df: pd.DataFrame, lookahead_days: int = 20, min_return: float = 0.08) -> tuple[float, int]:
    sample = df.dropna(subset=["Close", "bb_lower"]).tail(420).copy()
    touches = sample[sample["Close"] <= sample["bb_lower"] * 1.02]
    successes = 0
    attempts = 0

    for idx in touches.index:
        loc = sample.index.get_loc(idx)
        future = sample.iloc[loc + 1 : loc + 1 + lookahead_days]
        if future.empty:
            continue
        attempts += 1
        start = float(sample.loc[idx, "Close"])
        if float(future["Close"].max()) >= start * (1 + min_return):
            successes += 1

    if attempts == 0:
        return 0.0, 0
    return successes / attempts, attempts


def add_fundamental_growth(results: pd.DataFrame, max_symbols: int = 75) -> pd.DataFrame:
    enriched = results.copy()
    enriched["revenue_growth"] = np.nan
    enriched["earnings_growth"] = np.nan

    for symbol in enriched.head(max_symbols)["symbol"]:
        try:
            ticker = yf.Ticker(symbol)
            financials = ticker.financials
            if financials.empty:
                continue
            if "Total Revenue" in financials.index and financials.shape[1] >= 2:
                current, previous = financials.loc["Total Revenue"].iloc[:2]
                enriched.loc[enriched["symbol"].eq(symbol), "revenue_growth"] = _growth(current, previous)
            if "Net Income" in financials.index and financials.shape[1] >= 2:
                current, previous = financials.loc["Net Income"].iloc[:2]
                enriched.loc[enriched["symbol"].eq(symbol), "earnings_growth"] = _growth(current, previous)
        except Exception:
            continue

    return enriched


def _growth(current: float, previous: float) -> float:
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return float((current - previous) / abs(previous))


def _recompute_score_with_fundamentals(row: pd.Series) -> float:
    base = min(float(row["score"]), 85.0)
    growth_bonus = 0.0
    for field in ("revenue_growth", "earnings_growth"):
        value = row.get(field)
        if pd.notna(value):
            growth_bonus += float(np.interp(value, [-0.10, 0.0, 0.20, 0.50], [-5, 0, 5, 7.5]))
    return round(float(np.clip(base + growth_bonus, 0, 100)), 1)


def load_watchlist() -> list[str]:
    if not WATCHLIST_PATH.exists():
        return []
    try:
        return sorted(set(json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))))
    except Exception:
        return []


def save_watchlist(symbols: Iterable[str]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned = sorted({symbol.upper().strip() for symbol in symbols if symbol and symbol.strip()})
    WATCHLIST_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
