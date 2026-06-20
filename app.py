from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_screener import ScanConfig, add_indicators, load_watchlist, save_watchlist, scan_market


st.set_page_config(page_title="LEAPS Stock Screener", page_icon=":chart_with_upwards_trend:", layout="wide")


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_scan(config_dict: dict) -> pd.DataFrame:
    return scan_market(ScanConfig(**config_dict))


def main() -> None:
    st.title("LEAPS Stock Screener")
    st.caption("Find oversold US stocks with improving momentum for long-call LEAPS research.")

    with st.sidebar:
        st.header("Scan")
        universe_label = st.selectbox("Universe", ["Top market cap", "All US listings"])
        max_default = 200 if universe_label == "Top market cap" else 500
        max_limit = 250 if universe_label == "Top market cap" else 10_000
        max_symbols = st.number_input("Universe size", min_value=25, max_value=max_limit, value=max_default, step=25)
        min_price = st.number_input("Minimum stock price", min_value=0.0, value=5.0, step=1.0)
        min_avg_volume = st.number_input("Minimum 20-day avg volume", min_value=0, value=300_000, step=50_000)
        include_etfs = st.toggle("Include ETFs", value=False, disabled=universe_label == "Top market cap")
        add_fundamentals = st.toggle("Add fundamentals scoring", value=False, help="Slower. Uses available Yahoo Finance financial statements.")
        run_scan = st.button("Run scan", type="primary", use_container_width=True)

        st.divider()
        min_score = st.slider("Minimum score shown", 0, 100, 55)
        watchlist = load_watchlist()
        manual_symbol = st.text_input("Add symbol to watchlist", placeholder="AAPL").upper().strip()
        if st.button("Add to watchlist", use_container_width=True) and manual_symbol:
            save_watchlist([*watchlist, manual_symbol])
            st.rerun()

    config = ScanConfig(
        max_symbols=int(max_symbols),
        min_price=float(min_price),
        min_avg_volume=int(min_avg_volume),
        include_etfs=include_etfs and universe_label == "All US listings",
        universe="largest_market_cap" if universe_label == "Top market cap" else "all_us",
        add_fundamentals=add_fundamentals,
    )

    if run_scan or "scan_results" not in st.session_state:
        with st.spinner("Scanning free market data. Large universes can take a few minutes."):
            st.session_state.scan_results = cached_scan(config.__dict__)

    results = st.session_state.get("scan_results", pd.DataFrame())
    if results.empty:
        st.info("No candidates found yet. Try lowering the score, price, or volume filters.")
        return

    filtered = results[results["score"] >= min_score].copy()

    top_score = filtered["score"].max() if not filtered.empty else 0
    cross_count = int(filtered["macd_cross_setup"].sum()) if not filtered.empty else 0
    lower_band_count = int(filtered["near_lower_band"].sum()) if not filtered.empty else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Candidates", f"{len(filtered):,}")
    metric_cols[1].metric("Top score", f"{top_score:.1f}")
    metric_cols[2].metric("MACD setups", f"{cross_count:,}")
    metric_cols[3].metric("Near lower band", f"{lower_band_count:,}")

    tab_candidates, tab_watchlist, tab_chart, tab_method = st.tabs(["Candidates", "Watchlist", "Chart", "Scoring"])

    with tab_candidates:
        show = filtered[
            [
                "symbol",
                "name",
                "score",
                "price",
                "rsi",
                "bb_position",
                "macd_gap",
                "macd_cross_setup",
                "bounce_rate",
                "bounce_count",
                "avg_volume_20d",
            ]
        ].copy()
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                "bb_position": st.column_config.NumberColumn("BB position", format="%.3f"),
                "macd_cross_setup": st.column_config.CheckboxColumn("MACD setup"),
                "bounce_rate": st.column_config.NumberColumn("Bounce rate", format="%.0%%"),
                "avg_volume_20d": st.column_config.NumberColumn("20D avg volume", format="%d"),
            },
        )

        selected = st.multiselect("Add candidates to watchlist", options=filtered["symbol"].tolist())
        if st.button("Save selected", disabled=not selected):
            save_watchlist([*load_watchlist(), *selected])
            st.success("Watchlist updated.")

    with tab_watchlist:
        watchlist = load_watchlist()
        if not watchlist:
            st.info("Your watchlist is empty.")
        else:
            watch_df = results[results["symbol"].isin(watchlist)].copy()
            missing = sorted(set(watchlist) - set(watch_df["symbol"]))
            st.dataframe(watch_df, use_container_width=True, hide_index=True)
            if missing:
                st.caption(f"Not in latest scan results: {', '.join(missing)}")
            remove = st.multiselect("Remove symbols", options=watchlist)
            if st.button("Remove selected", disabled=not remove):
                save_watchlist([symbol for symbol in watchlist if symbol not in remove])
                st.rerun()

    with tab_chart:
        default_symbol = filtered.iloc[0]["symbol"] if not filtered.empty else results.iloc[0]["symbol"]
        symbol = st.selectbox("Symbol", options=results["symbol"].tolist(), index=results["symbol"].tolist().index(default_symbol))
        render_symbol_chart(symbol)

    with tab_method:
        st.markdown(
            """
            The score is a 0-100 research ranking for long-call LEAPS candidates.

            Core technical score, capped at 85 before fundamentals:

            - RSI contributes most when below 40.
            - Bollinger score rises when price is near or below the lower band.
            - MACD score rises when the MACD line is below the signal line but closing the gap.
            - Bounce history rewards stocks that previously rallied after lower-band touches.

            Optional fundamentals can add or subtract points based on available revenue and earnings growth.
            """
        )


def render_symbol_chart(symbol: str) -> None:
    history = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    if history.empty:
        st.warning("No chart data found.")
        return

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            labels = history.columns.get_level_values(level)
            if "Close" in labels:
                history.columns = labels
                break

    history = history.rename(columns=str.title)
    if "Close" not in history.columns:
        st.warning("Chart data did not include a close price.")
        return

    chart_data = add_indicators(history).dropna().tail(252)
    if chart_data.empty:
        st.warning("Not enough history to chart indicators.")
        return

    latest = chart_data.iloc[-1]
    stats = st.columns(4)
    stats[0].metric("Price", f"${latest['Close']:.2f}")
    stats[1].metric("RSI", f"{latest['rsi']:.1f}")
    stats[2].metric("MACD gap", f"{latest['macd'] - latest['macd_signal']:.3f}")
    stats[3].metric("Lower band", f"${latest['bb_lower']:.2f}")

    st.subheader(f"{symbol} price and Bollinger Bands")
    st.line_chart(chart_data[["Close", "bb_upper", "bb_mid", "bb_lower"]], height=360)

    st.subheader("RSI")
    st.line_chart(chart_data[["rsi"]], height=220)
    st.caption("RSI below 40 is part of the buy-candidate score.")

    st.subheader("MACD")
    st.line_chart(chart_data[["macd", "macd_signal", "macd_hist"]], height=260)


if __name__ == "__main__":
    main()
