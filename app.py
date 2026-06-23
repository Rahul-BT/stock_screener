from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

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
        add_fundamentals = st.toggle("Add fundamentals scoring", value=True, help="Slower. Uses available Yahoo Finance financial statements.")
        run_scan = st.button("Run scan", type="primary", use_container_width=True)

        st.divider()
        with st.expander("Scoring weights (advanced)", expanded=False):
            st.caption("Adjust contribution multipliers for technical components")
            weight_rsi = st.slider("RSI weight", 0.0, 2.0, 1.0, step=0.05)
            weight_bb = st.slider("Bollinger weight", 0.0, 2.0, 1.0, step=0.05)
            weight_macd = st.slider("MACD weight", 0.0, 2.0, 1.0, step=0.05)
            weight_bounce = st.slider("Bounce history weight", 0.0, 2.0, 1.0, step=0.05)
            cap_pre_fundamentals = st.slider("Technical cap (pre-fundamentals)", 0, 100, 85)
            fundamentals_weight = st.slider("Fundamentals weight", 0.0, 2.0, 1.0, step=0.05)

        min_score = st.slider("Minimum score shown", 0, 100, 70)
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
        weight_rsi=float(locals().get('weight_rsi', 1.0)),
        weight_bb=float(locals().get('weight_bb', 1.0)),
        weight_macd=float(locals().get('weight_macd', 1.0)),
        weight_bounce=float(locals().get('weight_bounce', 1.0)),
        cap_pre_fundamentals=int(locals().get('cap_pre_fundamentals', 85)),
        fundamentals_weight=float(locals().get('fundamentals_weight', 1.0)),
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

            show_watchlist = watch_df[
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
                show_watchlist,
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

    # Bollinger bands chart.
    bb_chart = chart_data[["Close", "bb_upper", "bb_mid", "bb_lower"]].copy()

    # Use Plotly so line colors and hover content can be customized precisely.
    fig = go.Figure()
    # Upper band (darker green)
    fig.add_trace(go.Scatter(x=bb_chart.index, y=bb_chart["bb_upper"],
                             mode="lines", name="BB Upper",
                             line=dict(color="#2E8B57", width=1),
                             hovertemplate="BB Upper: $%{y:.2f}<extra></extra>"))
    # Mid band (darker green, dashed) - include mid value in hover
    fig.add_trace(go.Scatter(x=bb_chart.index, y=bb_chart["bb_mid"],
                             mode="lines", name="BB Mid",
                             line=dict(color="#2E8B57", width=1, dash="dash"),
                             hovertemplate="BB Mid: $%{y:.2f}<extra></extra>", showlegend=False))
    # Lower band (darker green)
    fig.add_trace(go.Scatter(x=bb_chart.index, y=bb_chart["bb_lower"],
                             mode="lines", name="BB Lower",
                             line=dict(color="#2E8B57", width=1),
                             hovertemplate="BB Lower: $%{y:.2f}<extra></extra>"))
    # Price line (blue)
    fig.add_trace(go.Scatter(x=bb_chart.index, y=bb_chart["Close"],
                             mode="lines", name="Price",
                             line=dict(color="#1f77b4", width=2),
                             hovertemplate="Price: $%{y:.2f}<extra></extra>"))

    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      height=320,
                      margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(showspikes=True, spikemode="across")

    st.plotly_chart(fig, use_container_width=True)





    st.subheader("RSI")

    # Plot RSI using Plotly so threshold lines can be styled.
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=chart_data.index, y=chart_data["rsi"],
                                 mode="lines", name="RSI",
                                 line=dict(color="#1f77b4", width=2),
                                 hovertemplate="RSI: %{y:.1f}<extra></extra>", showlegend=False))
    # Threshold lines (light red)
    fig_rsi.add_trace(go.Scatter(x=chart_data.index, y=[70.0] * len(chart_data),
                                 mode="lines", name="Upper 70",
                                 line=dict(color="#FF7F7F", width=1, dash="dash"),
                                 hovertemplate="70<extra></extra>", showlegend=False))
    fig_rsi.add_trace(go.Scatter(x=chart_data.index, y=[30.0] * len(chart_data),
                                 mode="lines", name="Lower 30",
                                 line=dict(color="#FF7F7F", width=1, dash="dash"),
                                 hovertemplate="30<extra></extra>", showlegend=False))

    fig_rsi.update_layout(height=320, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.caption("RSI below 40 is part of the buy-candidate score. Thresholds: 30 (oversold) and 70 (overbought).")



    st.subheader("MACD")
    # Plot MACD and signal with custom colors
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=chart_data.index, y=chart_data["macd"],
                                  mode="lines", name="MACD",
                                  line=dict(color="#89CFF0", width=2),
                                  hovertemplate="MACD: %{y:.3f}<extra></extra>"))
    fig_macd.add_trace(go.Scatter(x=chart_data.index, y=chart_data["macd_signal"],
                                  mode="lines", name="Signal",
                                  line=dict(color="#FF7F7F", width=1.5, dash="dash"),
                                  hovertemplate="Signal: %{y:.3f}<extra></extra>"))
    fig_macd.update_layout(height=320, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_macd, use_container_width=True)


if __name__ == "__main__":
    main()
