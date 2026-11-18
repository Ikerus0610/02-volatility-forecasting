"""Fetch completed US equity sessions and retain Yahoo's dividend/split adjustments."""
import exchange_calendars as xcals
import pandas as pd
import streamlit as st
import yfinance as yf


def latest_session():
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    cal = xcals.get_calendar("XNYS", start=today - pd.Timedelta(days=14),
                             end=today + pd.Timedelta(days=2))
    return cal.schedule.loc[cal.schedule["close"] <= now].index[-1].tz_localize(None)


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def download_prices(tickers, through):
    frame = yf.download(
        list(tickers), start=(through - pd.DateOffset(years=12)).date(),
        end=(through + pd.Timedelta(days=1)).date(), interval="1d",
        auto_adjust=False, group_by="column", ignore_tz=True,
        progress=False, threads=False, timeout=15,
    )
    if frame is None or frame.empty or "Adj Close" not in frame:
        raise ValueError("Yahoo Finance returned no adjusted closing prices.")
    prices = frame["Adj Close"].reindex(columns=tickers).dropna(how="all").sort_index()
    missing = [t for t in tickers if prices[t].count() < 22]
    if missing:
        raise ValueError("Insufficient adjusted price history for " + ", ".join(missing))
    prices.index.name = "date"
    return prices.to_csv(date_format="%Y-%m-%d").encode()


def market_prices(tickers):
    through = latest_session()
    try:
        with st.spinner("Loading adjusted closes from Yahoo Finance…"):
            raw = download_prices(tuple(tickers), through)
    except Exception as exc:
        st.error(f"Yahoo Finance data unavailable. {exc}")
        if st.button("Retry Yahoo Finance"):
            st.rerun()
        st.stop()
    return raw, through
