import pandas as pd
import streamlit as st
import yfinance as yf
import exchange_calendars as xc
from research import NY, outcomes, estimate
from optionsdata import database, capture, expirations, save

st.set_page_config(page_title="ES Session Probabilities", layout="wide")
st.title("ES Session Probabilities")
st.caption("Prior-session SPX options volume · ES futures · New York time")

try:
    api_key = st.secrets.get("OPTIONSDATA_API_KEY")
except FileNotFoundError:
    api_key = None
if not api_key:
    import os
    api_key = os.getenv("OPTIONSDATA_API_KEY")

with st.sidebar:
    st.header("OptionsData connection")
    if not api_key:
        st.info("Set OPTIONSDATA_API_KEY in the app Secrets to enable SPX options.")
    if st.button("Check SPX access", disabled=not api_key):
        try:
            available = expirations(api_key)
            st.write(f"Accessible expirations: {len(available)}")
            st.write(available[:8])
        except Exception as exc:
            st.error(f"SPX access check failed: {exc}")
    now = pd.Timestamp.now(tz=NY)
    session_day = now.tz_localize(None).normalize()
    calendar = xc.get_calendar("XNYS")
    can_capture = (calendar.is_session(session_day)
                   and calendar.session_close(session_day).tz_convert(NY).hour == 16
                   and 976 <= now.hour * 60 + now.minute < 1020)
    if st.button("Save today's volume", disabled=not api_key or not can_capture):
        try:
            save(capture(api_key, session_day))
            st.success("Today's snapshot is stored.")
        except Exception as exc:
            st.error(f"Capture failed: {exc}")
    st.caption("Capture window: 16:16–16:59 New York time on regular sessions.")

@st.cache_data(ttl=1800)
def load_bars():
    bars = yf.Ticker("ES=F").history(period="59d", interval="30m", prepost=True, auto_adjust=False)
    if bars.empty:
        raise ValueError("No ES bars returned")
    bars.index=bars.index.tz_convert(NY)
    return bars.loc[bars.index + pd.Timedelta(minutes=30) <= pd.Timestamp.now(tz=NY)]

try:
    bars=load_bars()
except Exception as exc:
    st.error(f"ES history unavailable: {exc}")
    st.stop()
cal=xc.get_calendar("XNYS")
today=pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
days=cal.sessions_in_range(bars.index.min().date(),today+pd.Timedelta(days=8))
with database() as db:
    db.execute("CREATE TABLE IF NOT EXISTS labels(day TEXT, session TEXT, change REAL, PRIMARY KEY(day,session))")
    for previous, day in zip(days,days[1:]):
        if day > today:
            break
        for session,change in outcomes(bars,day,previous).items():
            if pd.notna(change):
                db.execute("INSERT OR REPLACE INTO labels VALUES(?,?,?)",(str(day.date()),session,float(change)))
    labels=pd.read_sql("SELECT * FROM labels ORDER BY day",db)
    signals=pd.read_sql("SELECT * FROM signals ORDER BY day",db)
st.caption(f"Latest ES bar starts at {bars.index.max():%Y-%m-%d %H:%M %Z}. Saved volume snapshots: {len(signals)}.")
origin=st.selectbox("Forecast from US session close",list(reversed([d for d in days if cal.session_close(d)<=pd.Timestamp.now(tz="UTC")])))
target=cal.next_session(origin)
st.write(f"Target trading date: **{target:%Y-%m-%d}**")
st.caption("Day: previous 16:00 to 16:00. Asia: prior calendar evening 19:00 to 02:00. New York: 09:30 to 16:00. Flat outcomes excluded; probabilities are conditional on a move.")
signal=signals.set_index("day")
for name in ("Day","Asia","New York"):
    h=[]
    scores=[]
    previous_labels=labels[(labels.session==name)&(labels.day<=str(origin.date()))&(labels.change!=0)]
    for row in previous_labels.itertuples():
        prior=str(cal.previous_session(pd.Timestamp(row.day)).date())
        if prior not in signal.index:
            continue
        v=signal.loc[prior]
        share=v.puts/(v.calls+v.puts)
        up=int(row.change>0)
        if len(h)>=30:
            p,base,_=estimate(h,share)
            scores.append(((p-up)**2,(base-up)**2))
        h.append((share,up))
    st.subheader(name)
    if len(previous_labels):
        st.write(f"Historical up frequency: {(previous_labels.change>0).mean():.1%} across {len(previous_labels)} moving sessions.")
    if str(origin.date()) in signal.index and len(h)>=30:
        v=signal.loc[str(origin.date())]
        p,_,n=estimate(h,v.puts/(v.calls+v.puts))
        st.write(f"Estimated up {p:.1%} · down {1-p:.1%} · {n} comparable observations out of {len(h)}.")
    else:
        st.info(f"Options forecast pending. Requires an origin snapshot and 30 paired moving sessions. Available pairs: {len(h)}.")
    if scores:
        st.write("Walk-forward mean Brier loss (lower is better)",pd.DataFrame(scores,columns=["Volume model","Base rate"]).mean().to_dict(),f"on {len(scores)} forecasts")
