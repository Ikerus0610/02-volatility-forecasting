"""Streamlit entry point. Run: streamlit run app.py"""
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
import streamlit as st
from volatility_lab.data import DataError, load_csv, annualized_vol, future_dates
from volatility_lab.demo import DEMO_PATH
from volatility_lab.evaluation import Config, evaluate, metric_table, common_rows, selected_rows, export_results, losses
from volatility_lab.models import MODEL_NAMES
from volatility_lab.charts import forecast_chart, error_chart, scatter_chart

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Volatility Forecasting Lab", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {max-width:1350px;padding-top:2.2rem;padding-bottom:3rem}
h1 {font-size:2.6rem!important;letter-spacing:-.055em;font-weight:700!important}
h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {border:1px solid #e2e8ee;border-radius:10px;padding:16px;background:#f8fafb}
[data-testid="stMetricLabel"] {color:#59697c}
[data-testid="stSidebar"] {border-right:1px solid #e2e8ee}
.eyebrow {font-size:11px;letter-spacing:.17em;color:#087f83;font-weight:700;margin-bottom:8px}
.subtitle {color:#617084;max-width:860px;font-size:16px}
</style>""", unsafe_allow_html=True)


@st.cache_data(max_entries=4, ttl=3600, show_spinner=False)
def cached_run(raw, source, asset, window, lam, start, end):
    data = load_csv(raw, source)
    config = Config(window, lam)
    results = evaluate(data.returns[asset], config, start, end)
    latest = data.prices.index[-1]
    overview = results.loc[[latest]] if latest in results.index else evaluate(data.returns[asset], config, latest, latest)
    return results, overview


def plot(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def show_metrics(frame):
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "MAE_pp": st.column_config.NumberColumn("MAE (vol pp)", format="%.3f"),
        "RMSE_pp": st.column_config.NumberColumn("RMSE (vol pp)", format="%.3f"),
        "QLIKE": st.column_config.NumberColumn("QLIKE", format="%.4f")})


def inspector(origin, row, data, asset, config):
    st.markdown(f"**Origin: {origin:%Y-%m-%d}, after the US market close**")
    st.caption("Model inputs end at this close. The future panel is shown only for retrospective verification.")
    train = data.returns[asset].loc[:origin].iloc[-config.window:]
    a, b, c = st.columns(3)
    a.metric("Training window", f"{int(train.notna().sum())} / {config.window}")
    b.metric("Training starts", train.index[0].strftime("%Y-%m-%d"))
    c.metric("Training ends", origin.strftime("%Y-%m-%d"))
    forecasts = []
    for model in MODEL_NAMES:
        forecasts.append({"Model": model, "Mean daily variance": row[model],
                          "Annualized vol (%)": float(annualized_vol(row[model])*100),
                          "Status": row[f"{model} status"]})
    st.dataframe(pd.DataFrame(forecasts), hide_index=True, width="stretch")
    future = future_dates(origin)
    forward = pd.DataFrame(index=future)
    forward.index.name = "session"
    forward["Adjusted close"] = data.prices[asset].reindex(future)
    forward["Log return"] = data.returns[asset].reindex(future)
    forward["Squared log return"] = forward["Log return"]**2
    for model in MODEL_NAMES:
        forward[f"{model} daily variance"] = [row[f"{model} h{h}"] for h in range(1, 6)]
    st.markdown("**The five following sessions**")
    st.dataframe(forward, width="stretch")
    observed = row.realized_variance
    if pd.notna(observed):
        st.success(f"Observed mean squared return: {observed:.10f} · annualized volatility: {annualized_vol(observed)*100:.2f}%")
    else:
        st.info("Future outcome unavailable: " + row.outcome_status.replace("_", " ") + ". It is excluded from scoring.")
    with st.expander("Inspect every training observation"):
        table = pd.DataFrame({"Adjusted close": data.prices[asset].reindex(train.index), "Log return": train,
                              "Squared log return": train**2})
        table["Historical 20 input"] = table.index.isin(train.index[-20:])
        table["EWMA initialization"] = np.arange(len(table)) < 20
        previous_position = data.prices.index.get_loc(train.index[0]) - 1
        if previous_position >= 0:
            previous_date = data.prices.index[previous_position]
            st.caption(f"Price anchoring the first training return: {previous_date:%Y-%m-%d} · {data.prices.at[previous_date, asset]:.6f}.")
        st.dataframe(table, width="stretch", height=320)
        st.download_button("Download this training window", table.to_csv().encode(), f"{asset}_{origin:%Y%m%d}_training.csv")
    with st.expander("GARCH fit details and messages"):
        keys = ["omega_daily", "alpha", "beta", "persistence", "convergence_flag", "iterations"]
        st.dataframe(pd.DataFrame({"Parameter": keys, "Value": [row.get(k, np.nan) for k in keys]}), hide_index=True)
        for model in MODEL_NAMES:
            st.write(f"**{model}:** {row[f'{model} status']} — {row[f'{model} message'] or 'No numerical warning.'}")


st.sidebar.markdown("### ◈ Volatility Lab")
st.sidebar.caption("RESEARCH PORTFOLIO / 02")
source_choice = st.sidebar.radio("Data source", ["Synthetic demo", "Upload CSV"])
demo_raw = DEMO_PATH.read_bytes()
if source_choice == "Synthetic demo":
    raw, source = demo_raw, "SYNTHETIC DEMO · seed 20261117"
else:
    uploaded = st.sidebar.file_uploader("Adjusted daily closes", type="csv")
    st.sidebar.download_button("Download CSV template", (ROOT/"data/template.csv").read_bytes(), "template.csv")
    if uploaded is None:
        st.title("Volatility Forecasting Lab")
        st.info("Upload adjusted daily closes with date,SPY and optional QQQ columns. The template illustrates the format; the demo provides a complete runnable dataset.")
        st.code("date,SPY,QQQ\n2024-01-02,470.00,400.00\n2024-01-03,468.50,397.50", language="text")
        st.stop()
    raw = uploaded.getvalue()
    source = "SYNTHETIC DEMO · uploaded copy" if hashlib.sha256(raw).digest() == hashlib.sha256(demo_raw).digest() else "USER CSV · adjusted closes declared by uploader"
try:
    data = load_csv(raw, source)
except (DataError, ValueError) as exc:
    st.title("Data validation")
    st.error(str(exc))
    st.stop()
synthetic = source.startswith("SYNTHETIC")
identity = (source_choice, data.fingerprint)
if st.session_state.get("identity") != identity:
    for key in ["run", "asset", "period", "inspect_origin", "error_origin"]:
        st.session_state.pop(key, None)
    st.session_state.identity = identity
asset = st.sidebar.selectbox("Instrument", data.prices.columns, key="asset")
idx = data.prices.index
default_start, default_end = idx[max(20, len(idx)-257)], idx[max(20, len(idx)-6)]
with st.sidebar.form("configuration"):
    window = st.number_input("Training window (returns)", 60, 2520, 756, step=21)
    lam = st.number_input("EWMA lambda", .01, .999, .94, step=.01, format="%.3f")
    period = st.date_input("Evaluation origins", (default_start.date(), default_end.date()),
                           min_value=idx[0].date(), max_value=idx[-1].date(), key="period")
    run_clicked = st.form_submit_button("Run evaluation", type="primary", width="stretch")
st.sidebar.caption("GARCH is refitted at every origin. Cached runs reuse identical data and settings for one hour.")
st.sidebar.divider()
view = st.sidebar.radio("Research views", ["Forecast overview", "Historical evaluation", "Model diagnostics", "Data & methodology"])
st.markdown('<div class="eyebrow">QUANTITATIVE RESEARCH / PROJECT 02</div>', unsafe_allow_html=True)
st.title("Volatility Forecasting Lab")
st.markdown('<div class="subtitle">Can a model forecast next week’s realized volatility better than recent history?</div>', unsafe_allow_html=True)
st.caption(f"{asset}  /  5 trading sessions  /  {source}")
if synthetic:
    st.info("SYNTHETIC DATA — artificial prices with changing volatility regimes. Results demonstrate the workflow, not market performance.", icon="🧪")
else:
    st.caption("Input must contain adjusted, completed daily closes. Adjustments and provider revisions cannot be verified from the CSV alone.")
config = Config(int(window), float(lam))
settings = (data.fingerprint, asset, config.window, config.lam, tuple(period))
if run_clicked:
    if len(period) != 2:
        st.error("Choose both the first and last evaluation origin.")
    else:
        started = time.perf_counter()
        with st.spinner("Running chronological forecasts and fitting GARCH at each origin…"):
            results, latest = cached_run(raw, source, asset, config.window, config.lam, str(period[0]), str(period[1]))
        st.session_state.run = (settings, results, latest, time.perf_counter()-started)

if view == "Data & methodology":
    st.subheader("Data quality & provenance")
    st.dataframe(data.audit, hide_index=True, width="stretch")
    for warning in data.warnings:
        st.warning(warning)
    longest = int(data.audit.set_index("asset").at[asset, "longest_return_run"])
    if longest < config.window + 5:
        st.warning(f"Longest valid return run: {longest}. A full training window plus one observed target needs {config.window+5}; common-date results may be empty.")
    missing = data.prices.loc[data.prices.isna().any(axis=1)]
    if len(missing):
        with st.expander("Missing prices by session"):
            st.dataframe(missing, width="stretch")
    st.caption(f"Data SHA-256: {data.fingerprint}")
    st.download_button("Download CSV template", (ROOT/"data/template.csv").read_bytes(), "template.csv", key="method_template")
    st.download_button("Download synthetic dataset", demo_raw, "synthetic_prices.csv")
    st.markdown((ROOT/"docs/methodology.md").read_text())
    with st.expander("Real-data source review"):
        st.markdown((ROOT/"docs/data_sources.md").read_text())

run = st.session_state.get("run")
if run is None or run[0] != settings:
    if view != "Data & methodology":
        st.subheader("Start a reproducible evaluation")
        st.write("Choose an instrument and training window, then select **Run evaluation**. The default evaluates 252 historical forecast origins and also estimates volatility at the latest close.")
        if run is not None:
            st.warning("Inputs changed. Run evaluation to refresh results for these settings.")
    st.stop()
_, results, latest, runtime = run
label = f"{asset} · {'SYNTHETIC' if synthetic else 'USER CSV'}"
st.caption(f"Run: {config.window} trailing returns · EWMA λ {config.lam:.3f} · {len(results)} requested origins · completed in {runtime:.1f}s (cached reruns may be faster)")

if view == "Forecast overview":
    st.subheader("Latest available forecasts")
    pool = pd.concat([results, latest]).sort_index()
    pool = pool.loc[~pool.index.duplicated(keep="last")]
    latest_origin = latest.index[-1]
    st.caption(f"Latest input close: {latest_origin:%Y-%m-%d}. Forecasts estimate fluctuation magnitude; return direction is a separate question.")
    cols = st.columns(3)
    for col, model in zip(cols, MODEL_NAMES):
        available = pool.loc[pool[model].notna()]
        with col:
            if available.empty:
                st.metric(model, "Unavailable")
                st.caption(latest.iloc[0][f"{model} message"])
            else:
                origin, row = available.index[-1], available.iloc[-1]
                st.metric(model, f"{annualized_vol(row[model])*100:.2f}%")
                st.caption(f"Annualized volatility · origin {origin:%Y-%m-%d}")
                st.caption(f"Horizon: {row.target_start:%Y-%m-%d} → {row.target_end:%Y-%m-%d}")
                if origin != latest_origin:
                    st.warning("Stale estimate: latest fit unavailable. Showing the last valid forecast in this run.")
    st.caption("These are annualized volatility estimates for the average daily variance over five sessions, not five-day returns or trading signals.")
    rows = common_rows(results)
    if len(rows):
        plot(forecast_chart(rows.tail(126), label))
    with st.expander("Inspect the latest origin"):
        inspector(latest_origin, latest.iloc[-1], data, asset, config)

elif view in ["Historical evaluation", "Model diagnostics"]:
    non_overlap = st.radio("Evaluation sampling", ["Daily (overlapping)", "Every 5 sessions (non-overlapping)"], horizontal=True, key="sampling") != "Daily (overlapping)"
    subset = selected_rows(results, non_overlap)
    shared = common_rows(results, non_overlap)
    observed = subset.realized_variance.notna().sum()
    a, b, c, d = st.columns(4)
    a.metric("Selected origins", len(subset))
    b.metric("Common evaluated dates", len(shared))
    c.metric("Observed, model excluded", int(observed-len(shared)))
    d.metric("Outcome unavailable", int(len(subset)-observed))
    st.caption(f"Non-overlapping grid is anchored to {results.index[0]:%Y-%m-%d}, before model/outcome exclusions. Daily targets overlap; even non-overlapping targets may be serially dependent. No significance claims are made.")
    if view == "Historical evaluation":
        st.subheader("Out-of-sample comparison")
        show_metrics(metric_table(results, non_overlap))
        st.caption("Lower is better. MAE/RMSE are annualized-volatility percentage points. QLIKE uses daily variances. The separate baseline row retains every available baseline date within this interval and sampling grid.")
        if len(shared):
            plot(forecast_chart(shared, label))
        else:
            st.warning("No common scored dates. Inspect data length, gaps, selected interval and GARCH status.")
        st.subheader("Inspect a forecast")
        options = subset.index.tolist()
        if st.session_state.get("inspect_origin") not in options:
            st.session_state.inspect_origin = options[-1]
        origin = st.selectbox("Forecast origin", options, format_func=lambda x: x.strftime("%Y-%m-%d"), key="inspect_origin")
        inspector(origin, results.loc[origin], data, asset, config)
    else:
        st.subheader("Error and convergence diagnostics")
        statuses = subset["GARCH(1,1) status"].value_counts().rename_axis("GARCH status").reset_index(name="Origins")
        st.dataframe(statuses, hide_index=True, width="stretch")
        st.caption("Status 'ok' requires convergence flag 0 and finite positive variance forecasts. Insufficient history and degenerate returns are distinct from optimizer failures.")
        failures = subset.loc[subset["GARCH(1,1) status"] != "ok"]
        if len(failures):
            st.dataframe(failures[["GARCH(1,1) status", "GARCH(1,1) message", "outcome_status"]], width="stretch")
        if len(shared):
            plot(error_chart(shared, label))
            plot(scatter_chart(shared, label))
            chosen_model = st.selectbox("Model for largest-error inspection", MODEL_NAMES)
            error, qlike = losses(shared.realized_variance, shared[chosen_model])
            worst = pd.DataFrame({"Absolute error (vol pp)": np.abs(error), "Signed error (vol pp)": error,
                                  "QLIKE": qlike}, index=shared.index).sort_values("Absolute error (vol pp)", ascending=False).head(10)
            st.markdown("**Largest errors in the selected sample**")
            st.dataframe(worst, width="stretch")
            error_options = worst.index.tolist()
            if st.session_state.get("error_origin") not in error_options:
                st.session_state.error_origin = error_options[0]
            origin = st.selectbox("Inspect a large-error origin", error_options, format_func=lambda x: x.strftime("%Y-%m-%d"), key="error_origin")
            with st.expander("Training inputs and subsequent outcome", expanded=False):
                inspector(origin, results.loc[origin], data, asset, config)
        st.markdown("**GARCH parameter history**")
        params = subset.reindex(columns=["omega_daily", "alpha", "beta", "persistence", "convergence_flag", "iterations"])
        st.dataframe(params, width="stretch", height=260)
        if params.persistence.ge(.999).any():
            st.warning("Some fitted persistence values are near one. Long-run variance interpretations can be unstable; the application only uses the finite five-session forecast.")

elif view == "Data & methodology":
    st.subheader("Run configuration and audit")
    st.json({"asset": asset, "training_returns": config.window, "lambda": config.lam,
             "evaluation_first_origin": str(results.index[0].date()), "evaluation_last_origin": str(results.index[-1].date()),
             "horizon_sessions": 5, "annualization": 252, "garch_refit": "every origin", "garch_return_scale": 100,
             "calendar": "XNYS", "data_source": source, "data_sha256": data.fingerprint})
    st.dataframe(results[["training_count", "outcome_status", "Historical 20 status", "EWMA status", "GARCH(1,1) status", "common_valid", "non_overlapping"]], width="stretch")

st.divider()
st.download_button("Download per-origin results (CSV)", export_results(results, data, asset, config),
                   f"{'synthetic' if synthetic else 'user_csv'}_{asset}_forecasts.csv", mime="text/csv")
st.caption("Export includes every requested origin, unscored outcomes, model statuses, five daily forecasts, GARCH parameters, units and data fingerprint. Latest overview origin is separate if outside the evaluation interval.")
st.download_button("Download latest-origin forecast (CSV)", export_results(latest, data, asset, config),
                   f"{'synthetic' if synthetic else 'user_csv'}_{asset}_latest.csv", mime="text/csv")
