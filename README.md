# Volatility Forecasting Lab

**Project 02 · public quantitative research portfolio · Markets / Sales & Trading**

Can the next five sessions of SPY realized volatility be forecast better than a trailing 20-session historical reference? This Streamlit lab compares Historical 20, EWMA and zero-mean normal GARCH(1,1), using rolling-origin forecasts and matched evaluation dates. QQQ is a separate selectable experiment.

**All bundled price data and demo results are SYNTHETIC. No real-market performance or trading edge is claimed.**

## Run locally (macOS / Linux / Windows)

Use **Python 3.12**. In the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`. No API keys are needed. `requirements.txt` pins the tested dependency environment; `requirements.in` lists direct dependencies. To run tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See [verification record](docs/verification.md) for the completed checks, synthetic reference results and remaining platform/browser verification limits.

## First complete experiment

1. Leave **Synthetic demo** and **SPY** selected. Default settings: 756 trailing returns, EWMA lambda 0.94, 252 historical origins, five-session horizon.
2. Click **Run evaluation**. GARCH is reestimated for every origin. Expect several seconds to tens of seconds depending on the host and interval length.
3. Open **Historical evaluation**. All three main rows use identical dates. Compare the extra baseline row to see exclusion effects.
4. Switch between daily and every-five-session origins. The five-session grid is anchored before exclusions, not formed by thinning surviving forecasts.
5. Select a forecast origin. Inspect every training return, the prior price anchoring the first return, each daily prediction and the five subsequent outcomes.
6. Open **Model diagnostics** for residual errors, forecast/outcome scatter, worst cases, convergence and GARCH parameters. **Data & methodology** contains provenance, validation and all formulas.
7. Download the per-origin CSV. Pending and excluded rows remain present with reasons. Latest-close results have a separate download if outside the evaluation interval.

The overview displays the most recent valid forecast per model among the selected evaluation origins and the latest dataset origin. Failed latest fits are explicitly marked unavailable or stale; a stale card retains its actual origin/horizon. Do not interpret it as a current forecast.

## Supply real data

Use `data/template.csv` for column format (invented values; too short for evaluation). Required: `date,SPY`; optional: `QQQ`. Dates must be YYYY-MM-DD and prices must be positive adjusted completed closes. The 1,801-row `data/synthetic_prices.csv` also exercises the upload path.

Default settings need **762 consecutive prices** to produce one complete common-date evaluation: 756 training returns plus five future returns. More data are strongly preferable. A gap breaks consecutive training windows; no multi-session return is treated as daily. Blank/NA cells are audited. Duplicates, invalid supplied values and non-trading dates are rejected. The UI permits windows from 60 to 2,520 returns; short windows can be statistically unstable. Data length, historical span and chosen evaluation dates determine sample size.

Alpha Vantage's adjusted daily endpoint and conditions were reviewed. It is documented as premium; authenticated access and public redistribution rights are not established here. Therefore CSV import is the complete real-data path and no vendor data or downloader is included. See [source review](docs/data_sources.md). CSV contents cannot prove that prices are adjusted or were available in exactly this form at the historical origin.

## What is being estimated?

At origin t, target `mean(r[t+1:t+6] ** 2)` uses decimal log returns. Annualized displayed volatility is `100 * sqrt(252 * variance)`, in percent. Historical 20 averages the last 20 squared returns. EWMA starts with the mean of the first 20 squared returns in each trailing training window and recursively updates through t. GARCH fits returns multiplied by 100, forecasts five daily variances, divides each by 10,000 and averages them. The final five dates remain unevaluated.

MAE/RMSE use annualized-volatility **percentage points**. QLIKE uses mean daily variances and a disclosed 1e-12 floor, with floor counts. Full formulas, failure rules, gap conventions, overlap treatment and interpretation limits are in [methodology](docs/methodology.md), also rendered inside the app.

A lower loss on a sample supports only a descriptive comparison on that sample. It does not prove significance, profitable trading, directional forecasting or persistent superiority. Daily targets overlap; five-session spacing does not establish independence. No test-set hyperparameter optimization is performed automatically. Changing settings after seeing results is still researcher discretion.

## Synthetic data reproducibility

```bash
python -m volatility_lab.demo
```

Seed 20261117 generates 1,801 synthetic adjusted-price-like closes on XNYS sessions, with seven changing volatility environments and correlated innovations. SPY/QQQ are labels for demonstration series; the generator is not calibrated to those ETFs. The two models are always estimated separately. Reproduction is expected under the pinned numerical/calendar environment. Screenshots and demo evidence must retain the SYNTHETIC label.

## Deploy on Streamlit Community Cloud

Upload this folder's **contents** to a GitHub repository. Keep `app.py`, `requirements.txt`, `volatility_lab/`, `docs/` and `data/` in the repository root. Select `app.py` as the entry point and Python 3.12 in the deployment settings. No secrets or persistent disk are required. `.streamlit/config.toml` sets the theme, 20 MB upload limit and disables usage-stat collection. Hosting instructions: [Streamlit deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

Source and parameter dependent results are cached in server memory for one hour (maximum four configurations); new settings/data invalidate the cache key. No user CSV is written to the project. Uploaded data reach the hosting server; provider permissions may favor local execution. Cache, virtual environments, secrets, personal CSVs and generated exports are ignored by Git. Only the two explicitly named synthetic/example CSV files are allowlisted.

## Layout

```text
app.py                         Streamlit views and memory cache
volatility_lab/data.py          CSV validation, calendar, returns and labels
volatility_lab/models.py        Historical, EWMA and GARCH forecasts
volatility_lab/evaluation.py    Chronological evaluation, masks, losses, export
volatility_lab/charts.py        Plotly graphics
volatility_lab/demo.py          Deterministic artificial-price generator
tests/                         Alignment, causality, units, gaps and app checks
docs/                          Methodology, source review and verification
data/                          Synthetic demo and short CSV template
```

HAR-RV is reserved for a future dataset of cleaned intraday returns and well-defined daily realized variance, including explicit treatment of overnight variation. Other extensions: data-vintage tracking, alternative innovations/leverage models, uncertainty and dependence-aware comparison tests. No order execution or trading backtest is included.
