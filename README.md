# Volatility Forecasting Lab

**Project 02 · public quantitative research portfolio · Markets / Sales & Trading**

Can the next five sessions of SPY realized volatility be forecast better than a trailing 20-session historical reference? This Streamlit lab compares Historical 20, EWMA and zero-mean normal GARCH(1,1), using rolling-origin forecasts and matched evaluation dates. QQQ is a separate selectable experiment.

The app downloads adjusted daily closes for SPY and QQQ from Yahoo Finance on opening. Bundled synthetic prices are test fixtures.

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

1. Open the app to see the SPY forecast. The first evaluation runs automatically.
2. Select QQQ or apply different research settings. GARCH is reestimated at every origin.
3. Open **Historical evaluation**. All three main rows use identical dates. Compare the extra baseline row to see exclusion effects.
4. Switch between daily and every-five-session origins. The five-session grid is anchored before exclusions, not formed by thinning surviving forecasts.
5. Select a forecast origin. Inspect every training return, the prior price anchoring the first return, each daily prediction and the five subsequent outcomes.
6. Open **Model diagnostics** for residual errors, forecast/outcome scatter, worst cases, convergence and GARCH parameters. **Data & methodology** contains provenance, validation and all formulas.
7. Download the per-origin CSV. Pending and excluded rows remain present with reasons. Latest-close results have a separate download if outside the evaluation interval.

The overview displays the most recent valid forecast per model among the selected evaluation origins and the latest dataset origin. Failed latest fits are explicitly marked unavailable or stale; a stale card retains its actual origin/horizon. Do not interpret it as a current forecast.

## Market data

The app requests twelve years of Yahoo Finance adjusted closes through the last completed XNYS session. Downloads are cached for one hour and keyed by session. Gaps remain visible in the audit. If Yahoo is unavailable, use the retry button. See [data provenance](docs/data_sources.md).

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

Upload this folder's **contents** to a GitHub repository. Keep `app.py`, `requirements.txt`, `volatility_lab/`, `docs/` and `data/` in the repository root. Select `app.py` as the entry point and Python 3.12 in the deployment settings. No secrets or persistent disk are required. Hosting instructions: [Streamlit deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

Source and parameter dependent results are cached in server memory for one hour (maximum four configurations); new settings/data invalidate the cache key.

## Layout

```text
app.py                         Streamlit views and memory cache
volatility_lab/yahoo.py         Yahoo Finance adjusted-close retrieval
volatility_lab/data.py          Input validation, calendar, returns and labels
volatility_lab/models.py        Historical, EWMA and GARCH forecasts
volatility_lab/evaluation.py    Chronological evaluation, masks, losses, export
volatility_lab/charts.py        Plotly graphics
volatility_lab/demo.py          Deterministic artificial-price generator
tests/                         Alignment, causality, units, gaps and app checks
docs/                          Methodology, source review and verification
data/                          Synthetic demo and short CSV template
```

HAR-RV is reserved for a future dataset of cleaned intraday returns and well-defined daily realized variance, including explicit treatment of overnight variation. Other extensions: data-vintage tracking, alternative innovations/leverage models, uncertainty and dependence-aware comparison tests. No order execution or trading backtest is included.
