# Verification record

Executed on 17 November 2026, Python 3.12.14, Linux.

**30 automated tests passed.** `python -m pytest -q` completed in approximately 14 seconds. `python -m pip check` found no broken requirements.

## Checks completed

- Exact target alignment, final five unobserved labels and incomplete future windows.
- Historical mean-square, EWMA initialization/recursion and analytical GARCH horizon recursion.
- Multiplication of returns by 100 and reversal of variance scaling by 10,000.
- Hand-calculated MAE, RMSE and QLIKE, zero-floor counts and empty samples.
- Altering prices strictly after an origin does not change its forecast for any model or any horizon; the future outcome does change.
- Strict rolling training boundaries; injected GARCH nonconvergence retains the all-dates baseline and fixed non-overlap grid.
- Duplicate/invalid prices, missing-session returns, optional QQQ, short histories, unsorted dates, exchange holiday and early-close validation.
- Real Streamlit AppTest execution of all four views, SPY and QQQ, daily/non-overlapping modes, selected forecast inspection and both CSV upload outcomes.
- Uploaded bytes were injected at the file-uploader boundary in AppTest because its native picker is not supported. All downstream validation, forecasts and views used production code.
- Streamlit server startup: health endpoint HTTP 200 `ok`, page HTTP 200.

## Verification limits

The server and Streamlit widget journeys were verified, but a rendered-browser visual inspection could not be completed in this environment: installed Chromium failed to create its process socket, and the optional headless-browser download was invalid. Native file-picker and browser-download interaction remain unverified. No visual screenshot is represented as validated. macOS and Community Cloud were not executed directly; Python 3.12 dependencies and standard Streamlit entry point are prepared for them. No paid real-data access was tested.

## Reproducible synthetic experiment

1,801 prices / 1,800 returns per instrument; 2017-01-03 through 2024-02-29. Seed 20261117. Default training window 756, lambda 0.94, 252 evaluation origins, five future sessions. These are artificial-data diagnostics, not market results.

| Asset | Sampling | Model | Common N | MAE (vol pp) | RMSE (vol pp) | QLIKE |
|---|---|---|---:|---:|---:|---:|
| SPY | Daily | Historical 20 | 252 | 3.8467 | 4.9208 | 0.2721 |
| SPY | Daily | EWMA | 252 | 4.2685 | 5.3303 | 0.2733 |
| SPY | Daily | GARCH(1,1) | 252 | 4.1866 | 5.3070 | 0.2565 |
| SPY | Every 5 sessions | Historical 20 | 51 | 3.7755 | 4.7949 | 0.2442 |
| SPY | Every 5 sessions | EWMA | 51 | 4.1727 | 5.1775 | 0.2488 |
| SPY | Every 5 sessions | GARCH(1,1) | 51 | 4.0098 | 5.0874 | 0.2286 |
| QQQ | Daily | Historical 20 | 252 | 5.5937 | 7.3492 | 0.2785 |
| QQQ | Daily | EWMA | 252 | 5.8443 | 7.8237 | 0.2675 |
| QQQ | Daily | GARCH(1,1) | 252 | 5.8951 | 7.8584 | 0.2607 |
| QQQ | Every 5 sessions | Historical 20 | 51 | 5.3340 | 7.5056 | 0.2744 |
| QQQ | Every 5 sessions | EWMA | 51 | 5.8440 | 8.1588 | 0.2783 |
| QQQ | Every 5 sessions | GARCH(1,1) | 51 | 5.9715 | 8.2406 | 0.2750 |

All 252 default GARCH fits converged for each synthetic instrument. No baseline dates were lost to model exclusion in this default example; the injected-failure tests explicitly verify exclusion behavior.

Dataset SHA-256: `509601a96d002a1bbc0d058c8dc5d276a5e96a8d79fea7698dc5be2fff7a99f9`.

No significance test, trading-P&L claim, or real-data model ranking is inferred from this demonstration.
