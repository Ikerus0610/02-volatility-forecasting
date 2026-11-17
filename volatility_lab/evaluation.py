"""Rolling-origin forecasts, common-date selection, metrics and audit export."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import HORIZON, annualized_vol, future_target, future_dates
from .models import MODEL_NAMES, historical, ewma, garch

QLIKE_FLOOR = 1e-12  # daily decimal-return variance; only used in QLIKE


@dataclass(frozen=True)
class Config:
    window: int = 756
    lam: float = 0.94

    def __post_init__(self):
        if not 60 <= self.window <= 2520:
            raise ValueError("Training window must contain 60 to 2520 returns.")
        if not 0 < self.lam < 1:
            raise ValueError("EWMA lambda must lie between 0 and 1.")


def forecast_at(returns: pd.Series, origin, config: Config) -> dict:
    """The forecasting boundary. Never accesses any observation after origin."""
    history = returns.loc[:origin]
    train = history.iloc[-config.window:]
    row = dict(origin=pd.Timestamp(origin), training_start=train.index[0], training_end=history.index[-1],
               training_count=int(train.notna().sum()), training_window=config.window, ewma_lambda=config.lam,
               historical_start=history.iloc[-20:].index[0])
    forecasts = {"Historical 20": historical(history)}
    if len(train) == config.window and train.notna().all():
        forecasts["EWMA"], forecasts["GARCH(1,1)"] = ewma(train, config.lam), garch(train)
    else:
        from .models import Forecast
        for name in MODEL_NAMES[1:]:
            forecasts[name] = Forecast(message=f"Requires {config.window} consecutive valid returns through this origin.")
    for name, result in forecasts.items():
        row[name] = result.variance
        row[f"{name} status"] = result.status
        row[f"{name} message"] = result.message
        for j, value in enumerate(result.daily, 1):
            row[f"{name} h{j}"] = value
        if name == "GARCH(1,1)":
            row.update(result.parameters)
    return row


def evaluate(returns, config, start, end):
    origins = returns.index[(returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))]
    if not len(origins):
        raise ValueError("No session origins in the selected interval.")
    targets = future_target(returns)
    rows = []
    for number, origin in enumerate(origins):
        row = forecast_at(returns, origin, config)
        position = returns.index.get_loc(origin)
        row["realized_variance"] = targets.loc[origin]
        row["outcome_status"] = ("pending" if position + HORIZON >= len(returns) else
                                 "missing_future_data" if pd.isna(targets.loc[origin]) else "observed")
        row["non_overlapping"] = number % HORIZON == 0
        # Future dates are deterministic calendar metadata, never model inputs.
        future = returns.index[position+1:position+HORIZON+1]
        if len(future) < HORIZON:
            future = future_dates(origin)
        row["target_start"], row["target_end"] = future[0], future[-1]
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("origin")
    frame["common_valid"] = frame.realized_variance.notna() & frame[list(MODEL_NAMES)].notna().all(axis=1)
    return frame


def selected_rows(results, non_overlapping=False):
    return results.loc[results.non_overlapping] if non_overlapping else results


def common_rows(results, non_overlapping=False):
    subset = selected_rows(results, non_overlapping)
    return subset.loc[subset.common_valid]


def losses(actual, forecast):
    y, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if y.shape != f.shape or not np.isfinite(y).all() or not np.isfinite(f).all() or (y < 0).any() or (f < 0).any():
        raise ValueError("Loss inputs must have identical shapes and finite non-negative variances.")
    error = (annualized_vol(f) - annualized_vol(y)) * 100  # percentage points
    yf, ff = np.maximum(y, QLIKE_FLOOR), np.maximum(f, QLIKE_FLOOR)
    qlike = np.maximum(yf/ff - (np.log(yf)-np.log(ff)) - 1.0, 0.0)
    return error, qlike


def metrics(actual, forecast):
    error, qlike = losses(actual, forecast)
    if not len(error):
        return dict(N=0, MAE_pp=np.nan, RMSE_pp=np.nan, QLIKE=np.nan, target_floored=0, forecast_floored=0)
    return dict(N=len(error), MAE_pp=float(np.abs(error).mean()), RMSE_pp=float(np.sqrt(np.mean(error**2))),
                QLIKE=float(qlike.mean()), target_floored=int((np.asarray(actual) < QLIKE_FLOOR).sum()),
                forecast_floored=int((np.asarray(forecast) < QLIKE_FLOOR).sum()))


def metric_table(results, non_overlapping=False):
    shared = common_rows(results, non_overlapping)
    rows = [dict(model=name, sample="Common dates", **metrics(shared.realized_variance, shared[name])) for name in MODEL_NAMES]
    subset = selected_rows(results, non_overlapping)
    baseline = subset.loc[subset.realized_variance.notna() & subset[MODEL_NAMES[0]].notna()]
    rows.append(dict(model=MODEL_NAMES[0], sample="All available baseline dates", **metrics(baseline.realized_variance, baseline[MODEL_NAMES[0]])))
    return pd.DataFrame(rows)


def export_results(results, dataset, asset, config):
    out = results.copy()
    for name in [*MODEL_NAMES, "realized_variance"]:
        out[f"{name} annualized_vol_pct"] = annualized_vol(out[name]) * 100
    out.insert(0, "asset", asset)
    out.insert(0, "data_sha256", dataset.fingerprint)
    out.insert(0, "data_source", dataset.source)
    out["horizon_sessions"] = HORIZON
    out["qlike_variance_floor"] = QLIKE_FLOOR
    out["non_overlap_anchor"] = results.index[0]
    out["variance_units"] = "mean daily squared decimal log return"
    return out.to_csv(index=True).encode("utf-8")
