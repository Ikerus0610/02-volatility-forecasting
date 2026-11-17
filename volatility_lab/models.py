"""Forecasts consume only a supplied trailing return window."""
from dataclasses import dataclass, field
import warnings
import numpy as np
from arch import arch_model

MODEL_NAMES = ("Historical 20", "EWMA", "GARCH(1,1)")
SCALE = 100.0


@dataclass
class Forecast:
    variance: float = np.nan
    daily: tuple = (np.nan,) * 5
    status: str = "unavailable"
    message: str = ""
    parameters: dict = field(default_factory=dict)


def historical(returns, horizon=5):
    x = np.asarray(returns, dtype=float)[-20:]
    if len(x) != 20 or not np.isfinite(x).all():
        return Forecast(message="Requires 20 consecutive valid returns.")
    value = float(np.mean(x*x))
    return Forecast(value, (value,)*horizon, "ok")


def ewma(returns, lam=0.94, horizon=5):
    x = np.asarray(returns, dtype=float)
    if not 0 < lam < 1:
        raise ValueError("EWMA lambda must be strictly between 0 and 1.")
    if len(x) < 20 or not np.isfinite(x).all():
        return Forecast(message="Requires a complete trailing training window.")
    # Seed with the first 20 squared returns in THIS trailing training window.
    # Seed represents the next-day variance after those 20 observations.
    variance = float(np.mean(x[:20]**2))
    for r in x[20:]:
        variance = lam * variance + (1-lam) * r*r
    # No mean reversion in EWMA: E_t[h(t+k)] = h(t+1) for every k >= 1.
    return Forecast(float(variance), (float(variance),)*horizon, "ok", parameters={"lambda": lam})


def garch(returns, horizon=5, maxiter=500):
    x = np.asarray(returns, dtype=float)
    if len(x) < 20 or not np.isfinite(x).all():
        return Forecast(message="Requires a complete trailing training window.")
    if np.std(x) < 1e-10:
        return Forecast(status="degenerate", message="Return standard deviation below 1e-10; GARCH not identifiable.")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = arch_model(x * SCALE, mean="Zero", vol="GARCH", p=1, o=0, q=1,
                               dist="normal", rescale=False)
            fit = model.fit(disp="off", update_freq=0, show_warning=False,
                            options={"maxiter": maxiter, "ftol": 1e-9})
            params = {"omega_daily": float(fit.params["omega"] / SCALE**2),
                      "alpha": float(fit.params["alpha[1]"]), "beta": float(fit.params["beta[1]"]),
                      "convergence_flag": int(fit.convergence_flag), "iterations": int(fit.optimization_result.nit)}
            params["persistence"] = params["alpha"] + params["beta"]
            message = " | ".join(str(w.message) for w in caught)
            if fit.convergence_flag != 0:
                return Forecast(status="convergence_failed", message=str(fit.optimization_result.message), parameters=params)
            daily = fit.forecast(horizon=horizon, method="analytic", reindex=False).variance.iloc[-1].to_numpy() / SCALE**2
            if not np.isfinite(daily).all() or (daily <= 0).any():
                return Forecast(status="invalid_forecast", message="Non-positive or non-finite forecast.", parameters=params)
            return Forecast(float(daily.mean()), tuple(daily), "ok", message, params)
    except Exception as exc:
        # One numerical failure must not abort the full chronological evaluation.
        return Forecast(status="fit_error", message=f"{type(exc).__name__}: {exc}")
