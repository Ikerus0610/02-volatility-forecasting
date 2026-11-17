"""Deterministic artificial prices. No downloaded market data."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import sessions

SEED = 20261117
ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "data" / "synthetic_prices.csv"


def generate():
    dates = sessions("2017-01-03", "2025-01-01")[:1801]
    rng = np.random.default_rng(SEED)
    # Deliberately changing environments, not fitted to SPY or QQQ history.
    levels = np.repeat([0.007, 0.018, 0.009, 0.026, 0.011, 0.022, 0.008],
                       [350, 250, 250, 300, 250, 150, 250])
    z = rng.standard_normal((1800, 2))
    z[:, 1] = 0.65*z[:, 0] + np.sqrt(1-0.65**2)*z[:, 1]
    returns = np.zeros((1800, 2))
    variance = np.array([levels[0]**2, (levels[0]*1.45)**2])
    previous = np.zeros(2)
    for i, level in enumerate(levels):
        long_var = np.array([level**2, (level*1.45)**2])
        variance = 0.03*long_var + 0.08*previous**2 + 0.89*variance
        previous = np.sqrt(variance)*z[i]
        returns[i] = previous
    logpath = np.vstack([np.zeros(2), np.cumsum(returns, axis=0)])
    frame = pd.DataFrame(np.exp(logpath)*[250., 150.], index=dates, columns=["SPY", "QQQ"])
    frame.index.name = "date"
    return frame


if __name__ == "__main__":
    generate().to_csv(DEMO_PATH, float_format="%.10f")
    print(f"Wrote {DEMO_PATH.name}: SYNTHETIC DATA, seed {SEED}")
