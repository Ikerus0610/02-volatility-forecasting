"""Adjusted-close input, exchange-session alignment and future labels."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import csv
import numpy as np
import pandas as pd
import exchange_calendars as xcals

HORIZON = 5
ANNUALIZATION = 252


class DataError(ValueError):
    pass


@dataclass
class Dataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    audit: pd.DataFrame
    warnings: list[str]
    fingerprint: str
    source: str


def calendar(start, end):
    # XNYS is the common US cash-equity session calendar used for both ETFs.
    return xcals.get_calendar("XNYS", start=pd.Timestamp(start), end=pd.Timestamp(end))


def sessions(start, end):
    return calendar(start, end).sessions.tz_localize(None)


def future_dates(origin, horizon=HORIZON):
    origin = pd.Timestamp(origin)
    idx = sessions(origin, origin + pd.Timedelta(days=40))
    return idx[idx > origin][:horizon]


def load_csv(raw: bytes, source="User CSV", now=None) -> Dataset:
    if len(raw) > 20_000_000:
        raise DataError("CSV exceeds the 20 MB limit.")
    try:
        text = raw.decode("utf-8-sig")
        header = next(csv.reader(text.splitlines()))
        if len(header) != len(set(header)):
            raise DataError("Duplicate column names are not allowed.")
        frame = pd.read_csv(BytesIO(raw), dtype=str)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise DataError("Use a UTF-8, comma-separated CSV with a date,SPY header.") from exc
    if not {"date", "SPY"}.issubset(frame.columns):
        raise DataError("Required columns: date,SPY. QQQ is optional.")
    if len(frame) < 22:
        raise DataError("At least 22 price rows are required. Use 762+ consecutive prices for a 756-return window and one evaluated forecast.")
    dates = frame.date.fillna("").str.strip()
    if not dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise DataError("Dates must be unambiguous YYYY-MM-DD session dates, without timestamps.")
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y-%m-%d", errors="raise"))
    except ValueError as exc:
        raise DataError("Invalid calendar date.") from exc
    if idx.has_duplicates:
        duplicates = idx[idx.duplicated()].strftime("%Y-%m-%d").tolist()
        raise DataError("Duplicate session dates: " + ", ".join(duplicates[:8]))
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    if idx.min() < pd.Timestamp("1993-01-29") or idx.max() > now.tz_convert("UTC").tz_localize(None).normalize():
        raise DataError("Use historical session dates from 1993-01-29 through today; future rows are not accepted.")
    cal = calendar(idx.min(), idx.max())
    grid = cal.sessions.tz_localize(None)
    extra = idx.difference(grid)
    if len(extra):
        raise DataError("Non-trading dates under XNYS: " + ", ".join(extra.strftime("%Y-%m-%d")[:8]))
    if cal.session_close(idx.max()) > now:
        raise DataError("The most recent session has not closed. Remove its unfinished daily price.")
    warnings = []
    if not idx.is_monotonic_increasing:
        warnings.append("Input dates were sorted into chronological order.")
    ignored = sorted(set(frame.columns) - {"date", "SPY", "QQQ"})
    if ignored:
        warnings.append("Ignored columns: " + ", ".join(ignored))
    prices = pd.DataFrame(index=idx)
    for asset in [a for a in ("SPY", "QQQ") if a in frame]:
        original = frame[asset].fillna("").str.strip()
        numeric = pd.to_numeric(original, errors="coerce")
        bad = original.ne("") & (~np.isfinite(numeric) | numeric.le(0))
        if bad.any():
            raise DataError(f"{asset}: {int(bad.sum())} invalid price(s). Prices must be positive finite numbers; leave missing cells blank.")
        if numeric.notna().sum() < 22:
            raise DataError(f"{asset}: fewer than 22 valid prices. Remove an unused optional QQQ column.")
        prices[asset] = numeric.to_numpy(dtype=float)
    prices = prices.sort_index().reindex(grid)
    prices.index.name = "date"
    # Do not drop missing sessions before taking differences: that would bridge gaps.
    returns = np.log(prices).diff()
    rows = []
    missing_rows = grid.difference(idx)
    for asset in prices:
        valid = returns[asset].notna()
        runs = valid.groupby((~valid).cumsum()).sum()
        missing = prices[asset].isna()
        gaps = returns[asset].isna() & prices[asset].notna()
        gaps.iloc[0] = False
        rows.append(dict(asset=asset, input_rows=len(frame), expected_sessions=len(grid),
                         absent_session_rows=len(missing_rows), missing_prices=int(missing.sum()),
                         returns_blocked_after_gap=int(gaps.sum()), valid_returns=int(valid.sum()),
                         longest_return_run=int(runs.max()), first_date=grid[0], last_date=grid[-1]))
        if missing.any():
            warnings.append(f"{asset}: {int(missing.sum())} missing prices. No interpolation or forward fill; returns across gaps are invalid.")
        if (returns[asset].abs() > 0.25).any():
            warnings.append(f"{asset}: absolute log returns above 25%; verify adjustments and source quality.")
    return Dataset(prices, returns, pd.DataFrame(rows), warnings, sha256(raw).hexdigest(), source)


def future_target(returns: pd.Series, horizon=HORIZON) -> pd.Series:
    """At t: mean(r[t+1]^2,...,r[t+h]^2). Require every future session."""
    forward = pd.concat([returns.shift(-h).pow(2) for h in range(1, horizon + 1)], axis=1)
    return forward.mean(axis=1, skipna=False).rename("realized_variance")


def annualized_vol(variance):
    """Decimal annualized volatility, e.g. 0.20 = 20%."""
    return np.sqrt(ANNUALIZATION * np.asarray(variance, dtype=float))
