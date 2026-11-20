"""Prior-session signals and chronological directional frequencies."""
import pandas as pd

NY = "America/New_York"
WINDOWS = {"Asia": (-1, "19:00", "02:00"), "New York": (0, "09:30", "16:00")}


def stamp(day, clock):
    return pd.Timestamp(f"{day:%Y-%m-%d} {clock}", tz=NY)


def outcomes(bars, day, previous):
    result = {}
    for name, (offset, start, end) in WINDOWS.items():
        a = stamp(day + pd.Timedelta(days=offset), start)
        b = stamp(day, end)
        expected = pd.date_range(a, b, freq="30min", inclusive="left")
        part = bars.reindex(expected)
        if part[["Open", "Close"]].notna().all().all():
            result[name] = part.Close.iloc[-1] - part.Open.iloc[0]
    prior = stamp(previous, "15:30")
    last = stamp(day, "15:30")
    if prior in bars.index and last in bars.index:
        result["Day"] = bars.at[last,"Close"] - bars.at[prior,"Close"]
    return result


def estimate(history, share):
    # Predefined put share buckets, with shrinkage to the training base rate.
    base = (sum(y for _, y in history) + 1) / (len(history) + 2)
    def bucket(x):
        return 0 if x < .45 else 2 if x > .55 else 1
    peers = [y for x,y in history if bucket(x) == bucket(share)]
    return (sum(peers) + 10 * base) / (len(peers) + 10), base, len(peers)
