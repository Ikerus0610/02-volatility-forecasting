import os
import sqlite3
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote
from urllib.error import HTTPError
import json
import pandas as pd

from research import NY

def get(key, path, **params):
    url = "https://api.optionsdata.ai/v1/" + path + "?" + urlencode(params)
    req = Request(url, headers={"Authorization":"Bearer " + key, "Accept":"application/json"})
    try:
        with urlopen(req, timeout=20) as response:
            if response.headers.get("X-Ignored-Query-Parameters"):
                raise ValueError("Provider ignored requested parameters.")
            return json.load(response)
    except HTTPError as exc:
        raise ValueError(f"OptionsData HTTP {exc.code} on {path.split('/')[0]}") from None


def expirations(key):
    rows = get(key,"underlyings/SPX").get("records",{}).get("term",[])
    return sorted({r["expiration"] for r in rows if r.get("expiration")})


def capture(key, day):
    now = pd.Timestamp.now(tz=NY)
    if now.date() != day.date() or not (16*60+16 <= now.hour*60+now.minute < 17*60):
        raise ValueError("Capture on a regular US session from 16:16 to 16:59 New York time.")
    dates = [d for d in expirations(key) if d > day.date().isoformat()]
    if not dates:
        raise ValueError("No accessible upcoming SPX expiration.")
    snap = get(key,"chains/SPX",expiration=dates[0], records="contract,trade")
    asof = pd.Timestamp(int(snap["as_of"]), unit="ns", tz="UTC").tz_convert(NY)
    if asof.date()!=now.date() or not 0 <= (now-asof).total_seconds() <= 300:
        raise ValueError("Stale chain snapshot.")
    records = snap["records"]
    volumes = {r["iid"]:r.get("vendor_day_volume") for r in records.get("trade",[])}
    totals = {"C":0, "P":0}
    count = 0
    for row in records.get("contract",[]):
        symbol = row["symbol"]
        if len(symbol)!=21 or symbol[:6].strip() not in ("SPX","SPXW"):
            continue
        volume = volumes.get(row["iid"])
        if volume is None or int(volume)<0:
            raise ValueError("Coverage incomplete: missing contract day volume.")
        totals[symbol[12]] += int(volume)
        count += 1
    if sum(totals.values())==0:
        raise ValueError("No usable option volume.")
    return (str(day.date()),now.isoformat(),dates[0],totals["C"],totals["P"],count)


def database():
    connection = sqlite3.connect(os.getenv("ES_LAB_DB","es_lab.sqlite"))
    connection.execute("CREATE TABLE IF NOT EXISTS signals(day TEXT PRIMARY KEY, available TEXT, expiry TEXT, calls INTEGER, puts INTEGER, contracts INTEGER)")
    return connection


def save(row):
    with database() as db:
        db.execute("INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?)",row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--osi", help="Exact 21-character historical option symbol")
    parser.add_argument("--date", help="Trading date YYYY-MM-DD for a one-minute history probe")
    args = parser.parse_args()
    key = os.environ.get("OPTIONSDATA_API_KEY")
    if not key:
        parser.exit(2,"Set OPTIONSDATA_API_KEY locally first.\n")
    if args.probe:
        print("Accessible SPX expirations:", expirations(key))
    if args.osi and args.date:
        start=pd.Timestamp(args.date+" 10:00",tz=NY)
        data=get(key,"contracts/"+quote(args.osi,safe="")+"/trades",start=start.isoformat(),end=(start+pd.Timedelta(minutes=1)).isoformat(),limit=10)
        print("History probe: first-page rows",len(data.get("rows",[])),"watermark",data.get("complete_through"),"fidelity",data.get("fidelity"))
    if args.capture:
        import exchange_calendars as xc
        now=pd.Timestamp.now(tz=NY)
        day=now.tz_localize(None).normalize()
        cal=xc.get_calendar("XNYS")
        if not cal.is_session(day) or cal.session_close(day).tz_convert(NY).hour!=16:
            parser.exit(2,"Regular US session required.\n")
        save(capture(key,day))
        print("Snapshot saved.")
