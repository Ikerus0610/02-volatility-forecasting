"""Real Streamlit script tests: default run and all four views for both assets."""
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest
from volatility_lab.yahoo import download_prices

APP=Path(__file__).resolve().parents[1]/'app.py'


def find(elements,label):
    return next(x for x in elements if x.label==label)


@pytest.fixture(autouse=True)
def yahoo(monkeypatch):
    prices = pd.read_csv(APP.parent / 'data/synthetic_prices.csv', parse_dates=['date']).set_index('date')
    response = pd.concat({'Adj Close': prices, 'Close': prices * 2}, axis=1)
    download_prices.clear()
    monkeypatch.setattr('volatility_lab.yahoo.yf.download', lambda *a, **kw: response)
    yield
    download_prices.clear()


def test_complete_app_journey():
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    assert not at.exception
    assert any('Yahoo Finance' in i.value for i in at.caption)
    assert not at.exception
    assert any(m.label=='GARCH(1,1)' and '%' in m.value for m in at.metric)
    find(at.radio,'Research views').set_value('Historical evaluation').run()
    assert not at.exception
    find(at.radio,'Evaluation sampling').set_value('Every 5 sessions (non-overlapping)').run()
    assert not at.exception
    assert any(m.label=='Common evaluated dates' and int(m.value)>0 for m in at.metric)
    origin=find(at.selectbox,'Forecast origin')
    origin.set_value(pd.Timestamp(origin.options[0])).run()
    assert not at.exception
    find(at.radio,'Research views').set_value('Model diagnostics').run()
    assert not at.exception
    find(at.radio,'Research views').set_value('Data & methodology').run()
    assert not at.exception
    find(at.selectbox,'Instrument').set_value('QQQ').run()
    find(at.button,'Run evaluation').click().run(timeout=120)
    assert not at.exception
    find(at.radio,'Research views').set_value('Forecast overview').run()
    assert not at.exception
    assert any('QQQ' in x.value for x in at.caption)
    assert any(m.label=='GARCH(1,1)' and '%' in m.value for m in at.metric)


def test_provider_failure(monkeypatch):
    monkeypatch.setattr("volatility_lab.yahoo.yf.download", lambda *a, **kw: pd.DataFrame())
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("Yahoo Finance data unavailable" in e.value for e in at.error)
    assert not at.metric
    assert find(at.button, "Retry Yahoo Finance")


def test_pending_dates_at_end_of_download():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    dates = find(at.date_input, 'Evaluation origins')
    dates.set_value((pd.Timestamp('2024-02-20').date(), pd.Timestamp('2024-02-29').date()))
    find(at.button, 'Run evaluation').click().run(timeout=120)
    find(at.radio, 'Research views').set_value('Historical evaluation').run()
    assert not at.exception
    assert any('Future outcome unavailable: pending' in x.value for x in at.info)
    assert any(m.label == 'Outcome unavailable' and m.value == '5' for m in at.metric)
