"""Real Streamlit script tests: default run and all four views for both assets."""
from pathlib import Path
from io import BytesIO
import pandas as pd
from streamlit.testing.v1 import AppTest
from streamlit.delta_generator import DeltaGenerator

APP=Path(__file__).resolve().parents[1]/'app.py'


def find(elements,label):
    return next(x for x in elements if x.label==label)


def test_complete_app_journey():
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    assert not at.exception
    assert any('SYNTHETIC DATA' in i.value for i in at.info)
    find(at.button,'Run evaluation').click().run(timeout=120)
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


def test_uploaded_source_empty_state():
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    find(at.radio,'Data source').set_value('Upload CSV').run()
    assert not at.exception
    assert any('Upload adjusted daily closes' in x.value for x in at.info)


def test_csv_upload_branch_and_pending_dates(monkeypatch):
    # AppTest does not drive native file pickers. Supply the same uploaded-byte
    # object at that boundary and exercise the production validation/run/views.
    raw=(APP.parent/'data/synthetic_prices.csv').read_bytes()
    monkeypatch.setattr(DeltaGenerator,'file_uploader',lambda self,*args,**kwargs: BytesIO(raw))
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    find(at.radio,'Data source').set_value('Upload CSV').run()
    assert not at.exception
    assert any('uploaded copy' in x.value for x in at.caption)
    date_input=find(at.date_input,'Evaluation origins')
    date_input.set_value((pd.Timestamp('2024-02-20').date(),pd.Timestamp('2024-02-29').date()))
    find(at.button,'Run evaluation').click().run(timeout=120)
    find(at.radio,'Research views').set_value('Historical evaluation').run()
    assert not at.exception
    assert any('Future outcome unavailable: pending' in x.value for x in at.info)
    assert any(m.label=='Outcome unavailable' and m.value=='5' for m in at.metric)


def test_csv_upload_reports_bad_data(monkeypatch):
    monkeypatch.setattr(DeltaGenerator,'file_uploader',lambda self,*args,**kwargs: BytesIO(b'date,SPY\n2024-01-02,-1\n'))
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    find(at.radio,'Data source').set_value('Upload CSV').run()
    assert not at.exception
    assert at.error
