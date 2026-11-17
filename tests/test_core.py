from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from volatility_lab.data import load_csv, DataError, sessions, future_target, annualized_vol
from volatility_lab.demo import generate, DEMO_PATH
from volatility_lab.models import historical, ewma, garch, MODEL_NAMES
from volatility_lab.evaluation import (Config, forecast_at, evaluate, common_rows, metric_table,
                                      metrics, losses, QLIKE_FLOOR, export_results)


@pytest.fixture(scope="module")
def demo():
    return load_csv(DEMO_PATH.read_bytes(), "SYNTHETIC TEST")


def serialize(frame):
    return frame.to_csv(index=True).encode()


def test_target_alignment_and_last_five_unobserved():
    r = pd.Series(np.arange(1., 13.)/100, index=pd.date_range("2020-01-01", periods=12))
    target = future_target(r)
    assert target.iloc[2] == pytest.approx(np.mean((np.arange(4., 9.)/100)**2))
    assert target.iloc[-6] == pytest.approx(np.mean(r.iloc[-5:]**2))
    assert target.iloc[-5:].isna().all()


def test_target_rejects_partial_future_window():
    r = pd.Series([.01]*12)
    r.iloc[5] = np.nan
    y = future_target(r)
    assert y.iloc[:5].isna().all()
    assert y.iloc[5] == pytest.approx(.0001)  # current r[t] is not in the target


def test_annualization_and_variance_units():
    assert annualized_vol(.0001) == pytest.approx(np.sqrt(252)*.01)
    assert annualized_vol(0) == 0
    assert np.isnan(annualized_vol(np.nan))


def test_historical_is_mean_square_including_origin_not_sample_variance():
    r = np.arange(1, 26)/1000
    out = historical(r)
    assert out.variance == pytest.approx(np.mean(r[-20:]**2))
    assert out.daily == (out.variance,)*5
    assert historical([.01]*19).status != "ok"


def test_ewma_seed_recursion_and_flat_horizon():
    x = np.r_[np.full(20, .01), .02, -.03]
    expected = .94*(.94*.0001+.06*.0004)+.06*.0009
    assert ewma(x).variance == pytest.approx(expected)
    assert ewma(x).daily == (ewma(x).variance,)*5
    assert ewma(x[:20]).variance == pytest.approx(.0001)


def test_garch_units_and_five_step_analytic_recursion(demo):
    train = demo.returns.SPY.iloc[1:757]
    result = garch(train)
    fit = arch_model(train.to_numpy()*100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
        disp="off", update_freq=0, show_warning=False, options={"maxiter":500, "ftol":1e-9})
    assert result.status == "ok"
    expected = fit.forecast(horizon=5, reindex=False).variance.iloc[-1].to_numpy()/10000
    np.testing.assert_allclose(result.daily, expected, rtol=1e-9)
    assert result.variance == pytest.approx(expected.mean())
    p = result.parameters
    h1 = p["omega_daily"] + p["alpha"]*train.iloc[-1]**2 + p["beta"]*fit.conditional_volatility[-1]**2/10000
    assert result.daily[0] == pytest.approx(h1)
    for i in range(1, 5):
        assert result.daily[i] == pytest.approx(p["omega_daily"]+p["persistence"]*result.daily[i-1])


def test_garch_degenerate_and_iteration_failure(demo):
    assert garch(np.zeros(100)).status == "degenerate"
    failed = garch(demo.returns.SPY.iloc[1:300], maxiter=1)
    assert failed.status == "convergence_failed"
    assert np.isnan(failed.variance)
    assert failed.parameters["convergence_flag"] != 0


def test_no_future_information_changes_forecast(demo):
    origin = demo.prices.index[900]
    config = Config(756)
    before = forecast_at(demo.returns.SPY, origin, config)
    changed = demo.prices.copy()
    future = changed.index > origin
    changed.loc[future, "SPY"] *= np.exp(np.linspace(.1, .8, future.sum()))
    after_data = load_csv(serialize(changed), "PERTURBED FUTURE")
    after = forecast_at(after_data.returns.SPY, origin, config)
    # Includes every horizon, parameters, seed-dependent EWMA and fit diagnostics.
    for model in MODEL_NAMES:
        assert before[model] == pytest.approx(after[model], rel=1e-12)
        for j in range(1,6):
            assert before[f"{model} h{j}"] == pytest.approx(after[f"{model} h{j}"], rel=1e-12)
    for parameter in ["omega_daily", "alpha", "beta"]:
        assert before[parameter] == pytest.approx(after[parameter], rel=1e-12)
    assert future_target(demo.returns.SPY).loc[origin] != pytest.approx(future_target(after_data.returns.SPY).loc[origin])


def test_window_is_strictly_trailing_and_includes_t(demo, monkeypatch):
    import volatility_lab.evaluation as ev
    observed=[]
    original=ev.garch
    def capture(x):
        observed.append(x.copy())
        return original(x)
    monkeypatch.setattr(ev,"garch",capture)
    origin=demo.prices.index[810]
    row=forecast_at(demo.returns.SPY,origin,Config(126))
    pd.testing.assert_series_equal(observed[0],demo.returns.SPY.iloc[685:811])
    assert row['training_count']==126
    assert row['training_end']==origin


def test_metrics_hand_calculation_and_units():
    actual=np.array([.01,.04])/252
    prediction=np.array([.04,.04])/252
    m=metrics(actual,prediction)
    assert m['MAE_pp']==pytest.approx(5)
    assert m['RMSE_pp']==pytest.approx(np.sqrt(50))
    assert m['QLIKE']==pytest.approx((.25-np.log(.25)-1)/2)


def test_zero_floor_disclosure_and_no_floor_in_volatility_loss():
    out=metrics([0,QLIKE_FLOOR*2],[0,0])
    assert out['target_floored']==1
    assert out['forecast_floored']==2
    assert out['QLIKE']==pytest.approx((2-np.log(2)-1)/2)
    assert out['MAE_pp']==pytest.approx(100*np.sqrt(252*2*QLIKE_FLOOR)/2)
    assert metrics([],[])['N']==0
    assert np.isnan(metrics([],[])['MAE_pp'])
    with pytest.raises(ValueError):
        losses([-.1],[.1])


def test_nonoverlap_grid_fixed_before_failed_fits_and_baseline_all_dates(demo,monkeypatch):
    import volatility_lab.evaluation as ev
    from volatility_lab.models import Forecast
    monkeypatch.setattr(ev,'garch',lambda x: Forecast(status='convergence_failed',message='injected test failure'))
    r=evaluate(demo.returns.SPY,Config(60),demo.prices.index[80],demo.prices.index[95])
    assert list(r.index[r.non_overlapping])==list(r.index[::5])
    assert len(common_rows(r))==0
    table=metric_table(r)
    assert list(table.N)==[0,0,0,16]
    assert metric_table(r,True).iloc[-1].N==4
    assert len(r)==16


def test_common_dates_pending_and_export(demo):
    r=evaluate(demo.returns.SPY,Config(126),demo.prices.index[-12],demo.prices.index[-1])
    assert r.iloc[-5:].outcome_status.eq('pending').all()
    assert r.iloc[-5:].realized_variance.isna().all()
    assert r.iloc[-5:]['Historical 20'].notna().all()
    assert not r.iloc[-5:].common_valid.any()
    assert len(common_rows(r))==7
    table=metric_table(r)
    assert table.iloc[:3].N.nunique()==1
    out=pd.read_csv(BytesIO(export_results(r,demo,'SPY',Config(126))))
    assert len(out)==12
    assert out.data_sha256.eq(demo.fingerprint).all()
    np.testing.assert_allclose(out['GARCH(1,1) annualized_vol_pct'],annualized_vol(r['GARCH(1,1)'])*100)


def test_missing_sessions_never_bridged(demo):
    frame=demo.prices.iloc[:100].copy()
    removed=frame.index[40]
    next_date=frame.index[41]
    data=load_csv(serialize(frame.drop(removed)))
    assert pd.isna(data.returns.at[removed,'SPY'])
    assert pd.isna(data.returns.at[next_date,'SPY'])
    assert data.audit.set_index('asset').at['SPY','absent_session_rows']==1
    assert data.audit.set_index('asset').at['SPY','returns_blocked_after_gap']==1
    assert pd.isna(future_target(data.returns.SPY).iloc[39])
    assert forecast_at(data.returns.SPY,frame.index[70],Config(60))['EWMA status']=='unavailable'
    assert forecast_at(data.returns.SPY,frame.index[70],Config(60))['Historical 20 status']=='ok'


@pytest.mark.parametrize('invalid',[0,-1,np.inf,'oops'])
def test_invalid_prices_rejected(demo,invalid):
    frame=demo.prices.iloc[:30].astype(object)
    frame.iloc[10,0]=invalid
    with pytest.raises(DataError,match='invalid price'):
        load_csv(serialize(frame))


def test_duplicates_and_non_sessions_rejected(demo):
    frame=demo.prices.iloc[:30]
    with pytest.raises(DataError,match='Duplicate session'):
        load_csv(serialize(pd.concat([frame,frame.iloc[[3]]])))
    with pytest.raises(DataError,match='Non-trading'):
        load_csv(serialize(frame.rename(index={frame.index[0]:pd.Timestamp('2017-01-01')})))


def test_sort_optional_asset_missing_cells_and_short_history(demo):
    frame=demo.prices.iloc[:30][['SPY']].copy()
    frame.iloc[15,0]=np.nan
    data=load_csv(serialize(frame.iloc[::-1]))
    assert list(data.prices)==['SPY']
    assert data.prices.index.is_monotonic_increasing
    assert data.audit.iloc[0].missing_prices==1
    assert any('sorted' in w for w in data.warnings)
    with pytest.raises(DataError,match='At least 22'):
        load_csv(serialize(frame.iloc[:10]))


def test_session_calendar_holiday_early_close_and_unfinished_data(demo):
    dates=sessions('2024-07-01','2024-07-08')
    assert pd.Timestamp('2024-07-04') not in dates
    assert pd.Timestamp('2024-07-03') in dates
    frame=pd.DataFrame({'SPY':100.},index=sessions('2024-05-01','2024-07-03'))
    frame.index.name='date'
    with pytest.raises(DataError,match='has not closed'):
        load_csv(serialize(frame),now='2024-07-03T16:59:00Z')
    assert len(load_csv(serialize(frame),now='2024-07-03T17:01:00Z').prices)==len(frame)


def test_synthetic_generation_reproduces_bundled_prices(demo):
    pd.testing.assert_frame_equal(generate(),demo.prices,check_exact=False,atol=6e-11,rtol=1e-12,check_freq=False)


@pytest.mark.parametrize('window,lam',[(59,.94),(2521,.94),(756,1),(756,0)])
def test_invalid_model_configuration(window,lam):
    with pytest.raises(ValueError):
        Config(window,lam)
