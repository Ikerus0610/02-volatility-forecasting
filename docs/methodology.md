### Research question and timing

Can SPY's next-five-session realized volatility be forecast better than its recent history? QQQ is a separate univariate experiment. Each origin is **after the completed close of session t**. No future observations enter a model, initialization or fit. The evaluation interval selects forecast origins, not training-data start dates. History before that interval remains available for training.

### Target and units

For adjusted closing price $P_t$, the decimal daily log return is

$$r_t=\log(P_t)-\log(P_{t-1}).$$

The target attached to origin t is the average squared return of the next five exchange sessions:

$$v_{t,5}=\frac{1}{5}\sum_{j=1}^{5}r_{t+j}^{2},\qquad \sigma_{t,5}^{ann}=\sqrt{252v_{t,5}}.$$

Stored variances are **mean daily squared decimal log returns**. Displayed volatility is $100\sigma^{ann}$ percent. This is a close-to-close proxy for realized variation, without demeaning; it includes overnight moves but cannot measure the intraday path or distinguish jumps. It is neither the squared five-day cumulative return nor a high-frequency realized-variance estimator. Annualization is a reporting convention, not a forecast for the coming year. Taking a square root of predicted variance does not in general equal the conditional expectation of realized volatility.

The final five input sessions have no complete future label. Missing prices inside a future window also invalidate its outcome. A forecast may exist even though its outcome is pending.

### Three comparable forecasts

**Historical 20.** $\hat v^{H}_{t,5}=\frac{1}{20}\sum_{i=0}^{19}r_{t-i}^{2}$. All five predicted daily variances equal this value. The reference needs 20 consecutive returns even if the other models need a longer window.

**EWMA.** Default $\lambda=0.94$. At every origin, use the same trailing W-return window as GARCH (default W=756). Initialize the next-day variance after the first 20 returns to their mean square. Process each remaining return in order using $h_{s+1}=\lambda h_s+(1-\lambda)r_s^2$, including $r_t$. This is reinitialized on each rolling window, not on the complete dataset. EWMA has no variance mean reversion; all five conditional expected variances equal $h_{t+1}$. Their mean is the five-session target forecast. Lambda is user-specified, never optimized on evaluation outcomes.

**GARCH(1,1).** $r_s=\sqrt{h_s}z_s$, $z_s\sim N(0,1)$; $h_{s+1}=\omega+\alpha r_s^2+\beta h_s$. Fit a zero-mean normal model afresh on each trailing W-return window. For subsequent horizons, $E_t[h_{t+j}]=\omega+(\alpha+\beta)E_t[h_{t+j-1}]$. Average the five analytical forecasts. The optimizer receives $100r$ and automatic rescaling is disabled; divide every predicted variance and omega by $100^2$. Alpha and beta are dimensionless. The library's default initial variance backcast and bounds use only this training window. No warm start from later fits, no fallback substitution. Near-unit persistence is reported rather than interpreted as a stable long-run variance.

Only convergence flag 0 with finite positive predicted variances is accepted. Exceptions and nonconvergence are logged by origin. A training-window return standard deviation below 1e-10 is deemed degenerate and GARCH is unavailable. Maximum optimizer iterations: 500; objective tolerance: 1e-9. Positive forecasts are not clipped or winsorized.

### Chronology and missing sessions

Inputs are aligned to the **XNYS exchange-session calendar** supplied by pinned `exchange-calendars`, used as a common US equity calendar for SPY and QQQ. Holidays and early-close days are respected. This is a maintained third-party calendar, not a direct exchange feed; unusual closures should be checked against the original data source. Completed session prices only; dates after the current close are rejected. Date strings must be YYYY-MM-DD. Duplicate dates, non-session dates and non-positive/non-finite supplied prices are rejected. Blank/NA prices remain missing and are audited. Unsorted dates are sorted with a notice. Large returns over 25% in absolute log terms are flagged, not clipped.

Reindex **before** differencing. If Tuesday is absent, both Tuesday's return and Wednesday's return are unavailable; a Monday-to-Wednesday move is never used as a one-day return. No interpolation, zero filling or forward filling. Historical 20 needs a complete 20-session return window; EWMA and GARCH require W consecutive valid returns. Missing dates at the beginning/end outside the CSV's span cannot be detected. Input adjustment quality and point-in-time data revisions cannot be established from a price CSV. A modern adjusted history is a retrospective data snapshot: causal code alone does not establish point-in-time data-vintage purity.

### Evaluation and losses

The three-model comparison uses exactly the same origins with all three valid forecasts and observed targets. A fourth metrics row reports Historical 20 on **all its available observed origins within the selected interval/sampling grid**, exposing the effect of exclusions. Origin-level status, pending labels and data failures remain in the CSV. Missing results are never converted to zero.

Let $e_i=100(\sqrt{252\hat v_i}-\sqrt{252v_i})$. Then

$$MAE=\frac{1}{N}\sum_i|e_i|,\qquad RMSE=\sqrt{\frac{1}{N}\sum_i e_i^2}.$$

Both are in **annualized-volatility percentage points**. For QLIKE only, set $\tilde v=\max(v,10^{-12})$ and $\tilde f=\max(\hat v,10^{-12})$:

$$QLIKE=\frac{1}{N}\sum_i\left[\frac{\tilde v_i}{\tilde f_i}-\log\left(\frac{\tilde v_i}{\tilde f_i}\right)-1\right].$$

This dimensionless normalized form is zero at equality and lower is better. The floor makes zero outcomes/forecasts finite, changes their loss and is explicitly counted for both columns. No variance floor is used for displayed forecasts or MAE/RMSE. Floating-point negative QLIKE roundoff is bounded at zero. Empty samples have undefined metrics, not zero errors.

Daily origins have overlapping targets. The alternative uses origins 0, 5, 10, ... counted on the session grid from the **first selected origin before any exclusions**, preventing missing fits from moving the grid. It removes target overlap but not necessarily serial dependence. No p-values, independence claims or significance tests are provided. Model comparisons are descriptive; choosing settings repeatedly after examining outcomes can overfit this evaluation period. No transaction-cost or trading-P&L claim follows from these losses.

### Scope and extensions

Volatility describes magnitude, not the direction of the return. The normal zero-mean GARCH specification omits leverage, fat-tailed innovations and parameter uncertainty. HAR-RV is deferred: it needs a trustworthy daily series of realized variance built from cleaned intraday returns (for example, five-minute observations), explicit session and overnight conventions, and daily/weekly/monthly lagged realized-variance components. It should not be presented as an intraday HAR-RV implementation using only squared daily closes.

Reference: [arch analytical forecasting documentation](https://arch.readthedocs.io/en/stable/univariate/forecasting.html), [exchange_calendars project](https://github.com/gerrymanoim/exchange_calendars).
