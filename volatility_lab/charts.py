import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .data import annualized_vol
from .models import MODEL_NAMES
from .evaluation import losses

COLORS = {"Historical 20": "#5b728e", "EWMA": "#c88732", "GARCH(1,1)": "#087f83", "Observed": "#253046"}


def style(fig, title, ytitle):
    fig.update_layout(title=dict(text=title, font=dict(size=17)), template="plotly_white", height=410,
                      margin=dict(l=20, r=20, t=65, b=40), font=dict(family="Arial", color="#253046"),
                      legend=dict(orientation="h", y=1.13, x=0), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title=ytitle, gridcolor="#e9edf2")
    fig.update_xaxes(title="Forecast origin (after close)", showgrid=False)
    return fig


def forecast_chart(rows, label):
    fig = go.Figure()
    for name, col in [("Observed", "realized_variance"), *[(m, m) for m in MODEL_NAMES]]:
        fig.add_trace(go.Scatter(x=rows.index, y=annualized_vol(rows[col])*100, name=name,
                                line=dict(color=COLORS[name], width=2.4 if name == "Observed" else 1.6),
                                hovertemplate="%{y:.2f}%<extra>"+name+"</extra>"))
    return style(fig, f"Five-session volatility · {label}", "Annualized volatility (%)")


def error_chart(rows, label):
    fig = go.Figure()
    for model in MODEL_NAMES:
        error, _ = losses(rows.realized_variance, rows[model])
        fig.add_trace(go.Scatter(x=rows.index, y=error, name=model, line=dict(color=COLORS[model])))
    fig.add_hline(y=0, line_dash="dot", line_color="#778399")
    return style(fig, f"Forecast minus observed · {label}", "Volatility error (percentage points)")


def scatter_chart(rows, label):
    fig = go.Figure()
    maximum = 1.
    actual = annualized_vol(rows.realized_variance)*100
    for model in MODEL_NAMES:
        prediction = annualized_vol(rows[model])*100
        if len(rows):
            maximum = max(maximum, actual.max(), prediction.max())
        fig.add_trace(go.Scatter(x=prediction, y=actual, name=model, mode="markers",
                                customdata=rows.index.strftime("%Y-%m-%d"),
                                marker=dict(color=COLORS[model], size=6, opacity=.65),
                                hovertemplate="%{customdata}<br>Forecast %{x:.2f}%<br>Observed %{y:.2f}%<extra>"+model+"</extra>"))
    fig.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], name="Equality", mode="lines",
                            line=dict(dash="dot", color="#929baa"), showlegend=False))
    style(fig, f"Forecast vs observed · {label}", "Observed annualized volatility (%)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Forecast annualized volatility (%)")
    return fig
