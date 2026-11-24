import numpy as np
import plotly.graph_objects as go

from config import PLOT_HEIGHT
from ui.theme import COLORS

_FONT = dict(color=COLORS["muted"], size=12)
_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": False}
PLOT_CONFIG = _CONFIG


def _masked(result, key):
    valid = np.asarray(result["conditions"]["fut_mask"][0], dtype=bool)
    return np.where(valid[None, :], result[key], np.nan), valid


def _layout(fig, y_title, y_range=None):
    # Same height, margins, legend and axes for every chart.
    fig.update_layout(
        height=PLOT_HEIGHT,
        margin=dict(l=8, r=8, t=40, b=8),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_FONT,
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Time (UTC)", tickformat="%H:%M", gridcolor=COLORS["grid"], zeroline=False, automargin=True),
        yaxis=dict(title=y_title, gridcolor=COLORS["grid"], zeroline=False, rangemode="tozero", range=y_range, automargin=True),
    )
    return fig


def _members_and_band(fig, t, samples, p10, p50, p90, unit_fmt, unit):
    idx = np.linspace(0, samples.shape[0] - 1, min(15, samples.shape[0]), dtype=int)
    for n, i in enumerate(idx):
        fig.add_trace(go.Scatter(x=t, y=samples[i], mode="lines", line=dict(width=1, color="rgba(154,169,191,0.25)"),
                                 name="Ensemble members", legendgroup="members", showlegend=(n == 0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=p10, mode="lines", fill="tonexty", fillcolor=COLORS["accent_soft"], line=dict(width=0),
                             name="P10–P90", customdata=p90,
                             hovertemplate=f"P10–P90: %{{y:{unit_fmt}}} – %{{customdata:{unit_fmt}}}{unit}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=p50, mode="lines", line=dict(color=COLORS["accent"], width=3), name="P50",
                             hovertemplate=f"P50: %{{y:{unit_fmt}}}{unit}<extra></extra>"))


def build_ghi_figure(result):
    t = result["conditions"]["times"]
    samples, valid = _masked(result, "samples_ghi")
    p10, p50, p90 = np.percentile(samples, [10, 50, 90], axis=0)
    clear = np.where(valid, result["conditions"]["fut_ghi_cs"][0], np.nan)
    fig = go.Figure()
    _members_and_band(fig, t, samples, p10, p50, p90, ".0f", " W/m²")
    fig.add_trace(go.Scatter(x=t, y=clear, mode="lines", line=dict(color=COLORS["reference"], width=2, dash="dash"),
                             name="Clear sky", hovertemplate="Clear sky: %{y:.0f} W/m²<extra></extra>"))
    return _layout(fig, "GHI (W/m²)")


def build_csi_figure(result):
    t = result["conditions"]["times"]
    samples, valid = _masked(result, "samples_csi")
    p10, p50, p90 = np.percentile(samples, [10, 50, 90], axis=0)
    fig = go.Figure()
    _members_and_band(fig, t, samples, p10, p50, p90, ".2f", "")
    fig.add_trace(go.Scatter(x=t, y=np.where(valid, 1.0, np.nan), mode="lines", line=dict(color=COLORS["reference"], width=2, dash="dash"),
                             name="Clear sky (CSI = 1)", hovertemplate="Clear sky: 1.00<extra></extra>"))
    return _layout(fig, "CSI (–)")
