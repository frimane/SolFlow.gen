import streamlit as st

from config import PANEL_HEIGHT

KPI_ITEMS = (
    ("Selected-day peak P50", "W/m²"),
    ("Typical P10–P90 band", "W/m²"),
    ("Generated range", "day(s)"),
    ("Ensemble members", "members"),
)


def render_topbar():
    st.markdown(
        '<div class="app-topbar"><div><span class="brand"><span class="brand-sol">sol</span>Flow</span>'
        '<span class="brand-sub"> Generator</span></div></div>',
        unsafe_allow_html=True,
    )


def panel(name: str, tag: str = "main"):
    # Fixed-height bordered panel; all main panels use the same height. The tag keeps
    # Streamlit element keys unique when two states of the same panel (loading, then
    # results) are drawn in one script run.
    return st.container(border=True, height=PANEL_HEIGHT, key=f"panel_{name}_{tag}")


def panel_header(title: str, subtitle: str = ""):
    sub = f'<span class="panel-sub">{subtitle}</span>' if subtitle else ""
    st.markdown(f'<div class="panel-header"><div class="panel-title">{title}{sub}</div></div>', unsafe_allow_html=True)


def empty_state(title: str, copy: str, kind: str = "info"):
    cls = "empty-state error" if kind == "error" else "empty-state"
    st.markdown(f'<div class="{cls}"><div class="title">{title}</div><div class="copy">{copy}</div></div>', unsafe_allow_html=True)


def kpi_cards(values=None):
    # Always the same four cards; values show a dash until a run exists.
    values = values or [None] * len(KPI_ITEMS)
    cards = []
    for (label, unit), value in zip(KPI_ITEMS, values):
        if value is None:
            body = '<div class="kpi-value empty">—</div>'
        else:
            body = f'<div class="kpi-value">{value}<span class="unit">{unit}</span></div>'
        cards.append(f'<div class="kpi-card"><div class="kpi-label">{label}</div>{body}</div>')
    st.markdown(f'<div class="kpi-list">{"".join(cards)}</div>', unsafe_allow_html=True)
