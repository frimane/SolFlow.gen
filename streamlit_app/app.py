from __future__ import annotations

import sys
from pathlib import Path

# Make the solflow_gen package importable when the app is started from this folder
# without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from checkpoint import load_checkpoint
from checkpoint_source import resolve_checkpoint_path
from config import PLOT_HEIGHT
from inference import generate_date_range
from ui.control_panel import render_control_panel
from ui.layout import render_topbar
from ui.results import render_empty, render_loading, render_results
from ui.styles import inject_styles

st.set_page_config(page_title="SolFlow geometry generator", layout="wide", initial_sidebar_state="collapsed")
inject_styles(PLOT_HEIGHT)
render_topbar()

# Three panels of identical height: setup | plot | summary.
left_col, plot_col, kpi_col = st.columns([3, 6.4, 3], gap="medium")
plot_area = plot_col.empty()
kpi_area = kpi_col.empty()
submitted, start_date, end_date, lat, lon, alt, ensemble_size, seed = render_control_panel(left_col)


def validate_range(start, end):
    if end < start:
        raise ValueError("To date must be on or after From date.")
    if end.year != start.year:
        raise ValueError("From and To dates must be in the same calendar year; December-to-January ranges are not supported.")
    if (end - start).days > 365:
        raise ValueError("The generated range cannot exceed one calendar year.")


if submitted:
    try:
        validate_range(start_date, end_date)
        checkpoint_path = resolve_checkpoint_path(st.session_state.get("checkpoint_selector"))
        body = render_loading(plot_area, kpi_area)
        with body:
            with st.status("Generating samples", expanded=True) as status:
                st.write("Loading checkpoint…")
                model = load_checkpoint(str(checkpoint_path), checkpoint_path.stat().st_mtime)
                n_dates = (end_date - start_date).days + 1
                st.write(f"Generating {n_dates} requested day(s) with a {model.n_days}-day checkpoint horizon…")
                result = generate_date_range(model, lat, lon, alt, start_date, end_date, ensemble_size, seed)
                status.update(label="Samples ready", state="complete", expanded=False)
        st.session_state["result"] = result
    except Exception as exc:
        # Shown inside the plot panel so the layout never changes.
        render_empty(plot_area, kpi_area, "Could not generate samples", str(exc), kind="error")
        st.session_state.pop("result", None)
        st.stop()

if "result" in st.session_state:
    render_results(plot_area, kpi_area, st.session_state["result"])
else:
    render_empty(plot_area, kpi_area)
