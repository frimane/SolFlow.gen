import streamlit as st

from inference import csv_bytes, metadata, npz_bytes, result_table
from ui.charts import PLOT_CONFIG, build_csi_figure, build_ghi_figure
from ui.layout import empty_state, kpi_cards, panel, panel_header

DAY_KEY = "selected_generator_day"


def _label(value):
    return value.strftime("%b %d").replace(" 0", " ")


def _plot_header(title, subtitle="", labels=None, tag="main"):
    # Header row shared by every plot-panel state: title left, day selector right.
    left, right = st.columns([4, 1.2], vertical_alignment="top")
    with right:
        if labels:
            selected = st.selectbox("Plot day", labels, key=DAY_KEY, label_visibility="collapsed")
        else:
            st.selectbox("Plot day", ["—"], key=f"plot_day_placeholder_{tag}", disabled=True, label_visibility="collapsed")
            selected = None
    with left:
        panel_header(title, subtitle)
    return selected


def _plot_tabs(empty_title=None, empty_copy="", kind="info", selected=None):
    # GHI and CSI tabs. Without a selected day both tabs show the same-sized empty box.
    ghi_tab, csi_tab = st.tabs(["GHI", "CSI"])
    for tab, name, builder in ((ghi_tab, "ghi", build_ghi_figure), (csi_tab, "csi", build_csi_figure)):
        with tab:
            if selected is None:
                empty_state(empty_title, empty_copy, kind)
            else:
                st.plotly_chart(builder(selected), width="stretch", config=PLOT_CONFIG, key=f"plot_{name}")


def _downloads(result=None, tag="main"):
    # Three buttons, always present; disabled until a run exists.
    if result is None:
        for label in ("Download all days CSV", "Download all days NPZ", "Download range metadata"):
            st.download_button(label, b"", file_name="unavailable", disabled=True, width="stretch", key=f"dl_off_{tag}_{label}")
        return
    if st.session_state.get("_downloads_for") != id(result):
        st.session_state["_downloads"] = (csv_bytes(result), npz_bytes(result), metadata(result))
        st.session_state["_downloads_for"] = id(result)
    csv, npz, meta = st.session_state["_downloads"]
    stem = f"geometry_samples_{result['start_date']:%Y%m%d}_{result['end_date']:%Y%m%d}"
    st.download_button("Download all days CSV", csv, f"{stem}.csv", "text/csv", width="stretch", key="dl_csv")
    st.download_button("Download all days NPZ", npz, f"{stem}.npz", "application/octet-stream", width="stretch", key="dl_npz")
    st.download_button("Download range metadata", meta, f"{stem}.json", "application/json", width="stretch", key="dl_json")


def _summary_panel(kpi_area, values=None, result=None, tag="main"):
    with kpi_area.container():
        with panel("summary", tag):
            panel_header("Summary")
            kpi_cards(values)
            _downloads(result, tag)
            if result is not None and result["skipped_days"]:
                st.markdown(f'<p class="download-note">{len(result["skipped_days"])} day(s) without a usable daylight window were skipped.</p>',
                            unsafe_allow_html=True)
            st.markdown(
                '<p class="download-note">Selected day changes only the plot. CSV and NPZ downloads contain every generated day: '
                "UTC timestamps, quantiles, samples, geometry and run metadata.</p>",
                unsafe_allow_html=True,
            )


def render_empty(plot_area, kpi_area, title="No samples yet", copy=None, kind="info"):
    # Pre-run and error state: same panels, header, tabs and plot box as the results.
    copy = copy or "Choose a date range and location, then select Generate samples to create an ensemble of whole-day GHI trajectories. The app uses solar geometry and clear-sky GHI only."
    with plot_area.container():
        with panel("plot"):
            _plot_header("Generated samples")
            _plot_tabs(title, copy, kind)
    _summary_panel(kpi_area)


def render_loading(plot_area, kpi_area):
    # Returns the panel body, which the caller fills with the status widget.
    with plot_area.container():
        with panel("plot", "loading"):
            _plot_header("Generated samples", tag="loading")
            body = st.container()
    _summary_panel(kpi_area, tag="loading")
    return body


def render_results(plot_area, kpi_area, result):
    days = result["days"]
    labels = [_label(d["forecast_date"]) for d in days]
    if st.session_state.get(DAY_KEY) not in labels:
        st.session_state.pop(DAY_KEY, None)
    with plot_area.container():
        with panel("plot"):
            # Header is rendered first so the selector exists before we read it.
            left, right = st.columns([4, 1.2], vertical_alignment="top")
            with right:
                selected_label = st.selectbox("Plot day", labels, key=DAY_KEY, label_visibility="collapsed")
            selected = days[labels.index(selected_label)]
            with left:
                panel_header(selected_label, f'{selected["lat"]:.4f}°, {selected["lon"]:.4f}° · {result["ensemble_size"]}-member ensemble')
            table = result_table(selected)
            _plot_tabs(selected=selected)
    day = table.loc[table["valid"].to_numpy(bool)]
    peak = float(day["ghi_p50_wm2"].max())
    spread = float((day["ghi_p90_wm2"] - day["ghi_p10_wm2"]).median())
    _summary_panel(kpi_area, [f"{peak:.0f}", f"{spread:.0f}", f"{len(days)}", f'{result["ensemble_size"]}'], result)
