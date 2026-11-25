import datetime as dt

import streamlit as st

from config import DEFAULT_POINT, DEFAULT_ENSEMBLE, DEFAULT_SEED, MAX_ENSEMBLE, MIN_ENSEMBLE
from ui.layout import panel, panel_header


def _month_day_label(value):
    return value.strftime("%b %d").replace(" 0", " ")


def render_control_panel(left_col):
    with left_col:
        with panel("setup"):
            panel_header("Generator setup")

            # The date selectors live outside the form so that "To date" always
            # offers only dates on or after the chosen "From date".
            today = dt.date.today()
            year_start, year_end = dt.date(today.year, 1, 1), dt.date(today.year, 12, 31)
            available = [year_start + dt.timedelta(days=i) for i in range((year_end - year_start).days + 1)]
            st.markdown('<div class="group-label">Date range</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            start_value = st.session_state.get("range_start_selector", today)
            if start_value not in available:
                start_value = today
            start_date = c1.selectbox("From", available, index=available.index(start_value), key="range_start_selector", format_func=_month_day_label)
            end_dates = [start_date + dt.timedelta(days=i) for i in range((year_end - start_date).days + 1)]
            end_value = st.session_state.get("range_end_selector", start_date)
            if end_value not in end_dates:
                end_value = start_date
            end_date = c2.selectbox("To", end_dates, index=end_dates.index(end_value), key="range_end_selector", format_func=_month_day_label)

            with st.form("generator_controls", clear_on_submit=False, border=False):
                st.markdown('<div class="group-label">Location</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                latitude = c1.number_input("Latitude (°)", -90.0, 90.0, DEFAULT_POINT["lat"], step=0.0001, format="%.4f")
                longitude = c2.number_input("Longitude (°)", -180.0, 180.0, DEFAULT_POINT["lon"], step=0.0001, format="%.4f")
                altitude_m = st.number_input("Elevation (m)", -500.0, 9000.0, DEFAULT_POINT["alt"], step=10.0, format="%.0f")
                st.markdown('<div class="group-label">Sampling</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                ensemble_size = c1.number_input("Ensemble size", MIN_ENSEMBLE, MAX_ENSEMBLE, min(DEFAULT_ENSEMBLE, MAX_ENSEMBLE), step=1, format="%d")
                seed = c2.number_input("Random seed", 0, 2**31 - 1, DEFAULT_SEED, step=1, format="%d")
                submitted = st.form_submit_button("Generate samples", type="primary", width="stretch")

            st.caption(
                "The checkpoint was trained with McClear (CAMS). The generated clear-sky index "
                "comes raw from the checkpoint, and the conversion to GHI is done with the "
                "Ineichen (pvlib) clear-sky model. For exact results, use McClear."
            )
    return submitted, start_date, end_date, float(latitude), float(longitude), float(altitude_m), int(ensemble_size), int(seed)