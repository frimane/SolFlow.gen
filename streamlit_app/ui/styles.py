import streamlit as st

from ui.theme import CARD_HEIGHT, COLORS, CONTROL_HEIGHT, GAP, HEADER_HEIGHT

# CSS on top of the theme in .streamlit/config.toml. It gives every control the same
# height, styles the panel header, KPI cards and empty states, keeps spacing on a
# 12 px rhythm and hides Streamlit chrome the app does not use. Panel heights are not
# set here: each panel is a fixed-height st.container (see ui/layout.py).
_CSS = """
<style>
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none; }
[data-testid="stHeader"] { background:transparent; height:0; }
.block-container { max-width:1500px; padding:1.25rem 1.5rem 1rem; }

/* typography */
.panel-title { color:%text%; font-size:1rem; font-weight:600; line-height:1.3; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.panel-sub { color:%muted%; font-size:.8rem; font-weight:400; margin-left:.5rem; }
.group-label { color:%muted%; font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin:.5rem 0 .25rem; }

/* top bar */
.app-topbar { display:flex; align-items:center; justify-content:space-between; gap:%gap%px;
    padding-bottom:%gap%px; margin-bottom:%gap%px; border-bottom:1px solid %border%; }
.brand { color:%text%; font-size:1.4rem; font-weight:700; letter-spacing:-.01em; }
.brand-sol { color:#FFD400; font-weight:800; }
.brand-sub { color:%muted%; font-size:.9rem; font-weight:400; margin-left:.75rem; }


/* panels: fixed-height bordered containers, identical in every state */
[class*="st-key-panel_"] { background:%surface%; border-radius:8px; }
[class*="st-key-panel_"], [class*="st-key-panel_"] [data-testid="stVerticalBlock"] { gap:%gap%px; }
.panel-header { display:flex; align-items:center; height:%header%px; margin-bottom:4px; }

/* controls: one height, one radius, one border */
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div { min-height:%control%px; height:%control%px; box-sizing:border-box; }
[data-testid="stNumberInput"] [data-baseweb="input"], [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background:%surface_2%; border:1px solid %border%; }
[data-testid="stNumberInput"] input { font-variant-numeric:tabular-nums; }
div[data-testid="stWidgetLabel"] label p { color:%muted%; font-size:.78rem; font-weight:500; }
[data-testid="stForm"] { border:none; padding:0; }
[data-testid="stForm"] [data-testid="stVerticalBlock"], [data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"] { gap:%gap%px; }

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    height:%control%px; min-height:%control%px; padding:0 1rem; border-radius:6px;
    font-size:.875rem; font-weight:600; box-sizing:border-box; width:100%; }
.stButton > button[kind="secondary"], .stDownloadButton > button, .stFormSubmitButton > button[kind="secondary"] {
    background:%surface_2%; color:%text%; border:1px solid %border%; }
.stDownloadButton > button:hover:not(:disabled) { border-color:%accent%; color:%text%; }
.stDownloadButton > button:disabled { opacity:.45; }
.stFormSubmitButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] { background:%accent%; color:#fff; border:1px solid %accent%; }
.stFormSubmitButton > button:hover { filter:brightness(1.1); }
button:focus-visible, input:focus-visible, [data-baseweb="select"]:focus-within { outline:2px solid %accent% !important; outline-offset:2px; }

/* tabs: native underline style, only spacing adjusted */
[data-baseweb="tab-list"] { gap:1.25rem; }
[data-baseweb="tab"] { height:%control%px; padding:0 .25rem; font-size:.875rem; font-weight:600; }
[data-baseweb="tab-panel"] { padding-top:%gap%px; }
[data-testid="stPlotlyChart"] { width:100%; }

/* KPI cards */
.kpi-list { display:flex; flex-direction:column; gap:%gap%px; margin:8px 0 12px; }
.kpi-card { display:flex; flex-direction:column; justify-content:center; height:%card%px; box-sizing:border-box;
    padding:0 1rem; background:%surface_2%; border:1px solid %border%; border-radius:6px; }
.kpi-label { color:%muted%; font-size:.75rem; font-weight:500; }
.kpi-value { color:%text%; font-size:1.25rem; font-weight:600; line-height:1.3; margin-top:.15rem; font-variant-numeric:tabular-nums; }
.kpi-value .unit { color:%muted%; font-size:.78rem; font-weight:400; margin-left:.35rem; }
.kpi-value.empty { color:%muted%; }
[data-testid="stMarkdownContainer"] p.download-note { color:%muted%; font-size:.75rem; line-height:1.5; margin:0; }

/* empty / message state, same box as the plot */
.empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;
    height:var(--plot-h); box-sizing:border-box; padding:1.5rem; gap:.35rem;
    background:%surface_2%; border:1px dashed %border%; border-radius:6px; }
.empty-state .title { color:%text%; font-size:1rem; font-weight:600; }
.empty-state .copy { color:%muted%; font-size:.875rem; line-height:1.55; max-width:30rem; }
.empty-state.error { border-color:%danger%; }
.empty-state.error .title { color:%danger%; }

/* status widget + alerts use the same radius and border */
[data-testid="stStatus"], [data-testid="stAlert"] { border-radius:6px; }

/* tablets / landscape phones: stack the three panels */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex:1 1 100% !important; width:100% !important; min-width:100% !important; }
}

/* phone screens */
@media (max-width: 640px) {
    .block-container { padding:.75rem .75rem 1rem; }
    .brand { font-size:1.2rem; }
    .brand-sub { font-size:.8rem; margin-left:.5rem; }
    /* panels grow with their content instead of scrolling inside the page */
    [class*="st-key-panel_"],
    [class*="st-key-panel_"] > div,
    [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-panel_"]),
    [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-panel_"]) > div {
        height:auto !important; min-height:0 !important; max-height:none !important; overflow:visible !important; flex:0 0 auto !important; }
    /* Streamlit wraps each fixed-height container in a layout wrapper with flex-basis = height */
    [data-testid="stLayoutWrapper"]:has(> [class*="st-key-panel_"]) {
        flex:0 0 auto !important; height:auto !important; min-height:0 !important; max-height:none !important; }
    /* the three main columns must never sit on top of each other */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        flex-direction:column !important; flex-wrap:wrap !important; gap:12px !important; }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width:100% !important; min-width:100% !important; flex:0 0 auto !important; }
    /* keep paired inputs (From/To, Lat/Lon, Ensemble/Seed) side by side */
    [class*="st-key-panel_setup"] [data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; gap:.5rem; }
    [class*="st-key-panel_setup"] [data-testid="stColumn"] { min-width:0 !important; width:auto !important; }
    .empty-state { height:260px; padding:1rem; }
    .kpi-value { font-size:1.1rem; }
}
</style>
"""


def inject_styles(plot_height: int):
    css = _CSS.replace("%gap%", str(GAP)).replace("%header%", str(HEADER_HEIGHT)).replace("%control%", str(CONTROL_HEIGHT)).replace("%card%", str(CARD_HEIGHT))
    for name, value in COLORS.items():
        css = css.replace(f"%{name}%", value)
    css = css.replace("</style>", f":root {{ --plot-h:{int(plot_height)}px; }}</style>")
    st.markdown(css, unsafe_allow_html=True)
