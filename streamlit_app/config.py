# App settings.

# Checkpoints are listed from the checkpoints/ folder next to this file. The two
# settings below are only used when nothing is selected there.
# HTTPS location of a checkpoint to download (leave empty to skip).
GENERATOR_CHECKPOINT_URL = ""
# Local .pt/.pth file, absolute or relative to this folder (leave empty to skip).
GENERATOR_CHECKPOINT_PATH = ""

# Clear-sky model for the clear-sky GHI channel: "ineichen" (needs pvlib) or
# "haurwitz" (built into solflow_gen.geometry, no extra dependency). Use the one
# closest to the clear-sky column the checkpoint was trained on.
CLEARSKY_MODEL = "ineichen"

DEFAULT_POINT = {"lat": 45.46, "lon": 3.53, "alt": 1070.0}
MIN_ENSEMBLE = 1
MAX_ENSEMBLE = 25
DEFAULT_ENSEMBLE = 25
DEFAULT_SEED = 0

# Every panel in the main row has the same outer height in every state (empty,
# generating, results). Both chart tabs use PLOT_HEIGHT.
PANEL_HEIGHT = 680
PLOT_HEIGHT = 520
