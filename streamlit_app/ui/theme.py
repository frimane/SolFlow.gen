# Colours and sizes used across the UI.
#
# One accent colour (blue) for interactive elements and the main data series.
# Amber is only used for the clear-sky reference curve, red only for errors,
# everything else is neutral.

COLORS = {
    "bg": "#0b1220",        # page background
    "surface": "#111b2e",   # panels
    "surface_2": "#16233a", # cards / inputs inside a panel
    "border": "#26354f",
    "text": "#e6edf7",
    "muted": "#9aa9bf",
    "accent": "#2f6fe4",
    "accent_soft": "rgba(47,111,228,0.22)",
    "reference": "#f5b942",
    "danger": "#f26d6d",
    "grid": "#22324c",
}

# Sizes in px. The control height is applied to every input, select and button.
CONTROL_HEIGHT = 40
HEADER_HEIGHT = 40
CARD_HEIGHT = 76
GAP = 12
