"""Shared visual theme for all pyVPLM plots.

The palette is a colorblind-safe categorical set (worst adjacent CVD distance
validated at 24, target is >= 12) assigned by role, not by position, so a series
keeps its color across every plot of the package. Apply the theme with the
:func:`theme` context manager so it never leaks into user figures::

    with theme():
        fig, ax = plt.subplots()
"""

from cycler import cycler
from matplotlib import rc_context

# -------[Categorical palette (fixed slot order - CVD-safe)]---------------------
PALETTE = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# -------[Colors by role: a series keeps its color across all plots]-------------
COLORS = {
    "primary": PALETTE[0],  # main data (fitting set, DOE points, regression)
    "secondary": PALETTE[5],  # comparison (cross-validation set, fitted curve)
    "tertiary": PALETTE[1],  # third series (nearest feasible points)
    "context": "#b5b4ad",  # background point clouds, de-emphasized
    "context_dark": "#52514e",  # foreground neutral (objective DOE points)
    "reference": "#8a897f",  # y=x lines, mean markers - recessive gray
    "highlight_bg": "#e8f1fc",  # 'best model' cell background (soft blue tint)
}

# light blue ramp used for magnitude backgrounds (impact factor); deliberately
# capped at light tints so foreground data marks stay clearly visible
SEQUENTIAL = ["#ffffff", "#f3f8fe", "#e8f1fc", "#dbe9fa", "#cde2fb"]

_RC = {
    "axes.prop_cycle": cycler(color=PALETTE),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
}


def theme():
    """Return a context manager applying the pyVPLM plot style.

    Based on :func:`matplotlib.rc_context`: the style only affects figures
    created inside the ``with`` block and is fully restored afterwards.
    """
    return rc_context(_RC)


def sequential_color(fraction: float) -> str:
    """Map a value in [0, 1] to the light->dark sequential background ramp."""
    fraction = min(1.0, max(0.0, float(fraction)))
    return SEQUENTIAL[round(fraction * (len(SEQUENTIAL) - 1))]
