"""Shared figure style: one categorical order, recessive axes, blue sequential and blue-red diverging maps."""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE = "#8a8984"  # exact solutions and the original code, drawn as neutral references
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f3b5b4", "#e34948", "#9c1f1f"]
)

# Fixed color per training variant so every figure agrees.
# The two variants without the Hamiltonian term train identical policies, so 'no_hjb'
# is dashed to keep both visible where their curves nearly coincide.
VARIANT_STYLE = {
    "full": {"label": "Full method (v1.1)", "color": SERIES[0], "marker": "o", "ls": "-"},
    "no_hamiltonian": {"label": "No Hamiltonian term", "color": SERIES[1], "marker": "v", "ls": "-"},
    "no_hjb_grad": {"label": "No residual-gradient terms", "color": SERIES[2], "marker": "s", "ls": "-"},
    "no_hjb": {"label": "No HJB terms", "color": SERIES[3], "marker": "^", "ls": "--"},
    "original": {"label": "Original code (v1.0)", "color": REFERENCE, "marker": "D", "ls": "-"},
}


def percent_log_axis(ax) -> None:
    """Log-scale the y axis of a plotted Axes and label it in percent, with ticks at 1, 2, 5 x 10^k."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [m * 10.0 ** k for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
             for m in (1, 2, 5)]
    ax.set_yticks([v for v in ticks if lo <= v <= hi])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.3g}%"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "legend.frameon": False,
    })
