"""Colours, hatches and markers of the paper's figures (ts-figures/plotcolours.py)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = ["#0070DE", "#ABD473", "#9482C9", "#C41F3B", "#00FF96", "#A330C9"]
HATCHES = ["/", "\\", "|", "-", "+", "x"]
MARKERS = ["o", "s", "D", "^", "v", ">"]

PAPER_STYLE = dict(color="black", marker="_", markersize=18, markeredgewidth=2.5, linestyle="none", zorder=5)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        "patch.edgecolor": "black",
        "patch.force_edgecolor": True,
        "savefig.bbox": "tight",
    }
)
