"""Matplotlib style and color palette (academic, deep red on white)."""

import matplotlib.pyplot as plt

PALETTE = {
    "primary":    "#7B1F23",   # deep red — main accent
    "secondary":  "#A65A57",   # red-brown — secondary
    "tertiary":   "#3F3F3F",   # near-black — text / axis
    "muted":      "#8C8C8C",   # gray — grid / reference
    "light":      "#D9D9D9",   # light gray — fills
    "background": "#FFFFFF",

    # 4-asset colors
    "SPY": "#7B1F23",
    "AGG": "#3F3F3F",
    "PE":  "#A65A57",
    "NPI": "#8C8C8C",
}

# Sequential red colormap for heatmaps (white -> deep red)
SEQ_CMAP = "Reds"
DIV_CMAP = "RdGy_r"


def apply_style():
    """Set matplotlib rcParams. Call once at startup."""
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     PALETTE["tertiary"],
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titleweight":   "bold",
        "axes.titlesize":     12,
        "axes.titlepad":      10,
        "axes.labelsize":     10,
        "axes.labelcolor":    PALETTE["tertiary"],
        "axes.prop_cycle":    plt.cycler(color=[
            PALETTE["primary"], PALETTE["tertiary"],
            PALETTE["secondary"], PALETTE["muted"],
        ]),
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.color":        PALETTE["tertiary"],
        "ytick.color":        PALETTE["tertiary"],
        "legend.frameon":     False,
        "legend.fontsize":    9,
        "grid.color":         PALETTE["light"],
        "grid.linewidth":     0.5,
        "grid.alpha":         0.7,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset":   "cm",
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })
