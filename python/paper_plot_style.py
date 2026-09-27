"""Shared Matplotlib typography for figures included at PRC column width."""


PAPER_RC_PARAMS = {
    # Paper plots are produced on a 7-inch canvas and included at about
    # 3.4 inches. Axis labels therefore render at about 9.3 pt and tick
    # labels at about 8.8 pt, matching the 9 pt REVTeX caption text.
    "font.size": 15,
    "axes.labelsize": 19,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
}
