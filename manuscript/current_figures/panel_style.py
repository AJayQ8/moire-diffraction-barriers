"""Place panel letters and headings on a common line above the plotting area."""


def panel_heading(ax, letter, title=None, *, pad=9, fontsize=8.3):
    """Use a fixed typographic offset so gutters and data limits do not move labels."""
    heading = dict(
        xy=(0, 1), xycoords="axes fraction", textcoords="offset points",
        ha="left", va="baseline", fontsize=fontsize, color="#202529",
        annotation_clip=False, clip_on=False,
    )
    ax.annotate(f"({letter})", xytext=(0, pad), fontweight="bold", **heading)
    if title:
        ax.annotate(title, xytext=(18, pad), **heading)
