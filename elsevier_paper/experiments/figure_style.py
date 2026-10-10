"""Shared journal artwork typography; use at the intended manuscript width.

Liberation Sans is an Arial-compatible TrueType face. The PDF backend embeds
its fonts, while SVG exports outline glyphs for portable offline viewing.
The elsarticle body and captions keep their template fonts independently.
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

# Measured from elsarticle[preprint,12pt]: textwidth=390 TeX points.
# All manuscript figures are included at full linewidth to preserve this scale.
FIGURE_WIDTH_IN = 390 / 72.27
FONT_FAMILY = "Liberation Sans"
BODY_PT = 9
TITLE_PT = 10
STAGE_PT = 11


@contextmanager
def journal_style():
    # Fail clearly instead of silently exporting a different figure font.
    font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    with mpl.rc_context({
        "font.family": FONT_FAMILY,
        "font.size": BODY_PT,
        "axes.labelsize": BODY_PT,
        "axes.titlesize": TITLE_PT,
        "axes.titleweight": "bold",
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "eaai-energy-figures-2026",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }):
        yield


def save_figure(fig, destination: Path):
    """Preserve the declared physical size rather than crop and rescale text."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(destination.with_suffix(suffix), dpi=400,
                    metadata={"Creator": "Energy forecasting journal figure scripts"}
                    if suffix in (".pdf", ".svg") else None)
    return destination.with_suffix(".pdf")
