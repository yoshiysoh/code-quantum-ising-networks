"""Figure sizes, panel letters and saving, following Nature's figure guide:
full width 183 mm, text 5-7 pt, lines at least 0.5 pt, panel letters 8 pt
bold lowercase. Importing this module applies these sizes on top of
matplotlibrc."""

import io
import json
import string
from pathlib import Path

import matplotlib as mpl
import numpy as np
from lxml import etree
from matplotlib.text import Text

COL1_WIDTH = 89 / 25.4  # inches
MAX_FIG_WIDTH = 183 / 25.4  # inches, full width
MAX_FIG_HEIGHT = 170 / 25.4  # inches
LABEL_FONT = 8.0  # pt
LABEL_BOLD = True
MIN_LINE_PT = 0.5

mpl.rcParams["lines.linewidth"] = max(
    mpl.rcParams["lines.linewidth"], MIN_LINE_PT
)
mpl.rcParams["axes.linewidth"] = max(
    mpl.rcParams["axes.linewidth"], MIN_LINE_PT
)
mpl.rcParams["font.size"] = 7
mpl.rcParams["figure.figsize"] = (COL1_WIDTH, COL1_WIDTH * 2.1 / 3.4)
NO_DATE = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}


def journal_labels(letters: str) -> list[str]:
    """Panel letters as Nature prints them: lowercase, no parentheses."""
    return [c.lower() for c in letters]


# ── Panel-letter y-axis alignment ─────────────────────────────────────────
def align_panel_labels_to_yaxis(
    fig,
    axes,
    labels,
    *,
    y: float = 1.04,
    fontsize: float | None = None,
    bold: bool | None = None,
    freeze_layout: bool = True,
):
    """Place panel-letter labels flush with each axis's y-axis.

    `labels` is any sequence of strings, such as a plain str or the list
    from journal_labels().

    fontsize/bold default to LABEL_FONT/LABEL_BOLD.

    Creates a Text artist at axes-fraction (0, y) for each axis, forces a
    draw to get a renderer, then snaps each label's x-position to the left
    edge of that axis's y-axis tight bbox. If freeze_layout, forces a
    second draw and disables constrained_layout afterward so the final
    savefig() can't re-shuffle axis positions and undo the alignment.
    """
    fontsize = LABEL_FONT if fontsize is None else fontsize
    bold = LABEL_BOLD if bold is None else bold
    axes_flat = list(np.ravel(axes))
    panel_artists = []
    for ax, label in zip(axes_flat, labels, strict=True):
        pt = ax.text(
            0,
            y,
            label,
            transform=ax.transAxes,
            fontfamily="Helvetica",
            fontweight="bold" if bold else "normal",
            fontsize=fontsize,
            va="bottom",
            ha="left",
        )
        panel_artists.append(pt)

    fig.savefig(io.BytesIO(), format="png")
    renderer = fig.canvas.get_renderer()
    for ax, pt in zip(axes_flat, panel_artists, strict=True):
        yax_bb = ax.yaxis.get_tightbbox(renderer)
        if yax_bb is None:
            yax_bb = ax.get_tightbbox(renderer)
        ax_win = ax.get_window_extent(renderer)
        pt.set_position(((yax_bb.x0 - ax_win.x0) / ax_win.width, y))

    if freeze_layout:
        fig.savefig(io.BytesIO(), format="png")
        fig.set_constrained_layout(False)

    return panel_artists


# ── Save ────────────────────────────────────────────────────────────────
def save_figure(
    fig,
    out_dir: Path,
    name: str,
    formats: tuple[str, ...] = ("pdf", "svg"),
    png_dpi: int = 600,
    bbox_inches: str | None = None,
) -> dict[str, Path]:
    """Save `fig` as out_dir/name.<fmt> for each requested format.

    PNG gets a white background; PDF/SVG stay transparent
    (savefig.transparent in matplotlibrc). No creation date is written, so
    the same figure gives the same file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        if fmt == "png":
            fig.savefig(
                path, dpi=png_dpi, bbox_inches=bbox_inches, facecolor="white"
            )
        else:
            fig.savefig(
                path, bbox_inches=bbox_inches, metadata=NO_DATE.get(fmt)
            )
        paths[fmt] = path
    return paths


def save_unlabeled(
    fig,
    out_dir: Path,
    name: str,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    png_dpi: int = 600,
    bbox_inches: str | None = None,
) -> dict[str, Path]:
    """Save `fig` without its panel labels and record where they were.

    Panel labels are the Text artists whose string is one lowercase
    letter and whose size and weight are LABEL_FONT/LABEL_BOLD. They
    are made transparent (so the layout is unchanged), the figure is
    saved as out_dir/name.<fmt>, and each label's baseline anchor is
    written to out_dir/name.labels.json in SVG points (origin top-left).
    The letters of the published figures are drawn at these anchors.
    Transparent labels still count toward bbox_inches="tight", so the
    saved canvas keeps the labelled extent.
    Returns the saved paths, as save_figure.
    """
    to_letter = dict(
        zip(journal_labels(string.ascii_lowercase), string.ascii_lowercase)
    )
    fig.draw_without_rendering()  # settles layout on any canvas type

    def is_label(t):
        weight = t.get_fontweight()
        bold = weight == "bold" or (isinstance(weight, int) and weight >= 700)
        return (
            t.get_visible()
            and t.get_text() in to_letter
            and abs(t.get_fontsize() - LABEL_FONT) < 1e-6
            and bold == LABEL_BOLD
        )

    labels = [t for t in fig.findobj(Text) if is_label(t)]
    # Label anchors from matplotlib's own SVG (<text> at the baseline).
    for i, t in enumerate(labels):
        t.set_gid(f"panel-label-{i}")
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches=bbox_inches)
    tree = etree.fromstring(buf.getvalue())
    anchors = []
    for i, t in enumerate(labels):
        (el,) = tree.xpath(
            f"//*[@id='panel-label-{i}']//*[local-name()='text']"
        )
        anchors.append(
            {
                "letter": to_letter[t.get_text()],
                "x": float(el.get("x")),
                "y": float(el.get("y")),
            }
        )
        t.set_gid(None)
    colors = [t.get_color() for t in labels]
    for t in labels:
        t.set_color("none")
    try:
        paths = save_figure(
            fig, out_dir, name, formats, png_dpi, bbox_inches=bbox_inches
        )
    finally:
        for t, c in zip(labels, colors):
            t.set_color(c)
    anchors.sort(key=lambda a: a["letter"])
    path = Path(out_dir) / f"{name}.labels.json"
    path.write_text(json.dumps(anchors, indent=2))
    print(
        f"{name}: unlabeled canvas, labels "
        + "".join(a["letter"] for a in anchors)
    )
    return paths
