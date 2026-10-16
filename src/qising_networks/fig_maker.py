"""Generic, project-agnostic figure-making infrastructure.

Import as `import fig_maker as fm`. Covers SVG panel assembly/compositing,
panel-label placement, journal-specific sizing/styling, and figure saving --
none of it tied to any one paper's science.

This file is the canonical template: it lives at `~/.config/fig_maker.py`
and gets copied (together with the paired `~/.config/matplotlibrc`) into
each new matplotlib+SVG-compositing paper repo as-is. The only thing a new
project needs to edit is `JOURNAL` below (one line) to select which of
`JOURNALS`' dimension/font/panel-label conventions apply.

Paper-specific content (task names, toy-model physics, palettes, etc.)
belongs in a project's own
`fig_maker_helper.py`, which should import this module rather than the
other way around.

Project-wide matplotlib *house style* (spine visibility, tick sizes,
constrained_layout, legend spacing — everything NOT constrained by a
journal's own requirements) lives in the paired `matplotlibrc` and is
intentionally not touched here. What IS journal-dependent (font.size,
line weights, default figure.figsize) is patched onto whatever
`matplotlibrc` a project has loaded by `set_journal()` below, every time
`JOURNAL` changes.
"""

import io
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import cairosvg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from matplotlib.text import Text

# ── Journal specs ──────────────────────────────────────────────────────────

# Figure lettering font, the same for every journal.
FONT_CANDIDATES = ("Helvetica", "Arial")

# rcParams before any set_journal() call.
_BASE_LINEWIDTH = mpl.rcParams["lines.linewidth"]
_BASE_AXES_LINEWIDTH = mpl.rcParams["axes.linewidth"]


@dataclass(frozen=True)
class JournalSpec:
    """One journal's figure dimension/font/panel-label conventions."""

    col1_mm: float
    col2_mm: float
    label_case: str  # "lower" or "upper"
    label_paren: bool  # wrap each panel letter as "(a)"
    label_bold: bool
    label_fontsize_pt: float
    fontsize_range_pt: tuple[float, float]  # (min, max) body/axis text
    min_line_pt: float
    min_dpi_raster: int
    vector_formats: tuple[str, ...]
    col1_5_mm: float | None = None
    max_height_mm: float | None = None
    label_italic: bool = False  # italicize the panel letter itself (RSIF)
    notes: tuple[str, ...] = field(default_factory=tuple)


JOURNALS: dict[str, JournalSpec] = {
    "nature": JournalSpec(
        col1_mm=89,
        col2_mm=183,
        max_height_mm=170,
        fontsize_range_pt=(5, 7),
        label_case="lower",
        label_paren=False,
        label_bold=True,
        label_fontsize_pt=8.0,
        min_line_pt=0.5,
        min_dpi_raster=300,
        vector_formats=("pdf", "eps"),
        notes=(
            "🟢 Panel-label case/bold/size/no-parens: 'Separate panels in "
            "multi-panelled figures should be labelled with 8-pt bold, "
            "upright (not italic) and lowercase a, b, c, etc.' "
            "(research-figure-guide.nature.com).",
            "🟢 Color: RGB for original-research Articles (this journal's "
            "target); CMYK applies only to Nature Reviews-style content.",
            "Main figures must be vector (no raster) except Extended Data, "
            "which is raster-capped at 300dpi.",
            "Max height is a conservative flat default; the official guide "
            "gives a caption-length-dependent table up to 225mm for short "
            "captions.",
            "Avoid red/green together and rainbow colormaps (accessibility).",
        ),
    ),
    "science": JournalSpec(
        col1_mm=57,
        col1_5_mm=121,
        col2_mm=184,
        fontsize_range_pt=(5, 7),
        label_case="upper",
        label_paren=False,
        label_bold=True,
        label_fontsize_pt=10.0,
        min_line_pt=0.5,
        min_dpi_raster=300,
        vector_formats=("ai", "eps", "pdf"),
        notes=(
            "🟢 Panel labels: 'Use capital letters for part labels in "
            "multipart figures ... 10 pt and bold in the final figure.'",
            "🟢 Font: 'Use a sans-serif font whenever possible (Helvetica "
            "is preferred)' -- this governs FIGURE lettering only; Science's "
            "separate 'use Times New Roman' instruction is for the "
            "manuscript's own body text (.docx), not figure labels.",
            "Two official Science documents disagree on min line weight "
            "(0.28pt vs 0.5pt) -- using the larger, safer value.",
            "Column widths are Science's 'final printed artwork' numbers "
            "(5.7/12.1/18.4cm), not the larger submission-canvas box from "
            "the separate figure-prep design guide (9/18.3cm, which "
            "includes caption space).",
        ),
    ),
    "pnas": JournalSpec(
        col1_mm=87,
        col1_5_mm=114,
        col2_mm=178,
        max_height_mm=225,
        fontsize_range_pt=(6, 8),  # hard limits are 6-12pt
        label_case="upper",
        label_paren=False,
        label_bold=True,
        label_fontsize_pt=8.0,
        min_line_pt=0.25,
        min_dpi_raster=300,  # halftone/photo floor; line art wants 1000-1200
        vector_formats=("eps", "ai", "pdf"),
        notes=(
            "🔵 Panel-label case/bold/parens are a DOCUMENTED DEFAULT, not "
            "a quoted PNAS rule -- both of PNAS's figure-guidance documents "
            "(Digital Art Guidelines PDF, Submitting-your-Manuscript page) "
            "were read in full and neither states a panel-letter "
            "case/weight/punctuation convention. Uppercase chosen to match "
            "Science; edit label_case/label_bold/label_paren above to "
            "override.",
            "🟢 Font: Arial/Helvetica/Times/Symbol all explicitly listed as "
            "acceptable -- NOT sans-serif-exclusive; Helvetica kept here "
            "only for consistency with the other three presets.",
            "🟢 Raster DPI is tiered: 300ppi halftone/photo, 600-900ppi "
            "combination (halftone+line art), 1000-1200ppi pure line art.",
        ),
    ),
    "aps": JournalSpec(
        col1_mm=85,
        col2_mm=170,  # 🔵 derived as 2x col1_mm, see note
        fontsize_range_pt=(6, 7),  # min from ">=2mm lettering height"
        label_case="lower",
        label_paren=True,
        label_bold=False,
        label_fontsize_pt=8.0,
        min_line_pt=0.5,
        min_dpi_raster=600,
        vector_formats=("eps", "ps"),
        notes=(
            "🟢 Panel labels: 'Label subfigures (a), (b), etc.' -- lower, "
            "parenthesized, confirmed in 5 independent current APS "
            "documents (both Style Guide editions, Style Basics, PRE's and "
            "PRL's own Information for Contributors pages).",
            "🟢/🔵 'Not bold' is an inference, not a literal quote: none of "
            "those 5 documents ever says 'bold' for panel labels, though "
            "the same guide explicitly says 'Use bold font for commands' "
            "elsewhere (algorithms) -- i.e. it clearly would say so if it "
            "meant it.",
            "🔵 col2_mm is DERIVED (2 x col1_mm), not quoted -- every "
            "current official APS document states the 1-column width "
            "(8.5cm) and says 'use 1.5 or 2 columns for more detailed "
            "figures' with no cm/in number ever attached, across 5 "
            "independent sources checked.",
            "🟢 Raster/vector file formats confirmed directly from the "
            "current 'Style Basics' page: .ps/.eps preferred, .jpg/.png "
            "for photos.",
        ),
    ),
    "rsif": JournalSpec(
        col1_mm=84,
        col2_mm=175,
        max_height_mm=250,
        fontsize_range_pt=(7.5, 9),
        label_case="lower",
        label_paren=True,
        label_bold=False,
        label_italic=True,
        label_fontsize_pt=9.0,
        min_line_pt=0.5,
        min_dpi_raster=300,
        vector_formats=("eps", "ps"),
        notes=(
            "🟢 Column widths/max height (Proc B, Trans B, Biol Lett, Interface "
            "share these): single-column 84mm, double-column 175mm, max height "
            "250mm (Royal Society Publishing style guide, figures page).",
            "🟢 Font size: 'Times, upright, 9pt, 11pt leading (unless restricted "
            "for space)'; minimum 7.5pt if reduced. NOTE: the style guide's "
            "literal text specifies serif Times, but published JRSI articles "
            "(checked directly) actually use sans-serif figure lettering in "
            "practice -- so font FAMILY here deliberately follows "
            "FONT_CANDIDATES (Helvetica/Arial) like the other four journals, "
            "not the guide's literal 'Times'; only the size numbers are taken "
            "from this quote.",
            "🟢 Panel labels: 'Always use lower-case lettering, except where "
            "capitals are essential' and alphabetical labels are 'italicised in "
            "roman brackets, e.g. (a)' -- the letter itself is italic, the "
            "parentheses are upright/roman.",
            "🔵 label_bold=False is a documented default, not a literal quote: "
            "neither official document ever says 'bold' for panel labels (same "
            "absence-of-mention pattern as the APS entry above).",
            "🔵 label_fontsize_pt=9.0 is DERIVED from the general 'figure text' "
            "spec (9pt/11pt leading), not a distinct quoted panel-label size -- "
            "unlike Nature/Science/PNAS/APS, no separate larger label size is "
            "ever given for RSIF.",
            "🟢 Min line weight: 'Lines should be at least 0.5pt in width' "
            "(style guide, figures page).",
            "🟢 Accepted formats/resolution (royalsociety.org author guidelines): "
            "'Accepted figure formats: PNG, EPS, TIFF, JPG, JPEG, TeX/LaTeX-coded "
            "figures should be converted to postscript format (PS or EPS)'; "
            "'Figures should be uploaded as separate files and must be at least "
            "300 DPI' -- flat 300 DPI, not tiered by content type like PNAS. No "
            "PDF/AI in the accepted vector list (checked twice against exact "
            "wording) -- unlike the other four journals here, whose vector list "
            "includes pdf.",
            "🔵 Colour mode (RGB/CMYK) confirmed absent from both documents "
            "checked -- left unmodeled, same as every other entry here.",
            "J. R. Soc. Interface itself has no colour charges even in print "
            "('does not have page or colour charges', RSIF's own for-authors "
            "page) -- more generous than the general Society-wide policy of "
            "free-online/charged-in-print colour that the style guide states for "
            "its sibling journals in the same column-width group.",
            "Evidence-source caveat: the style-guide citations above come from "
            "Royal Society Publishing's own style guide, hosted on Google Sites "
            "rather than royalsocietypublishing.org -- it is publicly linked "
            "editorial-production documentation, not a random third party, but "
            "noted here since it's a different domain than the other journals' "
            "🟢 sources in this file.",
        ),
    ),
}
JOURNALS["prl"] = JOURNALS["aps"]  # alias: same spec across PRL/PRB/PRE/PRX

JOURNAL = "nature"  # <- the one line a new project edits


def set_journal(name: str) -> JournalSpec:
    """Select `name`'s conventions from JOURNALS as the active journal.

    Sets module globals (JOURNAL, COL1_WIDTH, COL1_5_WIDTH, COL2_WIDTH,
    MAX_FIG_WIDTH, MAX_FIG_HEIGHT, LABEL_FONT, LABEL_BOLD, LABEL_ITALIC) and
    patches the
    journal-dependent rcParams (font.size, lines.linewidth, axes.linewidth,
    figure.figsize) onto whatever `matplotlibrc` a project has loaded.
    Everything else in matplotlibrc (spines, tick sizes, constrained_layout,
    ...) is untouched -- those are house-style choices, not journal
    requirements. font.size is set directly to the journal's maximum
    allowed body/axis text size (spec.fontsize_range_pt[1]). Safe to call
    repeatedly in one session: line weight is bumped against the fixed
    pre-any-journal baseline (_BASE_LINEWIDTH/_BASE_AXES_LINEWIDTH), never
    against whatever a previous call left behind, so switching journals
    back and forth doesn't ratchet values upward.
    """
    global JOURNAL, COL1_WIDTH, COL1_5_WIDTH, COL2_WIDTH
    global MAX_FIG_WIDTH, MAX_FIG_HEIGHT, LABEL_FONT, LABEL_BOLD, LABEL_ITALIC

    spec = JOURNALS[name]
    JOURNAL = name

    COL1_WIDTH = spec.col1_mm / 25.4
    COL1_5_WIDTH = spec.col1_5_mm / 25.4 if spec.col1_5_mm else None
    COL2_WIDTH = spec.col2_mm / 25.4
    MAX_FIG_WIDTH = COL2_WIDTH  # inches, full-width multi-panel figure
    MAX_FIG_HEIGHT = (spec.max_height_mm or 170) / 25.4  # inches
    LABEL_FONT = spec.label_fontsize_pt
    LABEL_BOLD = spec.label_bold
    LABEL_ITALIC = spec.label_italic

    mpl.rcParams["lines.linewidth"] = max(_BASE_LINEWIDTH, spec.min_line_pt)
    mpl.rcParams["axes.linewidth"] = max(_BASE_AXES_LINEWIDTH, spec.min_line_pt)
    mpl.rcParams["font.size"] = spec.fontsize_range_pt[1]
    mpl.rcParams["figure.figsize"] = (COL1_WIDTH, COL1_WIDTH * 2.1 / 3.4)

    return spec


def journal_labels(letters: str) -> list[str]:
    """Format literal panel letters per the current JOURNAL's convention.

    E.g. "abcdef" -> ["A", "B", ...] for Science/PNAS (uppercase, no
    parens), -> ["(a)", "(b)", ...] for APS (lowercase, parens), or
    unchanged for Nature (lowercase, no parens). Returns a list[str] rather
    than a str specifically so the parenthesized APS case still works with
    align_panel_labels_to_yaxis's zip(axes, labels) below -- a plain str is
    just a sequence of 1-char strings, and a list[str] of multi-char
    strings zips identically, with no signature change needed there.
    """
    spec = JOURNALS[JOURNAL]
    cased = [c.upper() if spec.label_case == "upper" else c.lower() for c in letters]
    return [f"({c})" for c in cased] if spec.label_paren else cased


# Sets COL1_WIDTH, COL2_WIDTH, MAX_FIG_WIDTH, LABEL_FONT and the like.
set_journal(JOURNAL)

_SVG_NS = "http://www.w3.org/2000/svg"
_SVG_NSMAP = {None: _SVG_NS, "xlink": "http://www.w3.org/1999/xlink"}


# ── SVG panel assembly ────────────────────────────────────────────────────


def svg_panel_dims(path: Path) -> tuple[float, float]:
    """Parse (width_pt, height_pt) from an SVG file's root <svg> tag."""
    text = Path(path).read_text()
    m = re.search(
        r'<svg[^>]*width=["\']([0-9.]+)pt["\'][^>]*height=["\']([0-9.]+)', text
    )
    if not m:
        raise ValueError(f"Cannot parse dims of {path}")
    return float(m.group(1)), float(m.group(2))


def add_panel_label(
    parent,
    x: float,
    y: float,
    text: str,
    fontsize: float | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
) -> None:
    """Append a Helvetica <text> panel-letter node to an lxml SVG tree.

    fontsize/bold/italic default to the current LABEL_FONT/LABEL_BOLD/
    LABEL_ITALIC (read lazily at call time, not snapshotted at import) so
    this responds correctly to set_journal() even if called after switching
    journals mid-session.
    """
    fontsize = LABEL_FONT if fontsize is None else fontsize
    bold = LABEL_BOLD if bold is None else bold
    italic = LABEL_ITALIC if italic is None else italic
    el = etree.SubElement(parent, f"{{{_SVG_NS}}}text")
    el.set("x", f"{x:.3f}")
    el.set("y", f"{y:.3f}")
    el.set("font-family", "Helvetica")
    el.set("font-weight", "bold" if bold else "normal")
    el.set("font-style", "italic" if italic else "normal")
    el.set("font-size", f"{fontsize:.6g}")
    el.text = text


def embed_svg_panel(
    parent,
    path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
) -> None:
    """Embed a (possibly cropped) SVG panel as a nested <svg> at (x, y, w, h).

    crop_top/crop_bottom trim constrained_layout whitespace via a viewBox
    shift; both default to 0.0 so panels can be embedded at natural size.
    """
    sr = etree.parse(str(path)).getroot()
    ow = sr.get("width", "").replace("pt", "").strip()
    oh = sr.get("height", "").replace("pt", "").strip()
    vb = sr.get("viewBox", f"0 0 {ow} {oh}")
    vb_x, vb_y, vb_w, vb_h = map(float, vb.split())
    g = etree.SubElement(parent, f"{{{_SVG_NS}}}svg")
    g.set("x", f"{x:.4f}")
    g.set("y", f"{y:.4f}")
    g.set("width", f"{w:.4f}")
    g.set("height", f"{h - crop_top - crop_bottom:.4f}")
    g.set(
        "viewBox",
        f"{vb_x} {vb_y + crop_top} {vb_w} {vb_h - crop_top - crop_bottom}",
    )
    for child in sr:
        g.append(child)


def new_svg_root(width_pt: float, height_pt: float):
    """Create a fresh lxml SVG root element sized (width_pt, height_pt)."""
    root = etree.Element(f"{{{_SVG_NS}}}svg", nsmap=_SVG_NSMAP)
    root.set("width", f"{width_pt:.3f}pt")
    root.set("height", f"{height_pt:.3f}pt")
    root.set("viewBox", f"0 0 {width_pt:.3f} {height_pt:.3f}")
    root.set("version", "1.1")
    return root


def write_composite_svg(
    root,
    out_svg: Path,
    out_pdf: Path | None = None,
    out_png: Path | None = None,
    png_dpi: int = 600,
) -> None:
    """Write an assembled SVG tree to disk, optionally rendering PDF/PNG."""
    etree.ElementTree(root).write(
        str(out_svg), pretty_print=True, xml_declaration=True, encoding="utf-8"
    )
    if out_pdf is not None:
        cairosvg.svg2pdf(url=str(out_svg), write_to=str(out_pdf))
    if out_png is not None:
        cairosvg.svg2png(
            url=str(out_svg),
            write_to=str(out_png),
            dpi=png_dpi,
            background_color="white",
        )


def measure_label_glyph_heights(
    labels: str,
    fontsize: float | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
) -> dict[str, float]:
    """Measure Helvetica glyph height (pt) for each char in `labels`.

    fontsize/bold/italic default to the current LABEL_FONT/LABEL_BOLD/
    LABEL_ITALIC (read lazily at call time, not snapshotted at import) so
    this responds correctly to set_journal() even if called after switching
    journals mid-session.
    """
    fontsize = LABEL_FONT if fontsize is None else fontsize
    bold = LABEL_BOLD if bold is None else bold
    italic = LABEL_ITALIC if italic is None else italic
    fig_m, ax_m = plt.subplots(figsize=(1, 1))
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)
    fig_m.canvas.draw()
    renderer = fig_m.canvas.get_renderer()
    heights = {}
    for ch in labels:
        t = ax_m.text(
            0.5,
            0.5,
            ch,
            fontfamily="Helvetica",
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
            fontsize=fontsize,
            transform=ax_m.transData,
        )
        fig_m.canvas.draw()
        heights[ch] = t.get_window_extent(renderer).height / fig_m.dpi * 72
        t.remove()
    plt.close(fig_m)
    return heights


# ── Panel-letter y-axis alignment ─────────────────────────────────────────
def align_panel_labels_to_yaxis(
    fig,
    axes,
    labels,
    *,
    y: float = 1.04,
    fontsize: float | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    freeze_layout: bool = True,
):
    """Place panel-letter labels flush with each axis's y-axis.

    `labels` is any sequence of strings (a plain str works -- it's a
    sequence of 1-char strings -- as does a list[str] from journal_labels()
    for journals like APS whose convention is multi-char, e.g. "(a)").

    fontsize/bold/italic default to the current LABEL_FONT/LABEL_BOLD/
    LABEL_ITALIC (read lazily at call time, not snapshotted at import) so
    this responds correctly to set_journal() even if called after switching
    journals mid-session.

    Creates a Text artist at axes-fraction (0, y) for each axis, forces a
    draw to get a renderer, then snaps each label's x-position to the left
    edge of that axis's y-axis tight bbox. If freeze_layout, forces a
    second draw and disables constrained_layout afterward so the final
    savefig() can't re-shuffle axis positions and undo the alignment.
    """
    fontsize = LABEL_FONT if fontsize is None else fontsize
    bold = LABEL_BOLD if bold is None else bold
    italic = LABEL_ITALIC if italic is None else italic
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
            fontstyle="italic" if italic else "normal",
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

    PNG always gets facecolor="white" (visible on dark chat backgrounds);
    PDF/SVG are left transparent per matplotlibrc's savefig.transparent.
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
            fig.savefig(path, bbox_inches=bbox_inches)
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

    Panel labels are the Text artists whose string is a journal label of
    one letter and whose size and weight are LABEL_FONT/LABEL_BOLD. They
    are made transparent (so the layout is unchanged), the figure is
    saved as out_dir/name.<fmt>, and each label's baseline anchor is
    written to out_dir/name.labels.json in SVG points (origin top-left).
    A separate assembly step draws the journal's letters at these anchors,
    so layout and lettering can change without rerunning the notebook.
    Transparent labels still count toward bbox_inches="tight", so the
    saved canvas keeps the labelled extent.
    Returns the saved paths, as save_figure.
    """
    to_letter = dict(zip(journal_labels(string.ascii_lowercase),
                         string.ascii_lowercase))
    fig.draw_without_rendering()  # settles layout on any canvas type

    def is_label(t):
        weight = t.get_fontweight()
        bold = weight == "bold" or (isinstance(weight, int) and weight >= 700)
        return (t.get_visible() and t.get_text() in to_letter
                and abs(t.get_fontsize() - LABEL_FONT) < 1e-6
                and bold == LABEL_BOLD)

    labels = [t for t in fig.findobj(Text) if is_label(t)]
    # Label anchors from matplotlib's own SVG (<text> at the baseline).
    for i, t in enumerate(labels):
        t.set_gid(f"panel-label-{i}")
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches=bbox_inches)
    tree = etree.fromstring(buf.getvalue())
    anchors = []
    for i, t in enumerate(labels):
        (el,) = tree.xpath(f"//*[@id='panel-label-{i}']//*[local-name()='text']")
        anchors.append({"letter": to_letter[t.get_text()],
                        "x": float(el.get("x")), "y": float(el.get("y"))})
        t.set_gid(None)
    colors = [t.get_color() for t in labels]
    for t in labels:
        t.set_color("none")
    try:
        paths = save_figure(fig, out_dir, name, formats, png_dpi,
                            bbox_inches=bbox_inches)
    finally:
        for t, c in zip(labels, colors):
            t.set_color(c)
    anchors.sort(key=lambda a: a["letter"])
    path = Path(out_dir) / f"{name}.labels.json"
    path.write_text(json.dumps(anchors, indent=2))
    print(f"{name}: unlabeled canvas, labels "
          + "".join(a["letter"] for a in anchors))
    return paths
