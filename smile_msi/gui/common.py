"""Shared GUI helpers, constants, and background-worker functions."""
from __future__ import annotations

import os
import re
import tempfile

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

# Display ion images the way numpy/matplotlib do: array[row, col] → view
# (x=col, y=row), so row 0 sits at the top (with the views' invertY) and the picture
# matches our matplotlib export. pyqtgraph defaults to 'col-major', which
# would transpose every tissue image. Every spatial view (ion/seg/TIC/components/…) feeds
# row-major (height, width) arrays, so this makes them all correct at once; the two
# genuinely non-spatial heatmaps (the cohort sample×m/z grid and the co-localization
# correlation matrix) pin themselves back to 'col-major'. Set at import — before any
# ImageItem is constructed — so the app and the test suite share the same convention.
pg.setConfigOption("imageAxisOrder", "row-major")

from .. import (spatial, multivariate, profiles, palettes)
from ..msi import MSIDataset
from ..spectrum_range import signal_mz_range  # re-exported: shared with the export engine
from . import icons                            # canonical action→icon vocabulary
from .icons import icon, glyph                 # re-exported: build icons by action name


# Categorical palettes now live in the engine-side, GUI-agnostic ``smile_msi.palettes`` (one
# source shared with the export engine + the colour picker); re-exported here so the many
# ``common.PALETTE`` / ``common.REGION_PALETTE`` callers keep working unchanged.
PALETTE = list(palettes.SEGMENT)
# distinct colors for per-pixel spectra overlaid by clicking the ion image

PIXEL_COLORS = ["#DD8452", "#E15759", "#B07AA1", "#59A14F", "#EDC948", "#FF9DA7",
                "#9C755F", "#F28E2B", "#76B7B2", "#4E79A7"]
# saturated colors for user-defined regions (distinct from the segment PALETTE)

REGION_PALETTE = list(palettes.REGION)

# Semantic A/B comparison colors — the single source of truth for "higher in A vs B"
# across the comparison tab, the volcano, and the Excel report (orange = A, blue = B).
# Centralized here so all views stay consistent; future work makes these user-editable.
REGION_A_COLOR = "#DD8452"   # orange — higher in region/group A
REGION_B_COLOR = "#4C72B0"   # blue   — higher in region/group B
AB_COLORS = {"A": REGION_A_COLOR, "B": REGION_B_COLOR}

# Default ROI-drawing color (outline, freehand fill, band, and the ROI spectrum trace).
# Vivid magenta sits outside the viridis/inferno/plasma ranges (which go yellow at high
# intensity), so the outline — and gaps in a freehand draw — stay legible over any ion
# image. Centralized here and user-overridable per window (see _pick_roi_color).
ROI_COLOR = "#FF2BD6"

# Plot accents + guide lines — the single source of truth for marks every spectrum,
# volcano, and scatter view used to re-hardcode. ACCENT is the green selection cursor /
# hover pen + the app's identity colour (it drives the Highlight role and the primary
# button in BOTH light and dark themes, so selections read the same everywhere);
# GUIDE_LINE is the faint zero / p=0.05 / x=0 rule; HIDDEN_FG dims hidden rows in the
# feature + segment trees. Centralizing them means a single edit re-themes every view at
# once (no more per-screen colour drift, and no more "blue in light, green in dark" split).
ACCENT = "#4FA56B"          # selection cursor, hover pen, "this ion" marker, Highlight + primary button
ACCENT_HOVER = "#5CB87B"    # primary button :hover (a touch lighter than ACCENT)
ACCENT_PRESSED = "#418F5C"  # primary button :pressed (a touch darker than ACCENT)
DANGER = "#C44E52"          # destructive verbs (Delete/Reset/Wipe), failed-step rows — reads on light + dark
GUIDE_LINE = "#888888"      # zero line, significance threshold, faint axis guide
HIDDEN_FG = "#888888"       # dimmed text for hidden features / clusters
HILITE = "#FFC857"          # amber: picked dendrogram branches + their pixels on the seg map

# Tissue-image canvases (the ion image, segmentation map, component score maps, montage,
# coloc/joint-seg maps, …) always paint on this fixed near-black charcoal *regardless of
# the app theme*. MSI colormaps (viridis/inferno/magma) are calibrated to read against
# black — the SCiLS convention — an optical/composite backdrop sits behind them, and a
# white canvas in light mode would wash low-intensity pixels into the page. So these views
# are deliberately DECOUPLED from palette(Base); pin them with :func:`dark_image_view` and
# :func:`retheme_open_plots` leaves them dark on a live Light/Dark switch.
ION_CANVAS_BG = "#0E1013"


def hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple:
    """``"#RRGGBB"`` → ``(r, g, b, alpha)``. Lets translucent fills (e.g. the
    difference-spectrum bands) derive from the centralized A/B colors instead of
    re-hardcoding the RGB, so a future color change propagates everywhere."""
    h = str(hex_color).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, int(alpha))


def guarded(fn, *args, **kwargs):
    """Call ``fn(*args, **kwargs)``, swallowing any exception and returning ``None`` on
    failure. For *best-effort UI refreshes* where one dead step must never wedge a broader
    sweep — restoring an undo state, reorienting every view after a rotation, re-evaluating
    each tab's button state — and a view that isn't built yet (or a ``None`` dataset) would
    otherwise raise. Shared so these refresh loops don't each re-roll a local ``try/except``.
    Do NOT use it to hide errors in a single deliberate action; there the caller should see
    the failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — best-effort refresh; a not-yet-built view is non-fatal
        return None

# Dimmed text for hints / secondary labels. MUTED_QSS drives it off the *palette's*
# PlaceholderText role (not a fixed grey) so it tracks the Light/Dark toggle and stays
# legible in both: a fixed #9a9a9a read fine on the dark charcoal but washed out to ~2.4:1
# on the light theme (unreadable). Safe now that the app forces Fusion + an explicit palette
# (set_theme), so palette() roles resolve to OUR values, not the platform's — the old
# cross-platform palette(mid) hazard this constant used to dodge no longer applies. MUTED_FG
# stays a concrete hex for the QColor/QBrush + baked-QSS (disabled-state) uses, where a
# stylesheet palette() reference can't reach.
MUTED_FG = "#9a9a9a"
MUTED_QSS = "color: palette(placeholder-text);"

def load_demo(progress=None):
    from .. import demo
    ds = demo.make_synthetic()
    ds.prime(progress=progress)
    return ds

def load_imzml(path, stride=1, progress=None):
    ds = MSIDataset.from_imzml(path, lazy=True, stride=stride)
    ds.prime(progress=progress)
    return ds

def pick_and_build(ds, snr, min_rel, tol_ppm, reduce, projection="mean", mask=None,
                   progress=None, prominence=1.0, max_peaks=0):
    # mask restricts peak DETECTION to a region's pixels (a per-sample feature list);
    # features are still integrated over every pixel so the ion images span the slide.
    # max_peaks=0 means no cap: a fixed top-N would make two regions with different
    # numbers of detectable ions report the same count.
    peaks = ds.pick_peaks(snr=snr, min_rel_intensity=min_rel, projection=projection,
                          mask=mask, prominence=prominence, max_peaks=max_peaks)
    mzs = [p["mz"] for p in peaks]
    # Extraction caches one float32 per pixel per feature, and every feature_matrix consumer
    # (segmentation, PCA/NMF, co-localization) then takes a float64 copy — 3x this figure. A
    # noisy slide picked with no cap can ask for more than the machine has, so report the
    # size rather than discover it as a MemoryError three steps later.
    peaks.cache_gb = ds.n_pixels * len(mzs) * 4 / 1e9
    if mzs:
        ds.ensure_features(mzs, tol_ppm=tol_ppm, reduce=reduce, progress=progress)
    return peaks

def find_spatial(ds, snr, min_rel, tol_ppm, reduce, min_frequency, min_morans,
                 norm="tic", mask=None, progress=None, prominence=1.0,
                 projection="mean", rescue_frequency=None, collapse_isotopes=False,
                 fdr_max=None, mode="negative", fdr_ppm=5.0, max_candidates=2000):
    """Worker for the spatially-aware feature finder: mean candidates →
    frequency gate → spatial-denoise gate → auto-width → (opt) isotope collapse →
    (opt) FDR threshold. Returns the engine's
    :class:`~smile_msi.spatial.SpatialFeatures` result (``.peaks`` + funnel counts);
    features for the survivors are already cached for instant ion images.

    With ``fdr_max`` set, every surviving feature is annotated and given a target–decoy
    q-value (:func:`annotate.attach_fdr`) and the list is thresholded at that FDR — the
    publication-grade alternative to an intensity cutoff. Unannotated 'unknown' ions are
    kept regardless, and the per-FDR-level ID counts land in ``res.params['fdr']``."""
    res = spatial.find_spatial_features(
        ds, snr=snr, min_rel_intensity=min_rel, tol_ppm=tol_ppm, reduce=reduce,
        min_frequency=min_frequency, min_morans=min_morans, norm=norm, mask=mask,
        max_candidates=max_candidates, prominence=prominence, projection=projection,
        rescue_frequency=rescue_frequency, collapse_isotopes=collapse_isotopes,
        progress=progress)
    if fdr_max is not None and res.peaks:
        from .. import annotate
        res.peaks, res.params["fdr"] = annotate.attach_fdr(
            ds, res.peaks, mode=mode, ppm=fdr_ppm, norm=norm, q_max=fdr_max)
    if res.peaks:
        ds.ensure_features([p["mz"] for p in res.peaks], tol_ppm=tol_ppm, reduce=reduce)
    return res

def run_segment(ds, mzs, k, auto, spatial_aware, tol_ppm, norm):
    seed = profiles.active_seed()                  # active-profile seed → reproducible clusters
    if spatial_aware:
        return multivariate.spatial_segment(ds, mzs, n_clusters=(None if auto else k),
                                            tol_ppm=tol_ppm, norm=norm, random_state=seed)
    if auto:
        return spatial.auto_segment(ds, mzs, tol_ppm=tol_ppm, norm=norm, random_state=seed)
    return spatial.segment(ds, mzs, n_clusters=k, tol_ppm=tol_ppm, norm=norm, random_state=seed)

# t-SNE/UMAP are ~O(n²); on a large slide they can hang the components tab for
# minutes. Cap how many pixels are embedded (a random subset is coloured, the rest
# stay transparent) so the tab stays responsive. PCA/NMF are linear → uncapped.
EMBED_PIXEL_CAP = 20000


def run_components(ds, mzs, method, n, tol_ppm, norm, mask=None):
    """Run a decomposition/embedding. ``mask`` (bool[n_pix]) restricts it to a region's
    pixels — PCA/NMF fit on those pixels only (score images blank elsewhere), and the
    UMAP/t-SNE embedding plots only that pixel pool — so the analysis can be scoped to
    one or a few ROIs from the Components tab."""
    seed = profiles.active_seed()                  # active-profile seed → reproducible decomposition
    if method == "PCA":
        return multivariate.pca_images(ds, mzs, n_components=n, tol_ppm=tol_ppm, norm=norm,
                                       mask=mask, random_state=seed)
    if method == "NMF":
        return multivariate.nmf_images(ds, mzs, n_components=n, tol_ppm=tol_ppm, norm=norm,
                                       mask=mask, random_state=seed)
    # Interactive exploration surface: drop UMAP's seed so it runs multicore (~3-5x on the
    # 20k-pixel cap) instead of single-threaded. The trade-off is run-to-run colour shuffle,
    # which is fine for an exploratory similarity map; PCA/NMF above stay seeded/reproducible.
    return multivariate.embedding(ds, mzs, method=method.lower(), tol_ppm=tol_ppm,
                                  norm=norm, sample=EMBED_PIXEL_CAP, mask=mask, random_state=seed,
                                  keep_features=True,   # retain ion matrix → UMAP Studio colour-by-lipid
                                  deterministic=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Item-view headers (QTableWidget / QTreeView) are drawn by the native macOS style
# with a light background even on a dark window — the one element that breaks dark
# mode. Re-style them (and a few other native-light surfaces) off the active palette
# so they track light *and* dark automatically. palette(...) resolves at paint time.
_THEME_QSS = """
QHeaderView::section {
    background-color: palette(button);
    color: palette(button-text);
    padding: 4px 6px;
    border: none;
    border-right: 1px solid palette(mid);
    border-bottom: 1px solid palette(mid);
}
QHeaderView::section:hover { background-color: palette(midlight); }
QTableCornerButton::section {
    background-color: palette(button);
    border: none;
    border-right: 1px solid palette(mid);
    border-bottom: 1px solid palette(mid);
}
QTableView {
    gridline-color: palette(mid);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}
"""

# Button tiers — object-name-scoped so only buttons built through the factories below
# (which set the objectName) are restyled; every other QPushButton keeps its native look.
# ``primaryAction`` is the one accent-filled run/apply button per screen; ``dangerAction``
# tints destructive verbs (Delete / Reset / Wipe) red. Both track the centralized ACCENT /
# C44E52 palette colours, so a single edit re-themes every primary/destructive button.
_BUTTON_QSS = f"""
QPushButton#primaryAction {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 5px 14px;
    font-weight: 600;
}}
QPushButton#primaryAction:hover    {{ background-color: {ACCENT_HOVER}; }}
QPushButton#primaryAction:pressed  {{ background-color: {ACCENT_PRESSED}; }}
QPushButton#primaryAction:disabled {{ background-color: palette(button); color: {MUTED_FG}; }}
QPushButton#dangerAction {{ color: {DANGER}; }}
QPushButton#dangerAction:disabled {{ color: {MUTED_FG}; }}

/* Menu / popover tool-buttons built through the factories (ControlBar.add_more,
   menu_button, RegionMultiSelect). A calm bordered pill that reads as a button and matches
   the native push-buttons around it. The drop-a-menu arrow is the SAME drawn chevron as a
   native QComboBox — see ``_dropdown_qss`` for the ``::menu-indicator`` rule — so the
   factories no longer append a literal "▾" (that doubled the arrow). The right padding
   below reserves the gutter that chevron sits in. */
QToolButton#menuButton {{
    border: 1px solid palette(mid);
    border-radius: 5px;
    padding: 4px 22px 4px 10px;
    color: palette(button-text);
    background-color: palette(button);
}}
QToolButton#menuButton:hover   {{ background-color: palette(midlight); }}
QToolButton#menuButton:pressed {{ background-color: palette(mid); }}
QToolButton#menuButton:disabled {{ color: {MUTED_FG}; border-color: palette(button); }}

/* Glossary "?" — a round help badge, NOT a dropdown: drop the menu arrow and the flat
   look so it reads as "help on this view" wherever it sits. Muted until hover. */
QToolButton#glossaryHelp {{
    border: 1px solid palette(mid);
    border-radius: 9px;                /* half of the fixed 18px → a clean circle */
    padding: 0px;
    color: {MUTED_FG};
    font-weight: 700;
}}
QToolButton#glossaryHelp:hover {{ background-color: palette(midlight); color: palette(button-text); }}
QToolButton#glossaryHelp::menu-indicator {{ image: none; width: 0px; }}
"""


# Where the per-theme chevron PNGs are cached. One file per colour so a live Light/Dark
# switch (which re-runs apply_theme with a new palette text colour) points the stylesheet
# at a freshly-tinted arrow without leaking handles.
_CHEVRON_DIR = os.path.join(tempfile.gettempdir(), "smile_msi_qss")


def _chevron_png(color: str) -> str:
    """Render the app's canonical down-chevron (the ``menu`` action icon) to a cached PNG
    and return a forward-slashed absolute path for a QSS ``image: url(...)``.

    This is the SINGLE arrow shared by every dropdown — the native ``QComboBox::down-arrow``
    *and* the ``QToolButton#menuButton::menu-indicator`` pill — so a combo, a menu-button,
    and a RegionMultiSelect all draw an identical chevron instead of three different arrows
    (native triangle vs. a hand-typed "▾"). Qt's QSS ``url()`` wants a real file (it does
    not parse ``data:`` URIs), hence the cache file. Painted at 2× for crisp HiDPI."""
    try:
        os.makedirs(_CHEVRON_DIR, exist_ok=True)
        path = os.path.join(_CHEVRON_DIR, f"chevron_{color.lstrip('#')}.png")
        if not os.path.exists(path):
            pm = icons.icon("menu", color=color).pixmap(QtCore.QSize(28, 28))
            if pm.isNull():                       # no QApplication / icon backend → hand-paint
                pm = _paint_chevron(color, 28)
            pm.save(path, "PNG")
        return path.replace(os.sep, "/")
    except Exception:                             # noqa: BLE001 — styling must never be fatal
        return ""


def fig_to_pixmap(fig) -> QtGui.QPixmap:
    """Render a matplotlib Figure to a QPixmap at its actual (Agg) export pixels. Works
    whether or not the figure is already bound to an Agg canvas. Does NOT close the figure —
    the caller owns its lifecycle (close it after if it's a throwaway preview render)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    img = QtGui.QImage(bytes(canvas.buffer_rgba()), w, h, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img.copy())


def _paint_chevron(color: str, size: int) -> QtGui.QPixmap:
    """A hand-painted ``⌄`` polyline — the fallback when the icon backend can't render."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidthF(size * 0.10)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    s = size
    p.drawPolyline([QtCore.QPointF(s * 0.28, s * 0.40),
                    QtCore.QPointF(s * 0.50, s * 0.62),
                    QtCore.QPointF(s * 0.72, s * 0.40)])
    p.end()
    return pm


def _dropdown_qss(arrow: str, arrow_muted: str) -> str:
    """The unified dropdown stylesheet: every native ``QComboBox`` gets the same calm
    bordered box as the menu-button pill, and both draw the SAME chevron (``arrow``). Built
    per-call because the chevron ``url()`` is palette-dependent. ``arrow_muted`` is the
    disabled tint. Appended after ``_BUTTON_QSS`` so its ``::menu-indicator`` rule overrides
    the pill's earlier one."""
    return f"""
/* ---- One box, one chevron — unify every dropdown ------------------------------------ */
QComboBox {{
    border: 1px solid palette(mid);
    border-radius: 5px;
    padding: 4px 24px 4px 10px;           /* match the menu-button pill; right gutter = chevron */
    background-color: palette(button);
    color: palette(button-text);
}}
QComboBox:hover    {{ background-color: palette(midlight); }}
QComboBox:focus    {{ border: 1px solid {ACCENT}; }}
QComboBox:disabled {{ color: {MUTED_FG}; border-color: palette(button); }}
/* The drop-down sub-control MUST carry the same fill as the field. Left transparent, Fusion
   repaints the WHOLE combo with a lighter native surface — the "lighter + taller" look that
   made dropdowns clash with the button pills. Filling it = one flat pill, matching #menuButton. */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: none;
    background-color: palette(button);
}}
QComboBox:hover::drop-down {{ background-color: palette(midlight); }}
QComboBox::down-arrow {{ image: url("{arrow}"); width: 11px; height: 11px; }}
QComboBox::down-arrow:disabled {{ image: url("{arrow_muted}"); }}

/* Editable / type-to-filter combos read as a TEXT INPUT, not a button: lighter base
   background (scoped via objectName "filterCombo"; set by make_filter_combo). */
QComboBox#filterCombo,
QComboBox#filterCombo::drop-down       {{ background-color: palette(base); }}
QComboBox#filterCombo:hover,
QComboBox#filterCombo:hover::drop-down {{ background-color: palette(base); }}
QComboBox#filterCombo:focus            {{ border: 1px solid {ACCENT}; }}

/* Popup list under any combo — match the table selection colours, calm frame. */
QComboBox QAbstractItemView {{
    border: 1px solid palette(mid);
    background-color: palette(base);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
    outline: none;
}}

/* The menu-button pill draws the SAME chevron (overrides the killed native indicator). */
QToolButton#menuButton::menu-indicator {{
    image: url("{arrow}");
    width: 11px; height: 11px;
    subcontrol-origin: padding;
    subcontrol-position: center right;
    right: 7px;
}}
QToolButton#menuButton::menu-indicator:disabled {{ image: url("{arrow_muted}"); }}
"""


def apply_theme(app):
    """Make the whole app honour the active light/dark palette consistently.

    Two native-style quirks break dark mode otherwise: item-view headers paint light
    (fixed via ``_THEME_QSS``), and pyqtgraph defaults to a hard black plot background
    with light axes — fine in dark mode but wrong in light mode. Both are now driven
    off the application palette, so the app looks right in either system theme. Must be
    called *before* any plot widget is built (pyqtgraph reads these at construction).

    The dropdown QSS (:func:`_dropdown_qss`) is assembled per-call because its chevron
    ``url()`` is tinted to the live palette's button-text colour — so a Light/Dark toggle
    re-tints every combo + menu-button arrow.

    Idempotent: the app stylesheet is *set* (not appended) so repeated calls — e.g. a
    live Appearance toggle via :func:`set_theme` — don't stack duplicate QSS."""
    pal = app.palette()
    arrow = _chevron_png(pal.color(QtGui.QPalette.ButtonText).name())
    arrow_muted = _chevron_png(MUTED_FG)
    app.setStyleSheet(_THEME_QSS + _BUTTON_QSS + _dropdown_qss(arrow, arrow_muted))
    pg.setConfigOption("background", pal.color(QtGui.QPalette.Base))
    pg.setConfigOption("foreground", pal.color(QtGui.QPalette.Text))


# --------------------------------------------------------------------------- #
# Light / Dark appearance
# --------------------------------------------------------------------------- #
# Two hand-tuned themes, nothing else. "System" was dropped: on Windows it resolved to a
# washed-out light look and meant the two themes never shared an identity (Fusion's blue
# highlight in light, our green in dark). Both modes now use the Fusion *style* with an
# explicit palette — the only combination where palette overrides reliably stick across
# platforms — and both take the green :data:`ACCENT` as their Highlight so a selection
# looks the same in either. :func:`apply_theme` then derives the QSS + the pyqtgraph plot
# background/foreground from whichever palette is active. ``DEFAULT_THEME`` is what a fresh
# install (and a migrated legacy "system" pref) lands on.
THEME_MODES = ("light", "dark")          # values persisted under the "theme" pref
DEFAULT_THEME = "dark"                    # dark-first app (tissue canvases are dark either way)


def _dark_palette():
    """The dark theme — a cool charcoal that pairs with the cool-slate light theme.

    ``Base`` is the pyqtgraph plot background (and panel/input fill); ``Highlight`` is the
    shared green :data:`ACCENT`; ``Mid`` is the border colour the header/table QSS reads."""
    c = QtGui.QColor
    text, bright, disabled = c(221, 225, 230), c(255, 255, 255), c(122, 128, 136)
    hl = QtGui.QColor(ACCENT)
    p = QtGui.QPalette()
    p.setColor(QtGui.QPalette.Window, c(29, 32, 36))
    p.setColor(QtGui.QPalette.WindowText, text)
    p.setColor(QtGui.QPalette.Base, c(21, 23, 26))            # ← pyqtgraph plot background
    p.setColor(QtGui.QPalette.AlternateBase, c(35, 38, 43))
    p.setColor(QtGui.QPalette.ToolTipBase, c(42, 46, 52))
    p.setColor(QtGui.QPalette.ToolTipText, text)
    p.setColor(QtGui.QPalette.Text, text)
    p.setColor(QtGui.QPalette.Button, c(42, 46, 52))
    p.setColor(QtGui.QPalette.ButtonText, text)
    p.setColor(QtGui.QPalette.BrightText, bright)
    p.setColor(QtGui.QPalette.Link, hl)
    p.setColor(QtGui.QPalette.Highlight, hl)
    p.setColor(QtGui.QPalette.HighlightedText, bright)
    p.setColor(QtGui.QPalette.PlaceholderText, c(167, 171, 176))   # muted text (MUTED_QSS) + input placeholders — ~7:1 on Window
    p.setColor(QtGui.QPalette.Dark, c(18, 19, 22))
    p.setColor(QtGui.QPalette.Mid, c(59, 64, 71))             # ← border colour (headers/tables)
    p.setColor(QtGui.QPalette.Midlight, c(51, 56, 63))        # ← hover fill (menu/tool buttons)
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, role, disabled)
    return p


def _light_palette():
    """The light theme — cool blue-grey "slate" surfaces with the shared green accent.

    A modern-dashboard look (not Fusion's flat grey) that pairs naturally with the cool
    charcoal dark theme. Ink is a dark slate rather than pure black so it reads softer on
    the near-white panels; ``Highlight`` is the green :data:`ACCENT`, same as dark mode."""
    c = QtGui.QColor
    ink, disabled = c(27, 37, 48), c(150, 159, 172)
    hl = QtGui.QColor(ACCENT)
    p = QtGui.QPalette()
    p.setColor(QtGui.QPalette.Window, c(235, 238, 243))       # cool-slate app background
    p.setColor(QtGui.QPalette.WindowText, ink)
    p.setColor(QtGui.QPalette.Base, c(248, 250, 252))         # ← panels / inputs / list rows
    p.setColor(QtGui.QPalette.AlternateBase, c(228, 232, 238))
    p.setColor(QtGui.QPalette.ToolTipBase, c(252, 253, 254))
    p.setColor(QtGui.QPalette.ToolTipText, ink)
    p.setColor(QtGui.QPalette.Text, ink)
    p.setColor(QtGui.QPalette.Button, c(244, 247, 250))
    p.setColor(QtGui.QPalette.ButtonText, ink)
    p.setColor(QtGui.QPalette.BrightText, c(255, 255, 255))
    p.setColor(QtGui.QPalette.Link, hl)
    p.setColor(QtGui.QPalette.Highlight, hl)
    p.setColor(QtGui.QPalette.HighlightedText, c(255, 255, 255))
    p.setColor(QtGui.QPalette.PlaceholderText, c(84, 92, 102))     # muted text (MUTED_QSS) + input placeholders — ~5.8:1 on Window (was ~2.4:1)
    p.setColor(QtGui.QPalette.Light, c(255, 255, 255))
    p.setColor(QtGui.QPalette.Dark, c(194, 201, 210))
    p.setColor(QtGui.QPalette.Mid, c(207, 214, 223))          # ← border colour (headers/tables)
    p.setColor(QtGui.QPalette.Midlight, c(227, 232, 239))     # ← hover fill (menu/tool buttons)
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, role, disabled)
    return p


def set_theme(app, mode):
    """Apply an appearance — ``"light"`` or ``"dark"`` (anything else falls back to
    :data:`DEFAULT_THEME`, which also migrates the retired ``"system"`` pref).

    Both modes use the Fusion style + an explicit palette so the OS theme is ignored
    (native Windows/macOS styles don't honour palette overrides for dark mode), and both
    share the green :data:`ACCENT` as Highlight. :func:`apply_theme` then derives the QSS
    and the pyqtgraph plot background/foreground from the active palette. Call
    :func:`retheme_open_plots` afterwards for a *live* switch — already-built plot widgets
    cache their background otherwise."""
    mode = mode if mode in THEME_MODES else DEFAULT_THEME
    app.setStyle("Fusion")                        # palette overrides only stick under Fusion
    app.setPalette(_dark_palette() if mode == "dark" else _light_palette())
    apply_theme(app)
    icons.on_theme_changed()         # drop cached icons so they recolour to the new palette


def dark_image_view(view):
    """Pin a tissue-image canvas to the fixed dark :data:`ION_CANVAS_BG`, *decoupled* from
    the app theme, and tag its GraphicsView so :func:`retheme_open_plots` leaves it dark on
    a live Light/Dark switch.

    Accepts a :class:`pyqtgraph.ImageView`, a :class:`~pyqtgraph.GraphicsView` /
    :class:`~pyqtgraph.PlotWidget`, or a :class:`~pyqtgraph.GraphicsLayoutWidget` — anything
    backing an MSI ion/segmentation/score image, which must read against black regardless of
    the chosen theme (see :data:`ION_CANVAS_BG`)."""
    gv = getattr(getattr(view, "ui", None), "graphicsView", None) or view
    try:
        gv.setProperty("ionCanvas", True)         # tag so retheme keeps it dark
        gv.setBackground(ION_CANVAS_BG)
    except Exception:                             # noqa: BLE001 — cosmetic, never fatal
        pass
    return view


def retheme_open_plots(app):
    """Push the active palette's background onto already-constructed pyqtgraph views.

    ``pg.setConfigOption`` only affects views built *after* it, so a live Appearance toggle
    must repaint the open ones explicitly. Tissue-image canvases tagged by
    :func:`dark_image_view` are kept on the fixed dark :data:`ION_CANVAS_BG` instead of the
    palette background — they read against black in either theme."""
    bg = app.palette().color(QtGui.QPalette.Base)
    for w in app.allWidgets():
        if isinstance(w, pg.GraphicsView):
            try:
                w.setBackground(ION_CANVAS_BG if w.property("ionCanvas") else bg)
            except Exception:                     # noqa: BLE001 — cosmetic, never fatal
                pass


def eye_icon(color=MUTED_FG, visible=True, size=16):
    """A small drawn eye glyph (no emoji): an open eye (optionally tinted ``color``)
    when ``visible``, or a greyed eye with a diagonal slash when hidden."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    col = QtGui.QColor(color if visible else MUTED_FG)
    pen = QtGui.QPen(col)
    pen.setWidthF(1.3)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.NoBrush)
    rect = QtCore.QRectF(2.0, size * 0.30, size - 4.0, size * 0.40)
    p.drawArc(rect, 0, 180 * 16)              # upper lid
    p.drawArc(rect, 180 * 16, 180 * 16)       # lower lid  → almond outline
    if visible:
        p.setBrush(col)
        p.drawEllipse(QtCore.QPointF(size / 2.0, size / 2.0), size * 0.13, size * 0.13)
    else:
        slash = QtGui.QPen(QtGui.QColor(MUTED_FG))
        slash.setWidthF(1.7)
        p.setPen(slash)
        p.drawLine(QtCore.QPointF(2.0, size - 2.0), QtCore.QPointF(size - 2.0, 2.0))
    p.end()
    return QtGui.QIcon(pm)


_COLORMAP_CACHE: dict = {}


def colormap(name):
    """Resolve a named colormap to a pyqtgraph ColorMap, memoized.

    The matplotlib registry lookup is non-trivial and ``refresh_ion_image`` calls
    this on every redraw (twice per image), so the resolved object is cached by
    name — ColorMaps are immutable here and shared safely across views."""
    cached = _COLORMAP_CACHE.get(name)
    if cached is not None:
        return cached
    result = None
    for getter in (lambda: pg.colormap.getFromMatplotlib(name.lower()),
                   lambda: pg.colormap.get(name)):
        try:
            cm = getter()
            if cm is not None:
                result = cm
                break
        except Exception:  # noqa: BLE001
            continue
    if result is None:
        result = pg.colormap.get("CET-L9")
    _COLORMAP_CACHE[name] = result
    return result

class SpectrumViewBox(pg.ViewBox):
    """ViewBox for spectrum plots whose rubber-band box zooms the m/z (X) axis ONLY.

    Stock ``RectMode`` rescales *both* axes from the drawn box, ignoring
    ``setMouseEnabled(y=False)`` — drag-to-zoom would stretch the intensity axis and the
    peaks would drift into a corner. We override the box-apply step (``showAxRect``) so it
    sets the X range from the box and leaves Y on auto-range, so the peaks always fill the
    window. Pass an instance to ``pg.PlotWidget(viewBox=...)``.

    A plain left-drag *pans* the m/z axis (grab-and-slide); hold Ctrl/Cmd while dragging to
    draw the X-only zoom box instead. We pick the mode per-drag and defer to pyqtgraph's own
    handler — Y never moves because the ViewBox has ``setMouseEnabled(y=False)``."""

    def mouseDragEvent(self, ev, axis=None):
        mods = ev.modifiers()
        box = bool(mods & (QtCore.Qt.KeyboardModifier.ControlModifier
                           | QtCore.Qt.KeyboardModifier.MetaModifier))
        self.setMouseMode(pg.ViewBox.RectMode if box else pg.ViewBox.PanMode)
        super().mouseDragEvent(ev, axis=axis)

    def showAxRect(self, ax, **kwargs):
        # ``ax`` is a QRectF already mapped into data coords by mouseDragEvent.
        r = ax.normalized()
        self.setXRange(r.left(), r.right(), padding=0)
        self.enableAutoRange(axis="y", enable=True)  # refit Y to peaks in the new window
        self.setAutoVisible(y=True)
        self.sigRangeChangedManually.emit(self.state["mouseEnabled"])


def lock_legend(legend):
    """Pin a pyqtgraph LegendItem so it can't be dragged around the plot.

    0.14's LegendItem has no ``setMovable()``, and dragging is wired straight into its
    ``mouseDragEvent``/``hoverEvent``. We stop it claiming the drag on hover (so a zoom-box
    drag that starts over the legend still reaches the spectrum underneath) and neutralise
    the drag handler itself. Clicks / double-clicks (label rename) still fire."""
    if legend is not None:
        legend.hoverEvent = lambda ev: None        # don't claim the drag → falls through
        legend.mouseDragEvent = lambda ev: ev.ignore()
    return legend


def _spectrum_view(plot):
    """Make a spectrum plot friendly: left-drag draws a box that zooms the m/z axis only,
    wheel/pinch zooms the m/z axis only, the baseline stays pinned at 0, and Y auto-fits
    the peaks in the visible m/z window so data never drifts into a corner.

    The X-only box zoom needs the plot's ViewBox to be a ``SpectrumViewBox`` (construct it
    via ``pg.PlotWidget(viewBox=SpectrumViewBox())``); RectMode on a stock ViewBox would
    rescale both axes regardless of ``setMouseEnabled(y=False)``."""
    vb = plot.getViewBox()
    vb.setMouseMode(pg.ViewBox.RectMode)         # drag draws a zoom box (X-only via SpectrumViewBox)
    vb.setMouseEnabled(x=True, y=False)          # mouse controls m/z only
    vb.enableAutoRange(axis="y", enable=True)
    vb.setAutoVisible(y=True)                    # Y scales to what's visible in X
    vb.setLimits(yMin=0)                         # can't go below the baseline
    return vb


class PinchZoom(QtCore.QObject):
    """Event filter mapping a macOS trackpad pinch onto a pyqtgraph ViewBox, zooming
    about the cursor. Qt delivers a pinch as a ``QNativeGestureEvent`` that pyqtgraph
    ignores, so a trackpad pinch otherwise does nothing on the image. Wheel/right-drag
    zoom keep working alongside this."""

    def __init__(self, graphics_view, viewbox):
        super().__init__(graphics_view)
        self._gv = graphics_view          # QGraphicsView, for cursor → scene mapping
        self._vb = viewbox                # the ViewBox to zoom

    def eventFilter(self, _obj, ev):
        if ev.type() != QtCore.QEvent.Type.NativeGesture:
            return False
        try:
            if ev.gestureType() != QtCore.Qt.NativeGestureType.ZoomNativeGesture:
                return False
            value = float(ev.value())
        except Exception:  # noqa: BLE001 — defend against platform API drift
            return False
        if not value:
            return True
        # value > 0 = pinch out = zoom in → shrink the view range (scale factor < 1)
        s = 1.0 / (1.0 + value)
        center = None
        try:
            pos = ev.position() if hasattr(ev, "position") else ev.pos()
            scene_pt = self._gv.mapToScene(pos.toPoint())
            center = self._vb.mapSceneToView(scene_pt)        # zoom about the cursor
        except Exception:  # noqa: BLE001 — fall back to centering on the view
            center = None
        self._vb.scaleBy((s, s), center=center)
        return True


def enable_pinch_zoom(image_view):
    """Wire trackpad pinch-to-zoom onto a ``pg.ImageView``'s ViewBox. Returns the filter
    — keep a reference on the owner (e.g. ``self._iv_pinch``) so Qt doesn't GC it."""
    gv = image_view.ui.graphicsView
    filt = PinchZoom(gv, image_view.view)
    gv.viewport().installEventFilter(filt)
    return filt


class FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps to a new row when it runs out of width.

    A plain ``QHBoxLayout`` reports a minimum width equal to the *sum* of its
    buttons, so a row of 4–5 buttons forces its container (and the dock that holds
    it) to stay very wide and un-shrinkable. FlowLayout reports a minimum width of
    just its widest single item, so the panel can be dragged narrow — the buttons
    reflow onto extra rows instead of pinning the panel open."""

    def __init__(self, parent=None, spacing=4):
        super().__init__(parent)
        self.setSpacing(spacing)
        self._items: list[QtWidgets.QLayoutItem] = []

    def addItem(self, item):          # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):              # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):              # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):    # noqa: N802
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):      # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):      # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):               # noqa: N802
        return self.minimumSize()

    def minimumSize(self):            # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            if item.isEmpty():        # hidden widgets take no space (Qt layout convention)
                continue
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = eff.x(), eff.y(), 0
        sp = self.spacing()
        for item in self._items:
            if item.isEmpty():        # skip hidden widgets so they reserve no row space
                continue
            w, h = item.sizeHint().width(), item.sizeHint().height()
            if x + w > eff.right() and line_h > 0:        # wrap to next row
                x, y = eff.x(), y + line_h + sp
                line_h = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), QtCore.QSize(w, h)))
            x += w + sp
            line_h = max(line_h, h)
        return y + line_h - rect.y() + m.bottom()


class ElidedLabel(QtWidgets.QLabel):
    """Single-line label that ellipsizes overflowing text instead of forcing its
    container wider. Its horizontal size hint is ignored by layouts, so siblings
    sharing a row keep equal widths regardless of how long the text is."""

    def __init__(self, text="", parent=None, mode=QtCore.Qt.ElideRight):
        super().__init__(text, parent)
        self._full = text
        self._mode = mode
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)

    def setText(self, text):
        self._full = text or ""
        self._elide()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()

    def _elide(self):
        fm = self.fontMetrics()
        super().setText(fm.elidedText(self._full, self._mode, max(0, self.width())))
        self.setToolTip(self._full if self._full else "")


class _NoWheelMixin:
    """Don't let a value widget hijack the mouse wheel.

    Spinning the wheel while merely *hovering* over a QComboBox / QSpinBox / QSlider silently
    steps its value. On a feature-set picker that swaps the active set out from under you
    mid-scroll (the "I scrolled and it changed my feature set" complaint); on ``ppm_spin`` or
    ``snr_spin`` it quietly re-tunes the analysis and you re-run against a tolerance you never
    chose. Neither announces itself, so the damage is silent and the result is plausible.

    Here the wheel only acts once the widget has focus (click or tab to it first); otherwise
    the event bubbles up so the surrounding panel scrolls instead. ``hasFocus()`` accounts for
    an editable combo's line-edit focus proxy.

    Pass ``lock_wheel=True`` where a wheel step must *never* change the value — even after a
    click has left the widget focused — so the only way to switch is to open the dropdown and
    pick. Use it when a stray step is expensive (the right-hand feature-set selector rebuilds
    the whole feature list). Scrolling an *open* dropdown is unaffected: that wheel goes to the
    popup view, not here."""

    def __init__(self, *args, lock_wheel=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock_wheel = lock_wheel
        # These widgets default to WheelFocus (the wheel itself grabs focus); drop that so a
        # hover-scroll neither focuses nor steps them.
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, ev):                 # noqa: N802 (Qt API)
        if self.hasFocus() and not self._lock_wheel:
            super().wheelEvent(ev)
        else:
            ev.ignore()                       # let the enclosing scroll area take the wheel


class NoScrollComboBox(_NoWheelMixin, QtWidgets.QComboBox):
    """A wheel-safe drop-in QComboBox — global QSS still matches it. See :class:`_NoWheelMixin`."""


class NoScrollSpinBox(_NoWheelMixin, QtWidgets.QSpinBox):
    """A wheel-safe drop-in QSpinBox. See :class:`_NoWheelMixin`."""


class NoScrollDoubleSpinBox(_NoWheelMixin, QtWidgets.QDoubleSpinBox):
    """A wheel-safe drop-in QDoubleSpinBox — where ppm tolerances and S/N cutoffs live."""


class NoScrollSlider(_NoWheelMixin, QtWidgets.QSlider):
    """A wheel-safe drop-in QSlider. A stray step on the segmentation Detail slider re-cuts
    the tree, so this one really must be clicked before it moves."""


def make_filter_combo(parent=None):
    """An editable, wheel-safe combo set up for type-to-filter use (NoInsert). The caller
    still configures items, width, and completer mode.

    Tagged ``objectName("filterCombo")`` so the unified dropdown QSS (:func:`_dropdown_qss`)
    styles it as a *text input* (lighter base background) rather than the solid button-box
    every non-editable combo gets — an editable field that looks clickable reads wrong."""
    combo = NoScrollComboBox(parent)
    combo.setEditable(True)
    combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
    combo.setObjectName("filterCombo")
    return combo


def filter_combo(items=(), *, tooltip="", parent=None):
    """The standard **type-to-filter dropdown** for a list too long to scan by eye — every ion
    in the working set, every saved feature list.

    :func:`make_filter_combo` plus the completer wiring each call site was repeating: a *popup*
    completer (a dropdown of matches, not a silent inline auto-complete) that matches anywhere
    in the entry rather than only at its start — so typing ``PC 34`` finds ``782.5670 · PC
    34:1`` even though the label begins with the m/z.

    ``items`` are plain strings or ``(label, data)`` pairs; read a pair back with
    ``currentData()``. The completer follows the model, so items may be added afterwards."""
    combo = make_filter_combo(parent)
    for it in items:
        combo.addItem(*it) if isinstance(it, tuple) else combo.addItem(it)
    combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
    combo.completer().setFilterMode(QtCore.Qt.MatchContains)
    if tooltip:
        combo.setToolTip(tooltip)
    return combo


def section_title(text, parent=None):
    """A small bold heading label — the one source of truth for the titles above plots
    and tables. Replaces scattered ``setStyleSheet("font-weight:bold")`` calls (which
    also reset other inherited style); using the font keeps the palette colour."""
    lbl = QtWidgets.QLabel(text, parent)
    f = lbl.font()
    f.setBold(True)
    lbl.setFont(f)
    return lbl


def note(text, parent=None):
    """A muted, word-wrapping descriptive label — the standard one-line "what this view
    does / what to do next" hint under a header. Word-wrap (not elide) keeps the full
    text and lets the panel shrink instead of being pinned to the sentence's width.
    Module-level twin of ``RightPanelMixin._note`` so any tab module can build one."""
    lbl = QtWidgets.QLabel(text, parent)
    lbl.setStyleSheet(MUTED_QSS)
    lbl.setWordWrap(True)
    return lbl


def plot_caption(text, parent=None):
    """A muted, slightly-smaller caption to sit directly under a plot — the one place to
    name an otherwise-hidden gesture ("click a point to view that ion", "drag to
    lasso-select pixels → region"). Thin wrapper over :func:`note` at ~0.9× size so it
    reads as a hint, not body text. Word-wraps, so it never pins the panel open."""
    lbl = note(text, parent)
    f = lbl.font()
    sz = f.pointSizeF()
    if sz > 0:
        f.setPointSizeF(sz * 0.92)
    lbl.setFont(f)
    return lbl


def glossary_button(entries, parent=None, *, tooltip="What do these terms mean?"):
    """A small ``?`` popover that explains a tab's jargon without renaming any label.

    ``entries`` is an iterable of ``(term, definition)`` pairs. Clicking drops a panel
    listing each bold term over its wrapping definition — the on-tab glossary affordance
    that lets the jargon-first labels (``Moran's I``, ``VIP``, ``q (FDR)``, …) stay while
    a biologist can still find out what they mean. Built like ``ControlBar.add_more`` so
    it matches the app's other popovers. Returns the ``?`` QToolButton."""
    btn = QtWidgets.QToolButton(parent)
    btn.setText("?")
    btn.setToolTip(tooltip)
    btn.setAccessibleName("Glossary")
    btn.setObjectName("glossaryHelp")              # round help-badge style (no menu arrow)
    # Hard 18px size: QSS max-width/height is honoured by Fusion but IGNORED by the native
    # macOS style (which blew the badge up to the row height). setFixedSize clamps min==max
    # on the widget, so every style renders the same small badge.
    btn.setFixedSize(18, 18)
    btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    menu = QtWidgets.QMenu(btn)
    box = QtWidgets.QWidget()
    col = QtWidgets.QVBoxLayout(box)
    col.setContentsMargins(12, 10, 12, 10)
    col.setSpacing(8)
    for term, definition in entries:
        t = QtWidgets.QLabel(str(term))
        tf = t.font()
        tf.setBold(True)
        t.setFont(tf)
        col.addWidget(t)
        d = QtWidgets.QLabel(str(definition))
        d.setStyleSheet(MUTED_QSS)
        d.setWordWrap(True)
        d.setMaximumWidth(380)
        col.addWidget(d)
    wa = QtWidgets.QWidgetAction(menu)
    wa.setDefaultWidget(box)
    menu.addAction(wa)
    btn.setMenu(menu)
    return btn


def set_header_tooltips(table, tips):
    """Persistently tooltip a ``QTableWidget``'s column headers so jargon column names
    (``AUC``, ``q (FDR)``, ``fold``, ``VIP``, …) explain themselves on hover.

    ``tips`` maps a column-header *text* (or an int column index) to its tooltip. Call
    once at table construction, or — for a table re-headed for several result kinds (e.g.
    the stats table) — once per fill with that kind's ``tips``: each call **replaces** the
    stored map, and the reapply hook is installed only once, so calls never accumulate
    conflicting tooltips. The hook re-applies whenever the headers change, so it survives
    :func:`fill_table` (which rebuilds the header items on every refill). A re-entrancy
    guard stops the ``setToolTip → headerDataChanged → reapply`` loop."""
    table._header_tips = dict(tips)                  # replace (not merge) on every call
    if getattr(table, "_header_tip_apply", None) is None:
        state = {"busy": False}

        def _apply(*_):
            if state["busy"]:
                return
            state["busy"] = True
            try:
                tp = getattr(table, "_header_tips", {})
                for c in range(table.columnCount()):
                    item = table.horizontalHeaderItem(c)
                    if item is None:
                        continue
                    tip = tp.get(c)
                    if tip is None:
                        tip = tp.get(item.text())
                    if tip:
                        item.setToolTip(tip)
            finally:
                state["busy"] = False

        table._header_tip_apply = _apply
        model = table.model()
        if model is not None:
            model.headerDataChanged.connect(_apply)   # connect ONCE; survives refills
    table._header_tip_apply()
    return table._header_tip_apply


def tab_page():
    """A tab page widget + its top ``QVBoxLayout``, pre-set with the standard tight
    margins/spacing. Qt's default layout margins (~11px) + spacing (~6px) stacked a wide
    grey band above the first control on every tab; this is the one source of truth for
    that top chrome so the pages start compact and consistent. Returns ``(widget, vbox)``."""
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(8, 4, 8, 4)
    v.setSpacing(4)
    return w, v


class ControlBar(QtWidgets.QWidget):
    """A wrapping control strip for the row of widgets at the top of an analysis tab.

    Hand-built ``QHBoxLayout`` control rows overflow off the right edge — and slide
    under the right-hand dock — once the window is narrow or the row gains a long
    status readout (the recurring "text cut off / stray chip" bug). ``ControlBar``
    lays its widgets out with :class:`FlowLayout`, so they reflow onto extra rows
    instead of being clipped, and parks long status text on its own full-width,
    word-wrapping line below the controls (never overflowing, never truncated).

    Usage::

        bar = ControlBar()
        bar.add_group("Region A:", combo_a, "vs B:", combo_b)   # one non-splitting unit
        bar.add(compare_button, overlay_checkbox)               # standalone items
        bar.set_status("23 features · All slide")               # wrapping line beneath
        layout.addWidget(bar)
    """

    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._outer = QtWidgets.QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(3)
        self._flow = FlowLayout(spacing=spacing)
        self._outer.addLayout(self._flow)
        self._status = None
        # A control strip, not a panel: never grow past its content height, or a tab whose
        # main content lacks vertical stretch would share the slack and inflate this bar.
        # Maximum alone isn't enough: FlowLayout.sizeHint() reports its height at the bar's
        # *minimum* width (where the buttons wrap into many rows), so the parent hands the
        # bar that tall hint and a fat gap opens below the last button on a wide window.
        # resizeEvent below pins the height to heightForWidth() at the *actual* width.
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

    def _fit_to_content(self):
        """Cap the bar to exactly the height its wrapped rows need at the current width.

        FlowLayout's height depends on width (fewer rows when wider), so its sizeHint —
        computed at the bar's narrow minimum width — overstates the height on a wide
        window. Pinning maximumHeight to heightForWidth(actual width) makes the bar hug
        the bottom of its last row, so the content below starts right beneath it instead
        of after a tall blank gap. Re-run on resize (width changed) and on LayoutRequest
        (a control or the status line was added / its text rewrapped), so the cap is never
        stale and clips. The h != maximumHeight() guard stops the setMaximumHeight ->
        LayoutRequest cycle once it settles (width is fixed across a pass, so it converges)."""
        lay = self.layout()
        if lay is None or not lay.hasHeightForWidth() or self.width() <= 0:
            return
        h = lay.heightForWidth(self.width())
        if h > 0 and h != self.maximumHeight():
            self.setMaximumHeight(h)

    def resizeEvent(self, ev):                          # noqa: N802 (Qt API)
        super().resizeEvent(ev)
        self._fit_to_content()

    def event(self, ev):
        if ev.type() == QtCore.QEvent.LayoutRequest:
            self._fit_to_content()
            # A child being shown/hidden (e.g. the contextual ROI or zoom sub-tools that
            # appear only while their toggle is on) reflows the strip, but the area a
            # now-hidden widget vacated isn't always repainted until the next toolbar
            # event — so the revealed controls used to linger on screen after their toggle
            # was switched off (you had to flip it again to clear them). Repaint the strip
            # on every relayout so contextual controls appear / disappear immediately.
            self.update()
        return super().event(ev)

    def add(self, *widgets):
        """Add one or more standalone widgets to the flow (each can wrap independently)."""
        for wdg in widgets:
            self._flow.addWidget(wdg)
        return self

    def add_group(self, *items):
        """Add a label + its controls as a single unit that always wraps together.

        String items become QLabels; everything else is added as-is — so
        ``add_group("Region A:", combo)`` keeps the caption glued to its control."""
        cell = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(cell)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        for it in items:
            h.addWidget(QtWidgets.QLabel(it) if isinstance(it, str) else it)
        self._flow.addWidget(cell)
        return cell

    def add_more(self, *items, label="More", glyph="", name=None):
        """Add a ``<label> ▾`` popover button that holds secondary controls, so the main bar
        keeps only the primary action(s) (progressive disclosure). ``name`` puts a canonical
        action icon on the leading edge (preferred, e.g. ``name="export"`` / ``"overflow"``);
        the legacy ``glyph`` still prefixes a literal character for the no-icon path.

        Items may be a bare widget, a ``("caption", widget)`` pair for a labelled row,
        or a plain caption string for a small heading. A plain action button inside the
        popover auto-closes it when clicked (terminal action); checkboxes / menu buttons
        stay open so you can flip several. Returns the popover button."""
        btn = QtWidgets.QToolButton()
        btn.setText(f"{glyph} {label}".strip())       # ▾ is drawn by QSS (::menu-indicator), not text
        btn.setObjectName("menuButton")               # bordered pill + drawn chevron (see _dropdown_qss)
        if name:
            btn.setIcon(icons.icon(name))
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setToolTip("More options for this view")
        btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(btn)
        box = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(box)
        col.setContentsMargins(12, 10, 12, 10)
        col.setSpacing(7)
        for it in items:
            wdg = None
            if isinstance(it, tuple):
                cap, wdg = it
                r = QtWidgets.QHBoxLayout()
                r.setContentsMargins(0, 0, 0, 0)
                r.setSpacing(6)
                r.addWidget(QtWidgets.QLabel(cap))
                r.addWidget(wdg, 1)
                col.addLayout(r)
            elif isinstance(it, str):
                col.addWidget(section_title(it))
            else:
                wdg = it
                col.addWidget(it)
            if isinstance(wdg, QtWidgets.QPushButton) and wdg.menu() is None:
                wdg.clicked.connect(menu.close)        # terminal action → dismiss popover
        wa = QtWidgets.QWidgetAction(menu)
        wa.setDefaultWidget(box)
        menu.addAction(wa)
        btn.setMenu(menu)
        self._flow.addWidget(btn)
        return btn

    def set_status(self, text=""):
        """Set the muted, full-width, word-wrapping status line below the controls.
        Returns the label so callers that referenced it before (e.g. ``self.cmp_info``)
        keep working — its ``.text()`` is the full, untruncated string."""
        if self._status is None:
            self._status = note("")
            self._outer.addWidget(self._status)
        self._status.setText(text)
        return self._status


def tool_button(glyph="", tooltip="", slot=None, parent=None, checkable=False, *,
                name=None):
    """A compact, auto-raised, **icon-only** button for universal repeating verbs
    (export ⤓ / refresh ↺ / remove ✕ / run ▶). This is the standard way to render the
    symbols that recur on many tabs so they look identical everywhere.

    Pass ``name`` (a canonical action) to get the shared QIcon — the preferred form, e.g.
    ``tool_button(name="export", tooltip="Export selection…", slot=…)``. The legacy form
    ``tool_button("⤓", "…")`` still works: a bare unicode ``glyph`` shows as text. The
    tooltip is the only label the user gets (keep it short, verb-first) and mirrors into
    ``accessibleName`` so screen readers + GUI tests can find the button by name."""
    b = QtWidgets.QToolButton(parent)
    if name:
        b.setIcon(icons.icon(name))               # icon-only: no text, tooltip is the label
    else:
        b.setText(glyph)
    b.setToolTip(tooltip)
    b.setAccessibleName(tooltip or (name or ""))
    b.setAutoRaise(True)
    if checkable:
        b.setCheckable(True)
    if slot is not None:
        (b.toggled if checkable else b.clicked).connect(slot)
    return b


# --------------------------------------------------------------------------- #
# Button factories — the one way to build a text button, so visual weight is
# consistent everywhere instead of ad-hoc per call site. Tiers:
#   primary   → the single run/apply action on a screen (accent fill, bold)
#   standard  → every other action (native look)
#   danger    → destructive verbs (Delete / Reset / Wipe) — red tint + confirm()
#   menu      → a button that drops a menu (text + ▾)
# Glyph vocabulary (use these; avoid emoji): the constants below + ⤓ save/export,
# ✕ clear/close, ↺ reset — already used via tool_button().
# --------------------------------------------------------------------------- #
# These now derive from the icons vocabulary so there is ONE place that defines them.
# They remain the *text* fallback (no-qtawesome builds, or any label still built by
# string concatenation); the buttons themselves carry the real QIcon via icon(name).
RUN_GLYPH = glyph("run")     # "▶" — prefix a run-style primary label (text fallback)
MENU_GLYPH = glyph("menu")   # "▾" — trailing marker that a button opens a menu


def _strip_glyph(text, *names):
    """Drop a now-redundant *leading* action glyph (and its trailing space) from a label
    once the button carries the equivalent icon — e.g. ``"▶ Run"`` → ``"Run"``. Keeps every
    real word, so any test asserting on the words still matches. No-op if absent."""
    if not text:
        return text
    for nm in names:
        g = glyph(nm)
        if g and text.startswith(g):
            return text[len(g):].lstrip()
    return text


def button(text, slot=None, *, tooltip="", primary=False, danger=False,
           icon=None, parent=None):
    """Build a QPushButton the standard way: optional tooltip + click slot, and an
    object-name that opts it into the primary (accent) or danger (red) style from
    ``_BUTTON_QSS``. Plain by default (native look).

    ``icon`` is a canonical action name (``"export"``, ``"delete"`` …) — the standard way
    to put the shared, consistent glyph on the button's leading edge. Pass an empty
    ``text`` for an icon-only button (give a ``tooltip`` — it also becomes the
    accessibleName so screen readers + GUI tests can still find it)."""
    b = QtWidgets.QPushButton(text, parent)
    if icon:
        b.setIcon(icons.icon(icon))
        if not text:                              # icon-only → keep it findable & explained
            b.setAccessibleName(tooltip or icon)
    if tooltip:
        b.setToolTip(tooltip)
    if primary:
        b.setObjectName("primaryAction")
    elif danger:
        b.setObjectName("dangerAction")
    if slot is not None:
        b.clicked.connect(slot)
    return b


def icon_button(name, text="", slot=None, *, tooltip="", primary=False, danger=False,
                parent=None):
    """A QPushButton carrying a canonical action icon — the one helper for action buttons.

    ``name`` is the action (``"export"``, ``"add"``, ``"delete"`` …). With ``text`` you get
    the standard *text + leading icon* button; with ``text=""`` an icon-only button (a
    ``tooltip`` is then required and mirrors into accessibleName). A leading glyph already
    in ``text`` is stripped so the icon isn't doubled. ``danger=True`` red-tints destructive
    verbs (pair with :func:`confirm`)."""
    return button(_strip_glyph(text, name), slot, tooltip=tooltip or text,
                  primary=primary, danger=danger, icon=name, parent=parent)


def primary_button(text, slot=None, *, tooltip="", action="run", parent=None):
    """The one run/apply action on a screen — accent-filled and bold so it reads as the
    obvious next step, with the shared ``action`` icon on its leading edge (default
    ``"run"`` → ▶). Pass ``action=None`` for an accent button with no icon. A literal
    leading ``▶`` in ``text`` is stripped so the play icon isn't doubled."""
    return button(_strip_glyph(text, action) if action else text, slot,
                  tooltip=tooltip, primary=True, icon=action, parent=parent)


def menu_button(text, items=None, *, tooltip="", parent=None, marker=True, name=None):
    """A text button that drops a menu on click (InstantPopup), with a trailing ``▾`` so
    it reads as a menu rather than a plain action. ``items`` is an iterable of
    ``(label, slot)`` pairs, ``QAction``s, or ``None`` for a separator; omit it to fill
    ``btn.menu()`` yourself. Standardizes the hand-built ``QToolButton`` + ``QMenu``
    pattern scattered across the tabs.

    ``name`` optionally puts a canonical action icon on the **leading** edge (e.g.
    ``name="export"`` for an export menu), keeping the trailing ▾ as the drops-a-menu
    marker — so a menu's verb reads the same as the equivalent plain button."""
    b = QtWidgets.QToolButton(parent)
    b.setText(text)                                # the ▾ is drawn by QSS (::menu-indicator), not text
    b.setObjectName("menuButton")                  # bordered pill + drawn chevron (see _dropdown_qss)
    if name:
        b.setIcon(icons.icon(name))
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    else:
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
    b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    if tooltip:
        b.setToolTip(tooltip)
    menu = QtWidgets.QMenu(b)
    b.setMenu(menu)
    for it in (items or []):
        if it is None:
            menu.addSeparator()
        elif isinstance(it, QtGui.QAction):
            menu.addAction(it)
        else:
            label, slot = it
            menu.addAction(label, slot)
    return b


def _natural_key(s):
    """Sort key that orders embedded numbers numerically, not lexically — so 'm/z 90' sorts
    before 'm/z 100' and 'region 2' before 'region 10'. Splitting on digit runs yields the
    same str/int slot parity for every input, so mixed-type comparisons never arise."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


class CheckList(QtWidgets.QWidget):
    """An embeddable *filter → bulk-select → tick* list. The one multi-select primitive.

    Fourteen modules hand-rolled a bare ``QListWidget`` of checkboxes with no way to tick more
    than one row at a time; on a slide with twenty regions that is twenty clicks. This adds the
    three affordances that make a checklist usable at scale, and every one of them **respects
    the active filter** — type three letters, hit *All*, and you have ticked exactly that subset:

    * **All / None / Invert** over the currently *visible* rows.
    * **Extended selection + Space** — click the first row, shift-click the last, press Space to
      toggle the whole run. (Deliberately *not* click-anywhere-to-toggle: that double-fires when
      the click lands on the checkbox itself.)
    * a live ``n of m selected`` count, so a filtered view never hides what is really ticked.

    ``entries`` are ``(key, label)``, ``(key, label, group)`` or ``(key, label, group, enabled)``.
    A disabled row is shown greyed and is skipped by every bulk action. Checked state is keyed,
    so it survives re-filtering and re-sorting. Read it back with :meth:`checked_keys`.

    ``icon_for(key)`` supplies a per-row swatch (region colours). ``extra_actions`` adds domain
    shortcuts next to All/None/Invert as ``(label, tooltip, predicate(key) -> bool)`` — the
    export hub's *Visible* / *Identified*. Both the filter and the shift+Space hint hide
    themselves on lists shorter than ``filter_min`` rows, where they cost more than they buy."""

    changed = QtCore.Signal()

    #: below this many rows the filter box and the shift+Space hint are hidden — on a
    #: three-region dialog that chrome is louder than the list it is meant to tame.
    FILTER_MIN = 8

    def __init__(self, entries=(), *, checked=(), noun="item", plural=None, parent=None,
                 icon_for=None, extra_actions=(), filter_min=None):
        super().__init__(parent)
        self._rows = []
        self._checked = set(checked)
        self._seen = set(checked)
        self._noun = noun
        self._icon_for = icon_for
        self._filter_min = self.FILTER_MIN if filter_min is None else filter_min
        plural = plural or f"{noun}s"                  # 'spectrum' must not become 'spectrums'

        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText(f"Filter {plural}…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._repopulate)
        col.addWidget(self._filter)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for text, tip, fn in (
                ("All", f"Tick every {noun} shown (respects the filter).", lambda: self.set_all(True)),
                ("None", f"Untick every {noun} shown (respects the filter).", lambda: self.set_all(False)),
                ("Invert", f"Flip every {noun} shown (respects the filter).", self.invert)):
            b = button(text, fn, tooltip=tip)
            row.addWidget(b)
        # Domain shortcuts ("Visible", "Identified", …) live beside All/None/Invert and obey
        # the same filter-respecting contract: predicate(key) decides each shown row's state.
        # The slot must swallow clicked(bool): PySide6 reads `lambda p=pred:` as arity-1 and
        # feeds the checked flag straight into `p`, so the predicate arrives as False.
        for text, tip, pred in extra_actions:
            row.addWidget(button(text, self._where_slot(pred), tooltip=tip))
        row.addStretch(1)
        self._count = QtWidgets.QLabel("")
        self._count.setStyleSheet(MUTED_QSS)
        row.addWidget(self._count)
        col.addLayout(row)

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._list.itemChanged.connect(self._item_changed)
        self._list.installEventFilter(self)
        col.addWidget(self._list, 1)
        self._hint = note("Click a row, shift-click another, then press Space to toggle the run.")
        col.addWidget(self._hint)

        self.set_entries(entries)

    # ---- data ------------------------------------------------------------ #
    def set_entries(self, entries, *, check_new=False):
        """Replace the rows. Keys already ticked stay ticked; unknown keys are dropped.

        ``check_new`` ticks keys this list has never seen before — the "a newly added sample
        joins the run by default" rule, which a plain re-populate would otherwise leave off."""
        self._rows = [(e[0], e[1], (e[2] if len(e) > 2 else "") or "",
                       bool(e[3]) if len(e) > 3 else True) for e in (entries or [])]
        keys = {k for k, _, _, _ in self._rows}
        if check_new:
            self._checked |= {k for k, _, _, en in self._rows if en and k not in self._seen}
        self._seen |= keys
        self._checked &= keys
        self._sync_chrome()
        self._repopulate()

    def _sync_chrome(self):
        """Hide the filter + shift-Space hint on lists too short to need them."""
        big = len(self._rows) > self._filter_min
        self._filter.setVisible(big)
        self._hint.setVisible(big)
        if not big and self._filter.text():
            self._filter.clear()               # a stale needle must never hide rows silently

    def checked_keys(self):
        """The ticked keys — *all* of them, not just the ones the filter is showing."""
        return set(self._checked)

    def checked_in_order(self):
        """The ticked keys in row order. Correlation matrices, group columns and export
        filenames all inherit the list's order, so a set would scramble them."""
        return [k for k, _, _, _ in self._rows if k in self._checked]

    def count(self):
        """How many rows the list holds — *all* of them, not the subset the filter shows."""
        return len(self._rows)

    def set_checked(self, keys):
        enabled = {k for k, _, _, en in self._rows if en}
        self._checked = set(keys) & enabled
        self._repopulate()

    def set_icon_for(self, icon_for):
        """Swap the per-row swatch source. Long-lived lists rebuilt from changing regions
        need this: the colours move with the rows."""
        self._icon_for = icon_for
        self._repopulate()

    # ---- bulk actions (visible + enabled rows only) ----------------------- #
    def _visible(self):
        return [self._list.item(i) for i in range(self._list.count())
                if self._list.item(i).flags() & QtCore.Qt.ItemIsEnabled]

    def set_all(self, on):
        for it in self._visible():
            self._checked.add(it.data(QtCore.Qt.UserRole)) if on else \
                self._checked.discard(it.data(QtCore.Qt.UserRole))
        self._repopulate()
        self.changed.emit()

    def invert(self):
        for it in self._visible():
            k = it.data(QtCore.Qt.UserRole)
            self._checked.discard(k) if k in self._checked else self._checked.add(k)
        self._repopulate()
        self.changed.emit()

    def _where_slot(self, predicate):
        """A clicked(bool)-proof slot bound to one predicate."""
        return lambda *_: self.set_where(predicate)

    def set_where(self, predicate):
        """Tick the shown rows satisfying ``predicate(key)`` and untick the rest of them.
        Rows the filter hides keep whatever state they had."""
        for it in self._visible():
            k = it.data(QtCore.Qt.UserRole)
            self._checked.add(k) if predicate(k) else self._checked.discard(k)
        self._repopulate()
        self.changed.emit()

    def eventFilter(self, obj, ev):                    # noqa: N802 (Qt API)
        if obj is self._list and ev.type() == QtCore.QEvent.KeyPress \
                and ev.key() == QtCore.Qt.Key_Space:
            # Keys, not items: _repopulate() destroys every QListWidgetItem, so holding the
            # objects across it and re-selecting them raises "Internal C++ object already deleted".
            sel_keys = [it.data(QtCore.Qt.UserRole) for it in self._list.selectedItems()
                        if it.flags() & QtCore.Qt.ItemIsEnabled]
            if sel_keys:
                # the run follows the FIRST row's new state, so one press can't half-toggle it
                on = sel_keys[0] not in self._checked
                for k in sel_keys:
                    self._checked.add(k) if on else self._checked.discard(k)
                self._repopulate()
                for i in range(self._list.count()):     # keep the run selected for a second press
                    it = self._list.item(i)
                    if it.data(QtCore.Qt.UserRole) in sel_keys:
                        it.setSelected(True)
                self.changed.emit()
                return True
        return super().eventFilter(obj, ev)

    # ---- rendering -------------------------------------------------------- #
    def _repopulate(self, *_):
        needle = self._filter.text().strip().lower()
        self._list.blockSignals(True)
        self._list.clear()
        for key, label, group, enabled in self._rows:
            if needle and needle not in f"{label} {group}".lower():
                continue
            it = QtWidgets.QListWidgetItem(label if not group else f"{label}   ·  {group}")
            it.setData(QtCore.Qt.UserRole, key)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            if not enabled:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.Checked if key in self._checked else QtCore.Qt.Unchecked)
            if self._icon_for is not None:
                try:
                    ic = self._icon_for(key)
                except Exception:                          # noqa: BLE001 — swatch is cosmetic
                    ic = None
                if ic is not None:
                    it.setIcon(ic)
            self._list.addItem(it)
        self._list.blockSignals(False)
        n_en = sum(1 for _, _, _, en in self._rows if en)
        self._count.setText(f"{len(self._checked)} of {n_en} selected")

    def _item_changed(self, item):
        key = item.data(QtCore.Qt.UserRole)
        if item.checkState() == QtCore.Qt.Checked:
            self._checked.add(key)
        else:
            self._checked.discard(key)
        n_en = sum(1 for _, _, _, en in self._rows if en)
        self._count.setText(f"{len(self._checked)} of {n_en} selected")
        self.changed.emit()


def check_tree_leaves(tree, *, visible_only=True):
    """Every checkable leaf of a ``QTreeWidget``, skipping headers and (by default) rows the
    filter has hidden. The tree analog of :meth:`CheckList._visible`."""
    out = []

    def walk(item):
        for i in range(item.childCount()):
            ch = item.child(i)
            if visible_only and ch.isHidden():
                continue
            if ch.childCount():
                walk(ch)
            elif ch.flags() & QtCore.Qt.ItemIsUserCheckable and ch.flags() & QtCore.Qt.ItemIsEnabled:
                out.append(ch)

    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        if visible_only and top.isHidden():
            continue
        if top.childCount():
            walk(top)
        elif top.flags() & QtCore.Qt.ItemIsUserCheckable and top.flags() & QtCore.Qt.ItemIsEnabled:
            out.append(top)
    return out


def check_list_items(lw, *, visible_only=True):
    """Every checkable, enabled row of a ``QListWidget``, skipping (by default) the ones a
    filter has hidden. The flat sibling of :func:`check_tree_leaves`."""
    out = []
    for i in range(lw.count()):
        it = lw.item(i)
        if visible_only and it.isHidden():
            continue
        if it.flags() & QtCore.Qt.ItemIsUserCheckable and it.flags() & QtCore.Qt.ItemIsEnabled:
            out.append(it)
    return out


def check_table_items(table, col=0, *, visible_only=True):
    """Every checkable, enabled cell of a ``QTableWidget``'s tick column, skipping (by default)
    the rows a filter or sort has hidden. The table sibling of :func:`check_tree_leaves`."""
    out = []
    for r in range(table.rowCount()):
        if visible_only and table.isRowHidden(r):
            continue
        it = table.item(r, col)
        if it is None:
            continue
        if it.flags() & QtCore.Qt.ItemIsUserCheckable and it.flags() & QtCore.Qt.ItemIsEnabled:
            out.append(it)
    return out


def check_table_bar(table, col=0, noun="row", *, on_change=None):
    """:func:`_bulk_bar` for a ``QTableWidget`` whose column ``col`` is a tick box — the
    segmentation's cluster table, the calibration levels. Same contract as everywhere else."""
    return _bulk_bar(
        table, noun,
        rows_visible=lambda: check_table_items(table, col),
        rows_all=lambda: check_table_items(table, col, visible_only=False),
        get_state=lambda it: it.checkState() == QtCore.Qt.Checked,
        set_state=lambda it, on: it.setCheckState(
            QtCore.Qt.Checked if on else QtCore.Qt.Unchecked),
        on_change=on_change)


def coalesce(owner, fn):
    """Wrap ``fn`` so a burst of signals costs two runs, not one per signal.

    A lone call runs ``fn`` straight away (so a single tick updates the label synchronously);
    further calls before the event loop next idles are dropped, and one deferred run at the
    end covers them. Ticking a parent row carrying ``ItemIsAutoTristate`` emits ``itemChanged``
    for every child, so a handler that walks the whole tree is O(N²) — measured at 7.8 s for a
    1000-ion feature list and 70 s for 3000."""
    timer = QtCore.QTimer(owner)
    timer.setSingleShot(True)
    timer.setInterval(0)
    timer.timeout.connect(fn)

    def run(*_):
        if timer.isActive():                        # mid-cascade — the pending run covers it
            return
        fn()
        timer.start()

    return run


def _bulk_bar(widget, noun, rows_visible, rows_all, get_state, set_state, on_change):
    """The **All / None / Invert + live count** row, over whatever rows the caller enumerates.

    The whole contract lives here so a tree, a plain list and :class:`CheckList` cannot drift
    apart: every action is scoped to the *visible* rows (it composes with a filter box above
    it — narrow, hit All, and you have ticked exactly that subset), rows the filter hides keep
    the state they had, and the count reports over *every* row so a filtered view can never
    hide what is really ticked.
    """
    row = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    count = QtWidgets.QLabel("")
    count.setStyleSheet(MUTED_QSS)

    def refresh_count():
        try:
            every = rows_all()
            n = sum(1 for it in every if get_state(it))
            count.setText(f"{n} of {len(every)} selected")
        except RuntimeError:                        # the bar went away before a deferred run
            return

    def apply(fn):
        widget.blockSignals(True)                   # one repaint, not one per row
        for it in rows_visible():
            set_state(it, fn(it))
        widget.blockSignals(False)
        refresh_count()
        if on_change is not None:
            on_change()

    for text, tip, fn in (
            ("All", f"Tick every {noun} shown (respects the filter).", lambda _it: True),
            ("None", f"Untick every {noun} shown (respects the filter).", lambda _it: False),
            ("Invert", f"Flip every {noun} shown (respects the filter).",
             lambda it: not get_state(it))):
        # `lambda *_, f=fn` — a keyword-only default can't be clobbered by clicked(bool).
        lay.addWidget(button(text, (lambda *_, f=fn: apply(f)), tooltip=tip))
    lay.addStretch(1)
    lay.addWidget(count)
    row.refresh_count = refresh_count
    widget.itemChanged.connect(coalesce(row, refresh_count))
    refresh_count()
    return row


def check_tree_bar(tree, noun="item", *, on_change=None):
    """:func:`_bulk_bar` for a checkable ``QTreeWidget`` — the trees that group rows under
    headers and so can't be a flat :class:`CheckList`. Bold headers and NoItemFlags
    placeholders aren't checkable, so no bulk action can tick them. Call the attached
    ``refresh_count()`` after repopulating the tree."""
    return _bulk_bar(
        tree, noun,
        rows_visible=lambda: check_tree_leaves(tree),
        rows_all=lambda: check_tree_leaves(tree, visible_only=False),
        get_state=lambda it: it.checkState(0) == QtCore.Qt.Checked,
        set_state=lambda it, on: it.setCheckState(
            0, QtCore.Qt.Checked if on else QtCore.Qt.Unchecked),
        on_change=on_change)


def check_list_bar(lw, noun="item", *, on_change=None, reorder=False):
    """:func:`_bulk_bar` for a plain checkable ``QListWidget`` — the lists that carry per-item
    state (a label override, an original-index role) and so can't be a :class:`CheckList`,
    whose repopulate would destroy it.

    ``reorder`` adds ↑/↓ buttons that move the selected rows. Drag already reorders such a
    list, but a drag is fiddly in a 240px box and invisible until you try it."""
    row = _bulk_bar(
        lw, noun,
        rows_visible=lambda: check_list_items(lw),
        rows_all=lambda: check_list_items(lw, visible_only=False),
        get_state=lambda it: it.checkState() == QtCore.Qt.Checked,
        set_state=lambda it, on: it.setCheckState(
            QtCore.Qt.Checked if on else QtCore.Qt.Unchecked),
        on_change=on_change)
    if reorder:
        lay = row.layout()
        up = tool_button(name="up", tooltip=f"Move the selected {noun}(s) up",
                         slot=lambda: move_selected_rows(lw, -1, on_change))
        down = tool_button(name="down", tooltip=f"Move the selected {noun}(s) down",
                           slot=lambda: move_selected_rows(lw, +1, on_change))
        lay.insertWidget(3, up)          # after All/None/Invert, before the stretch
        lay.insertWidget(4, down)
    return row


def move_selected_rows(lw, delta, on_change=None):
    """Shift the selected rows of a ``QListWidget`` by ``delta`` (-1 up, +1 down), keeping them
    selected and contiguous-safe: a run already against the wall doesn't collapse into itself.

    ``takeItem``/``insertItem`` moves the *item*, so its check state, icon and per-item data
    roles ride along — the reason this is a move and not a rebuild."""
    rows = sorted(lw.row(it) for it in lw.selectedItems())
    if not rows:
        return
    if (delta < 0 and rows[0] == 0) or (delta > 0 and rows[-1] == lw.count() - 1):
        return                                       # the run is already at the wall
    lw.blockSignals(True)
    for r in (rows if delta < 0 else reversed(rows)):
        it = lw.takeItem(r)
        lw.insertItem(r + delta, it)
    lw.blockSignals(False)
    lw.clearSelection()                              # takeItem leaves the old rows selected
    # NoUpdate: the default setCurrentItem is ClearAndSelect, which would drop every row of a
    # multi-row move but the current one — so set the cursor first, then restore the run.
    lw.setCurrentItem(lw.item(rows[0] + delta), QtCore.QItemSelectionModel.NoUpdate)
    for r in rows:                                   # keep the moved run selected
        lw.item(r + delta).setSelected(True)
    if on_change is not None:
        on_change()


def install_space_toggle(lw):
    """Space toggles the checkbox of every selected row, following the *first* row's new state
    so one press can never half-toggle a mixed run. The plain-``QListWidget`` twin of
    :class:`CheckList`'s key-based handler — this one can hold the items, because nothing
    repopulates the list underneath it."""
    class _Filter(QtCore.QObject):
        def eventFilter(self, obj, ev):              # noqa: N802 (Qt API)
            if obj is lw and ev.type() == QtCore.QEvent.KeyPress \
                    and ev.key() == QtCore.Qt.Key_Space:
                sel = [it for it in lw.selectedItems()
                       if it.flags() & QtCore.Qt.ItemIsUserCheckable
                       and it.flags() & QtCore.Qt.ItemIsEnabled]
                if sel:
                    on = sel[0].checkState() != QtCore.Qt.Checked
                    for it in sel:
                        it.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
                    return True
            return super().eventFilter(obj, ev)

    f = _Filter(lw)                                  # parented to lw → outlives this call
    lw.installEventFilter(f)
    return f


class CheckPicker(QtWidgets.QToolButton):
    """A ``Label ▾`` popover holding a **searchable, sortable, checkable** list — the
    standard "pick which of these participate" control (which samples a run uses, which
    regions to pool, …). Replaces the hand-built popover+checklist pattern that was
    copied per picker, and adds the filter/sort those copies lacked.

    State is an *unchecked-keys* set you own, reached through ``unchecked_fn()``: any entry
    whose key is **absent** counts as checked, so entries added later opt IN automatically.
    Several CheckPickers can share one set (pass the same getter) — each repopulates from
    the live set when its menu opens, so they stay consistent without cross-wiring, and
    reassigning the underlying attribute is honoured on the next open.

    ``entries_fn()`` returns the pickable rows in natural ("Roster") order, each a
    ``(key, label)`` or ``(key, label, group)`` tuple; it is re-read every time the menu
    opens (so it can be lazy/expensive). **All** / **None** act on the currently *visible*
    (filtered) rows, so you can type a few letters and (un)tick just that subset — the
    bulk-select move a flat checklist can't do. Sorting offers Name — and Group when any
    row carries one — with Roster preserving ``entries_fn`` order. ``on_change`` (optional)
    fires after any edit for callers that recompute live."""

    def __init__(self, text, entries_fn, unchecked_fn, *, tooltip="",
                 on_change=None, noun="item", parent=None):
        super().__init__(parent)
        self._entries_fn = entries_fn
        self._unchecked_fn = unchecked_fn
        self._on_change = on_change
        self._sort = "Name"                               # sorted by default — the whole point
        self.setText(text)                                # ▾ drawn by QSS (menuButton chevron)
        self.setObjectName("menuButton")
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        if tooltip:
            self.setToolTip(tooltip)

        menu = QtWidgets.QMenu(self)
        box = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(box)
        col.setContentsMargins(10, 8, 10, 8)
        col.setSpacing(6)

        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText(f"Filter {noun}s…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(lambda *_: self._repopulate())
        col.addWidget(self._filter)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        all_btn = QtWidgets.QPushButton("All")
        none_btn = QtWidgets.QPushButton("None")
        all_btn.setToolTip(f"Tick every {noun} shown (respects the filter).")
        none_btn.setToolTip(f"Untick every {noun} shown (respects the filter).")
        all_btn.clicked.connect(lambda: self.set_all(True))
        none_btn.clicked.connect(lambda: self.set_all(False))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Sort:"))
        self._sort_combo = NoScrollComboBox()
        self._sort_combo.setToolTip("Order the list.")
        self._sort_combo.currentTextChanged.connect(self._on_sort)
        row.addWidget(self._sort_combo)
        col.addLayout(row)

        self._list = QtWidgets.QListWidget()
        self._list.setMaximumHeight(260)
        self._list.setMinimumWidth(240)
        self._list.itemChanged.connect(self._item_changed)
        col.addWidget(self._list)

        wa = QtWidgets.QWidgetAction(menu)
        wa.setDefaultWidget(box)
        menu.addAction(wa)
        menu.aboutToShow.connect(self.refresh)
        self.setMenu(menu)

    # ---- data ---------------------------------------------------------------- #
    def _rows(self):
        """``entries_fn()`` normalized to ``(key, label, group)``."""
        out = []
        for e in (self._entries_fn() or []):
            out.append((e[0], e[1], (e[2] if len(e) > 2 else "") or ""))
        return out

    def _ordered(self, rows):
        if self._sort == "Name":
            return sorted(rows, key=lambda r: _natural_key(r[1]))
        if self._sort == "Group":
            return sorted(rows, key=lambda r: (_natural_key(r[2]), _natural_key(r[1])))
        return rows                                       # Roster: entries_fn order

    def refresh(self):
        """Re-read entries and rebuild the list (called on every menu open). Offers the
        Group sort only when some row carries a group."""
        rows = self._rows()
        wanted = ["Roster", "Name"] + (["Group"] if any(g for _, _, g in rows) else [])
        cur = [self._sort_combo.itemText(i) for i in range(self._sort_combo.count())]
        if cur != wanted:
            self._sort_combo.blockSignals(True)
            self._sort_combo.clear()
            self._sort_combo.addItems(wanted)
            if self._sort not in wanted:
                self._sort = "Name"
            self._sort_combo.setCurrentText(self._sort)
            self._sort_combo.blockSignals(False)
        self._repopulate(rows)

    def _repopulate(self, rows=None):
        rows = self._rows() if rows is None else rows
        needle = self._filter.text().strip().lower()
        unchecked = self._unchecked_fn()
        lw = self._list
        lw.blockSignals(True)
        lw.clear()
        shown = 0
        for key, label, group in self._ordered(rows):
            if needle and needle not in str(label).lower() and needle not in str(group).lower():
                continue
            it = QtWidgets.QListWidgetItem(label)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setData(QtCore.Qt.UserRole, key)
            it.setCheckState(QtCore.Qt.Unchecked if key in unchecked else QtCore.Qt.Checked)
            lw.addItem(it)
            shown += 1
        if not shown:
            ph = QtWidgets.QListWidgetItem("(no matches)" if needle else "(nothing to pick)")
            ph.setFlags(QtCore.Qt.NoItemFlags)
            lw.addItem(ph)
        lw.blockSignals(False)

    # ---- edits --------------------------------------------------------------- #
    def _on_sort(self, name):
        self._sort = name or "Name"
        self._repopulate()

    def _item_changed(self, item):
        key = item.data(QtCore.Qt.UserRole)
        if key is None:
            return
        unchecked = self._unchecked_fn()
        if item.checkState() == QtCore.Qt.Checked:
            unchecked.discard(key)
        else:
            unchecked.add(key)
        if self._on_change:
            self._on_change()

    def set_all(self, on):
        """Tick (``on``) or untick every *currently visible* row — bulk-select scoped to the
        active filter, so typing then All/None operates on just that subset."""
        unchecked = self._unchecked_fn()
        for i in range(self._list.count()):
            key = self._list.item(i).data(QtCore.Qt.UserRole)
            if key is None:
                continue
            unchecked.discard(key) if on else unchecked.add(key)
        self._repopulate()
        if self._on_change:
            self._on_change()

    @property
    def list_widget(self):
        """The underlying checkable ``QListWidget`` (repopulate via :meth:`refresh`)."""
        return self._list


def attach_list_filter(edit, list_widget, *, match_data=False):
    """Wire a filter ``QLineEdit`` to a ``QListWidget``: rows whose text doesn't contain the
    (case-insensitive) needle are *hidden* (not removed), so check states and order survive
    and it composes with existing All/None buttons. The inline counterpart to
    :class:`CheckPicker` for the many hand-built checkable lists inside dialogs, where a
    popover button doesn't fit the layout.

    Returns the ``apply()`` callback — call it after repopulating the list so the current
    filter re-applies to fresh rows. Set ``match_data=True`` to also match against each
    item's ``UserRole`` string (e.g. a key that isn't shown in the label)."""
    def apply(*_):
        needle = edit.text().strip().lower()
        for i in range(list_widget.count()):
            it = list_widget.item(i)
            hay = it.text().lower()
            if match_data:
                hay += " " + str(it.data(QtCore.Qt.UserRole) or "").lower()
            it.setHidden(bool(needle) and needle not in hay)
    edit.textChanged.connect(apply)
    return apply


def filter_edit(placeholder="Filter…", *, parent=None):
    """A small clearable filter ``QLineEdit`` — the standard search box above a long list.
    Pair with :func:`attach_list_filter`."""
    e = QtWidgets.QLineEdit(parent)
    e.setPlaceholderText(placeholder)
    e.setClearButtonEnabled(True)
    return e


def confirm(parent, title, text, *, danger=True, ok_text=None):
    """Standard yes/no gate before a destructive or hard-to-reverse action. Returns True
    only if the user accepts; defaults to the safe (No) button so an accidental Return
    doesn't fire it. Use for Delete / Reset / Wipe and anything else not undoable."""
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon(QtWidgets.QMessageBox.Warning if danger else QtWidgets.QMessageBox.Question)
    yes = box.addButton(ok_text or "Continue", QtWidgets.QMessageBox.AcceptRole)
    box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
    box.setDefaultButton(box.buttons()[-1])          # Cancel is the safe default
    box.exec()
    return box.clickedButton() is yes


def _with_active_ion(spin, active_mz_getter):
    """Wrap an m/z spinbox with an "active ion" button that fills the current selection."""
    row = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(spin, 1)
    h.addWidget(button("active ion", lambda: spin.setValue(float(active_mz_getter() or 0.0)),
                       tooltip="Use the currently selected m/z"))
    return row


def form_from_params(params, *, values=None, active_mz_getter=None):
    """Build form rows + a values-getter from a list of ``registry.ParamSpec``-like objects.

    This is the single place a parameter form is built from the analysis registry — both
    the Flow designer (``flowdialog.py``) and the per-tab **Configure & Run** popups
    (:class:`ConfigureRunDialog`) go through it, so every analysis is configured the same
    way. Duck-typed on each spec's ``.name``, ``.label``, ``.kind``
    (``float|int|choice|bool|mz``), ``.default``, ``.lo``, ``.hi``, ``.step``,
    ``.choices``, ``.help``.

    ``values`` optionally seeds initial values by name. ``active_mz_getter`` (a callable
    returning float | None), when given, adds an "active ion" button beside ``mz`` fields.

    Returns ``(rows, getter)`` where ``rows`` is a list of ``(QLabel, field_widget)`` ready
    to drop into a ``QFormLayout`` and ``getter()`` returns ``{name: value}`` read live
    from the widgets.
    """
    values = values or {}
    rows, getters = [], {}
    for ps in params:
        init = values.get(ps.name, ps.default)
        if ps.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(init))
            getters[ps.name] = lambda w=w: bool(w.isChecked())
        elif ps.kind == "choice":
            w = NoScrollComboBox()
            w.addItems([str(c) for c in (ps.choices or [])])
            if init is not None:
                w.setCurrentText(str(init))
            getters[ps.name] = lambda w=w: w.currentText()
        elif ps.kind == "int":
            w = NoScrollSpinBox()
            w.setRange(int(ps.lo if ps.lo is not None else 0),
                       int(ps.hi if ps.hi is not None else 10**6))
            if ps.step:
                w.setSingleStep(int(ps.step))
            w.setValue(int(init if init is not None else 0))
            getters[ps.name] = lambda w=w: int(w.value())
        else:                                            # float / mz
            w = NoScrollDoubleSpinBox()
            w.setDecimals(4 if ps.kind == "mz" else 3)
            w.setRange(float(ps.lo if ps.lo is not None else 0.0),
                       float(ps.hi if ps.hi is not None else 1e9))
            if ps.step:
                w.setSingleStep(float(ps.step))
            w.setValue(float(init if init is not None else 0.0))
            getters[ps.name] = lambda w=w: float(w.value())
        field = _with_active_ion(w, active_mz_getter) if (ps.kind == "mz"
                                                          and active_mz_getter) else w
        lbl = QtWidgets.QLabel(ps.label)
        if getattr(ps, "help", ""):
            lbl.setToolTip(ps.help)
            field.setToolTip(ps.help)
        rows.append((lbl, field))
    return rows, (lambda: {n: g() for n, g in getters.items()})


class ConfigureRunDialog(QtWidgets.QDialog):
    """Standard *configure settings, then Run* popup for an analysis that computes.

    Auto-builds its parameter form from a list of ``registry.ParamSpec`` (via
    :func:`form_from_params`) — the same registry the Flow designer reads — so every
    analysis is configured the same way: a title, an optional one-line "what/when" note,
    the form, then **Cancel / Run**. Live, continuously-updating controls belong on the
    tab's persistent bar, *not* here; this dialog is for the transactional parameters that
    only matter at the moment you Run.

    Use the :meth:`run` classmethod: it shows the dialog modally and returns the
    ``{name: value}`` dict on Run, or ``None`` on Cancel::

        params = registry.REGISTRY["auto_segment"].params
        vals = ConfigureRunDialog.run(self, title="Segmentation", params=params,
                                      values=self._seg_params)
        if vals is None:
            return                       # cancelled
        self._seg_params = vals
        self.do_segment()                # reads self._seg_params

    ``extra_rows`` lets a caller add its own ``(label, widget)`` rows above the
    registry-built ones (e.g. a region picker the caller reads itself).
    """

    def __init__(self, parent=None, *, title="Configure & run", params=(), values=None,
                 help_text="", run_label="Run", run_action="run", extra_rows=None,
                 active_mz_getter=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        v = QtWidgets.QVBoxLayout(self)
        if help_text:
            v.addWidget(note(help_text))
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 4, 0, 4)
        for lbl, field in (extra_rows or []):
            form.addRow(lbl, field)
        rows, self._getter = form_from_params(params, values=values,
                                              active_mz_getter=active_mz_getter)
        for lbl, field in rows:
            form.addRow(lbl, field)
        v.addLayout(form)
        foot = QtWidgets.QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(button("Cancel", self.reject))
        self.b_run = primary_button(run_label, self.accept, action=run_action)
        self.b_run.setDefault(True)
        foot.addWidget(self.b_run)
        v.addLayout(foot)

    def values(self):
        """The current ``{name: value}`` dict read live from the form widgets."""
        return self._getter()

    @classmethod
    def run(cls, parent=None, **kw):
        """Show modally; return the ``{name: value}`` dict on Run, or ``None`` on Cancel."""
        dlg = cls(parent, **kw)
        ok = dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted
        return dlg.values() if ok else None


class ParamForm(QtWidgets.QWidget):
    """Auto-generated parameter form for a ``registry.StepDef``'s ParamSpec list.

    A self-contained ``QWidget`` (a ``QFormLayout`` of labelled widgets) built from a list
    of ``registry.ParamSpec``-like objects — duck-typed on ``.name``, ``.label``, ``.kind``
    (``float|int|choice|bool|mz``), ``.default``, ``.lo``, ``.hi``, ``.step``, ``.choices``,
    ``.help`` — with no ownership of any FlowStep or window. It emits :attr:`changed`
    (zero-arg) whenever the user edits a field, and reads out live via :meth:`values`.

    Widget mapping is preserved exactly from the Flow designer's form:
    ``bool``→QCheckBox; ``choice``→QComboBox; ``int``→QSpinBox (lo default 0, hi default
    10**6); ``float``/``mz``→QDoubleSpinBox (decimals 4 for ``mz`` else 3, lo default 0.0,
    hi default 1e9, ``singleStep`` from ``ps.step`` when set). ``ps.help`` becomes a tooltip
    on both the label and the field. If ``active_mz`` (a zero-arg callable → current m/z) is
    given, an ``mz`` field grows an "active ion" button that fills the spinbox; passing
    ``None`` omits that button entirely.

    Two behaviours here deliberately DIVERGE from the original ``_param_widget`` to fix
    latent bugs (see class body):
      * a stored ``choice`` value absent from ``ps.choices`` is preserved (added as an
        item), not silently rewritten to ``choices[0]`` on round-trip;
      * a negative seed/default on an ``int``/``float`` field with ``lo=None`` widens the
        lower bound instead of being clamped up to the default 0 floor.
    """

    changed = QtCore.Signal()

    def __init__(self, params, values=None, *, parent=None, active_mz=None):
        super().__init__(parent)
        values = values or {}
        self._specs = list(params)
        self._widgets = {}            # name -> the read/write widget (the spin/combo/check)
        self._getters = {}            # name -> zero-arg value reader
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        for ps in self._specs:
            w = self._build_widget(ps, values.get(ps.name, ps.default))
            self._widgets[ps.name] = w
            field = _with_active_ion(w, active_mz) if (ps.kind == "mz" and active_mz) else w
            lbl = QtWidgets.QLabel(ps.label)
            if getattr(ps, "help", ""):
                lbl.setToolTip(ps.help)
                field.setToolTip(ps.help)
            form.addRow(lbl, field)

    def _build_widget(self, ps, init):
        """Construct + seed the widget for one ParamSpec, wiring its edit signal to
        :attr:`changed` and registering its live getter under ``ps.name``."""
        if ps.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(init))
            self._getters[ps.name] = lambda w=w: bool(w.isChecked())
            w.toggled.connect(lambda *_: self.changed.emit())
            return w
        if ps.kind == "choice":
            w = NoScrollComboBox()
            w.addItems([str(c) for c in (ps.choices or [])])
            # BUGFIX (a): the original did a bare ``setCurrentText(str(val))`` — a no-op when
            # the stored value isn't among ps.choices, leaving the combo on index 0 so
            # ``values()`` would silently REWRITE the stored value to choices[0] on round-trip.
            # Preserve an out-of-choices value by adding it as an item and selecting it.
            self._seed_combo(w, init)
            self._getters[ps.name] = lambda w=w: w.currentText()
            w.currentTextChanged.connect(lambda *_: self.changed.emit())
            return w
        if ps.kind == "int":
            w = NoScrollSpinBox()
            lo = int(ps.lo) if ps.lo is not None else 0
            hi = int(ps.hi) if ps.hi is not None else 10**6
            lo, hi = self._widen(lo, hi, ps.lo, ps.hi, init, ps.default, int)
            w.setRange(lo, hi)
            if ps.step:
                w.setSingleStep(int(ps.step))
            w.setValue(int(init if init is not None else 0))
            self._getters[ps.name] = lambda w=w: int(w.value())
            w.valueChanged.connect(lambda *_: self.changed.emit())
            return w
        # float / mz
        w = NoScrollDoubleSpinBox()
        w.setDecimals(4 if ps.kind == "mz" else 3)
        lo = float(ps.lo) if ps.lo is not None else 0.0
        hi = float(ps.hi) if ps.hi is not None else 1e9
        lo, hi = self._widen(lo, hi, ps.lo, ps.hi, init, ps.default, float)
        w.setRange(lo, hi)
        if ps.step:
            w.setSingleStep(float(ps.step))
        w.setValue(float(init if init is not None else 0.0))
        self._getters[ps.name] = lambda w=w: float(w.value())
        w.valueChanged.connect(lambda *_: self.changed.emit())
        return w

    @staticmethod
    def _widen(lo, hi, spec_lo, spec_hi, init, default, cast):
        """BUGFIX (b): a negative seed/default with ``lo=None`` would be clamped up to the
        default 0 floor. When the caller pinned no explicit ``lo``, drop the floor below any
        negative value we actually need to hold (seed value or the spec default); likewise
        raise a ``hi=None`` ceiling if the value overshoots it. An explicit ``spec_lo`` /
        ``spec_hi`` is respected as given and never widened."""
        for v in (init, default):
            if v is None:
                continue
            try:
                v = cast(v)
            except (TypeError, ValueError):
                continue
            if spec_lo is None and v < lo:
                lo = v
            if spec_hi is None and v > hi:
                hi = v
        return lo, hi

    @staticmethod
    def _seed_combo(w, val):
        """Select ``val`` in ``w``; if it isn't one of the predefined choices, add it as an
        item first so the value survives a values()→set_values() round-trip (bugfix a)."""
        if val is None:
            return
        s = str(val)
        if w.findText(s) < 0:
            w.addItem(s)
        w.setCurrentText(s)

    def values(self) -> dict:
        """The current ``{name: value}`` dict read live from the widgets."""
        return {n: g() for n, g in self._getters.items()}

    def set_values(self, d: dict):
        """Repopulate every field from ``d`` (missing names fall back to each ParamSpec's
        default). Per-widget edit signals are blocked during the write so the bulk update
        emits :attr:`changed` at most once, at the end."""
        d = d or {}
        for ps in self._specs:
            w = self._widgets[ps.name]
            val = d.get(ps.name, ps.default)
            blocked = w.blockSignals(True)
            try:
                if ps.kind == "bool":
                    w.setChecked(bool(val))
                elif ps.kind == "choice":
                    self._seed_combo(w, val)
                elif ps.kind == "int":
                    iv = int(val if val is not None else 0)
                    # widen an unpinned bound so a repopulated value isn't clamped (bugfix b)
                    lo, hi = self._widen(w.minimum(), w.maximum(), ps.lo, ps.hi, iv, None, int)
                    w.setRange(lo, hi)
                    w.setValue(iv)
                else:
                    fv = float(val if val is not None else 0.0)
                    lo, hi = self._widen(w.minimum(), w.maximum(), ps.lo, ps.hi, fv, None, float)
                    w.setRange(lo, hi)
                    w.setValue(fv)
            finally:
                w.blockSignals(blocked)
        self.changed.emit()


class _TablePlaceholder(QtCore.QObject):
    """Paints a centered muted message over a ``QTableWidget`` while it has no rows, so an
    un-run analysis reads as "results appear here" instead of a big black void. The label is
    a mouse-transparent child of the table viewport; it tracks the viewport size and shows /
    hides itself as rows come and go."""

    def __init__(self, table, text):
        super().__init__(table)
        self.table = table
        self.lbl = QtWidgets.QLabel(text, table.viewport())
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl.setWordWrap(True)
        self.lbl.setMargin(18)
        self.lbl.setStyleSheet(MUTED_QSS)
        self.lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        table.viewport().installEventFilter(self)
        model = table.model()
        if model is not None:
            for sig in (model.rowsInserted, model.rowsRemoved,
                        model.modelReset, model.layoutChanged):
                sig.connect(self.sync)
        self.sync()

    def sync(self, *_):
        try:
            self.lbl.setGeometry(self.table.viewport().rect())
            vis = self.table.rowCount() == 0
            self.lbl.setVisible(vis)
            if vis:
                self.lbl.raise_()
        except RuntimeError:
            pass                                 # table/viewport torn down — nothing to sync

    def eventFilter(self, _obj, ev):             # noqa: N802 (Qt API)
        if ev.type() == QtCore.QEvent.Resize:
            self.sync()
        return False


def table_placeholder(table, text):
    """Give an empty ``QTableWidget`` a centered "do X first / results appear here" prompt
    instead of a blank void. Auto-hides once the table has rows. Returns the controller."""
    return _TablePlaceholder(table, text)


class RangeSlider(QtWidgets.QWidget):
    """Minimal horizontal two-handle range slider over 0–100 (dual-handle contrast
    handles). Emits ``valueChanged(lo, hi)`` live while dragging either handle, and
    ``editingFinished(lo, hi)`` once when the drag ends (mouse release) — connect any
    expensive work (re-rendering an image) to the latter so it runs *after* the window
    is set, not on every pixel of the drag."""
    valueChanged = QtCore.Signal(float, float)
    editingFinished = QtCore.Signal(float, float)

    def __init__(self, lo=0.0, hi=99.0, parent=None):
        super().__init__(parent)
        self._min, self._max = 0.0, 100.0
        self._lo, self._hi = float(lo), float(hi)
        self._drag = None
        self._r = 8                                   # handle radius (px)
        self.setMinimumHeight(24)
        self.setMinimumWidth(110)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def values(self):
        return self._lo, self._hi

    def setValues(self, lo, hi):
        lo, hi = float(lo), float(hi)
        if hi < lo + 1:
            hi = min(self._max, lo + 1)
        self._lo, self._hi = max(self._min, lo), min(self._max, hi)
        self.update()

    def _track(self):
        m = self._r + 2
        return m, max(m + 1, self.width() - m)

    def _x_to_val(self, x):
        x0, x1 = self._track()
        return self._min + (x - x0) / (x1 - x0) * (self._max - self._min)

    def _val_to_x(self, v):
        x0, x1 = self._track()
        return x0 + (v - self._min) / (self._max - self._min) * (x1 - x0)

    @staticmethod
    def _ex(ev):
        return ev.position().x() if hasattr(ev, "position") else ev.x()

    def mousePressEvent(self, ev):
        x = self._ex(ev)
        self._drag = "lo" if abs(x - self._val_to_x(self._lo)) <= abs(x - self._val_to_x(self._hi)) else "hi"
        self._move_to(x)

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._move_to(self._ex(ev))

    def mouseReleaseEvent(self, ev):
        was_dragging = self._drag is not None
        self._drag = None
        if was_dragging:
            self.editingFinished.emit(self._lo, self._hi)

    def _move_to(self, x):
        v = max(self._min, min(self._max, self._x_to_val(x)))
        if self._drag == "lo":
            self._lo = min(v, self._hi - 1)
        else:
            self._hi = max(v, self._lo + 1)
        self.update()
        self.valueChanged.emit(self._lo, self._hi)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        cy = self.height() / 2.0
        x0, x1 = self._track()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(128, 128, 128, 90))                  # groove
        p.drawRoundedRect(QtCore.QRectF(x0, cy - 3, x1 - x0, 6), 3, 3)
        xlo, xhi = self._val_to_x(self._lo), self._val_to_x(self._hi)
        p.setBrush(QtGui.QColor(*hex_to_rgba(ACCENT, 210)))          # selected span
        p.drawRoundedRect(QtCore.QRectF(xlo, cy - 3, xhi - xlo, 6), 3, 3)
        p.setBrush(QtGui.QColor("#e6e6e6"))                          # handles
        p.setPen(QtGui.QPen(QtGui.QColor("#202020"), 1))
        for xv in (xlo, xhi):
            p.drawEllipse(QtCore.QPointF(xv, cy), self._r, self._r)
        p.end()


class RangeSliderField(QtWidgets.QWidget):
    """A :class:`RangeSlider` paired with two type-in ``%`` spin boxes (low / high) kept in
    sync, so the window can be dragged *or* typed exactly. Mirrors the slider's two-signal
    contract: ``valueChanged(lo, hi)`` fires live (cheap UI sync), ``editingFinished(lo, hi)``
    fires once the value is *committed* — when a drag ends or a typed value is entered — so
    the consumer re-renders only after the window is set, not on every intermediate step."""
    valueChanged = QtCore.Signal(float, float)
    editingFinished = QtCore.Signal(float, float)

    def __init__(self, lo=0.0, hi=100.0, parent=None):
        super().__init__(parent)
        self.slider = RangeSlider(lo, hi)
        self.lo_spin = NoScrollDoubleSpinBox()
        self.hi_spin = NoScrollDoubleSpinBox()
        for s in (self.lo_spin, self.hi_spin):
            s.setRange(0.0, 100.0)
            s.setDecimals(0)
            s.setSuffix(" %")
            s.setKeyboardTracking(False)      # commit on Enter / focus-out, not each keystroke
            s.setFixedWidth(58)
        self.lo_spin.setValue(float(lo))
        self.hi_spin.setValue(float(hi))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.lo_spin)
        lay.addWidget(QtWidgets.QLabel("–"))
        lay.addWidget(self.hi_spin)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.editingFinished.connect(self.editingFinished.emit)
        self.lo_spin.valueChanged.connect(self._on_spin)
        self.hi_spin.valueChanged.connect(self._on_spin)

    def values(self):
        return self.slider.values()

    def setValues(self, lo, hi):
        self.slider.blockSignals(True)
        self.slider.setValues(lo, hi)
        self.slider.blockSignals(False)
        self._sync_spins(*self.slider.values())

    def _sync_spins(self, lo, hi):
        for s, v in ((self.lo_spin, lo), (self.hi_spin, hi)):
            s.blockSignals(True)
            s.setValue(float(v))
            s.blockSignals(False)

    def _on_slider(self, lo, hi):
        self._sync_spins(lo, hi)
        self.valueChanged.emit(lo, hi)

    def _on_spin(self, *_):
        """A typed value was committed → push it through the slider (which enforces the
        lo<hi gap), reflect any clamp back into the spins, and emit both signals (a typed
        entry is itself a finished edit, so it triggers the re-render directly)."""
        self.slider.blockSignals(True)
        self.slider.setValues(self.lo_spin.value(), self.hi_spin.value())
        self.slider.blockSignals(False)
        lo, hi = self.slider.values()
        self._sync_spins(lo, hi)
        self.valueChanged.emit(lo, hi)
        self.editingFinished.emit(lo, hi)


class _SortItem(QtWidgets.QTableWidgetItem):
    """Table cell that sorts numerically when its text is a number, else lexically —
    so clicking a column header sorts m/z, ppm, AUC, etc. by value."""
    def __lt__(self, other):
        try:                                              # tolerate thousands commas + a trailing %
            return (float(self.text().replace(",", "").rstrip("%"))
                    < float(other.text().replace(",", "").rstrip("%")))
        except (ValueError, AttributeError):
            return self.text() < other.text()

class RegionMultiSelect(QtWidgets.QToolButton):
    """A combo-like selector that allows picking *several* regions (their union acts as one
    group). Drop-in for the A/B ``QComboBox`` selectors: it mirrors ``addItems`` / ``clear`` /
    ``currentText`` / ``setCurrentText`` / ``setCurrentIndex`` and emits ``currentIndexChanged``
    on any change, and adds :meth:`selected_names`. Multiple ions are checked in a popup list."""
    currentIndexChanged = QtCore.Signal()

    def __init__(self, placeholder="(pick regions)", parent=None):
        super().__init__(parent)
        self._placeholder = placeholder
        self._group_of = {}                               # region name -> group tag (A/B/…)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setObjectName("menuButton")                  # standard bordered-pill dropdown look
                                                          # (see _BUTTON_QSS) — matches every
                                                          # other ▾ menu button in the app
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        menu = QtWidgets.QMenu(self)
        # quick actions so picking "all regions" (or a whole group) for a side is one click,
        # not N ticks — the recurring pain when a slide has many regions to compare.
        panel = QtWidgets.QWidget(menu)
        pv = QtWidgets.QVBoxLayout(panel)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(4)
        self._filter = QtWidgets.QLineEdit(panel)         # narrow a long region list
        self._filter.setPlaceholderText("Filter…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        pv.addWidget(self._filter)
        qa = QtWidgets.QHBoxLayout()
        b_all = QtWidgets.QToolButton(panel)
        b_all.setText("All")
        b_all.setToolTip("Select every region")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none = QtWidgets.QToolButton(panel)
        b_none.setText("None")
        b_none.setToolTip("Clear the selection")
        b_none.clicked.connect(lambda: self._set_all(False))
        qa.addWidget(b_all)
        qa.addWidget(b_none)
        qa.addStretch(1)
        pv.addLayout(qa)
        # per-group select buttons (populated by set_item_groups) — "pick all of Group A"
        self._grp_row = QtWidgets.QHBoxLayout()
        pv.addLayout(self._grp_row)
        self._list = QtWidgets.QListWidget(panel)
        self._list.setMaximumHeight(220)
        pv.addWidget(self._list)
        act = QtWidgets.QWidgetAction(menu)
        act.setDefaultWidget(panel)                       # keeps the menu open while ticking
        menu.addAction(act)
        self.setMenu(menu)
        self._list.itemChanged.connect(self._on_item)
        self._sync_text()

    def _set_all(self, on):
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            it = self._list.item(i)
            if on and self._list.isRowHidden(i):          # "All" respects an active filter
                continue
            it.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self._list.blockSignals(False)
        self._on_item()

    def _apply_filter(self, *_):
        q = self._filter.text().strip().lower()
        for i in range(self._list.count()):
            it = self._list.item(i)
            self._list.setRowHidden(i, bool(q) and q not in it.text().lower())

    def set_item_groups(self, mapping):
        """Tell the picker each region's group tag (``{name: group}``) so it can offer a
        one-click 'select all of Group X' for each distinct group — turning all-A-vs-all-B
        into one click per side instead of ticking each region by hand."""
        self._group_of = {str(k): (str(v).strip() if v else "") for k, v in (mapping or {}).items()}
        while self._grp_row.count():                       # rebuild the group buttons
            item = self._grp_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        groups = []
        for g in self._group_of.values():
            if g and g not in groups:
                groups.append(g)
        if not groups:
            return
        self._grp_row.addWidget(QtWidgets.QLabel("Group:"))
        for g in groups:
            b = QtWidgets.QToolButton()
            b.setText(g.replace("Group ", "") or g)        # "Group A" -> "A" (compact)
            b.setToolTip(f"Select every region tagged '{g}'")
            b.clicked.connect(lambda _=False, gg=g: self._select_group(gg))
            self._grp_row.addWidget(b)
        self._grp_row.addStretch(1)

    def _select_group(self, group):
        self._check_only([n for n, g in self._group_of.items() if g == group])
        self.currentIndexChanged.emit()

    # --- QComboBox-compatible surface ------------------------------------- #
    def clear(self):
        self._list.blockSignals(True)
        self._list.clear()
        self._list.blockSignals(False)
        self._sync_text()

    def addItems(self, names):
        self._list.blockSignals(True)
        for n in names:
            it = QtWidgets.QListWidgetItem(str(n))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)
            self._list.addItem(it)
        self._list.blockSignals(False)
        self._apply_filter()                              # keep a live filter applied to new rows
        self._sync_text()

    def currentText(self):
        s = self.selected_names()
        return s[0] if s else ""

    def setCurrentText(self, name):
        self._check_only([name])

    def setCurrentIndex(self, i):
        self._check_only([self._list.item(i).text()] if 0 <= i < self._list.count() else [])

    # --- multi-select API -------------------------------------------------- #
    def selected_names(self):
        return [self._list.item(i).text() for i in range(self._list.count())
                if self._list.item(i).checkState() == QtCore.Qt.Checked]

    def set_selected(self, names):
        self._check_only(list(names or []))

    def _check_only(self, names):
        want = set(names)
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            it = self._list.item(i)
            it.setCheckState(QtCore.Qt.Checked if it.text() in want else QtCore.Qt.Unchecked)
        self._list.blockSignals(False)
        self._sync_text()

    def _on_item(self, *_):
        self._sync_text()
        self.currentIndexChanged.emit()

    def _sync_text(self):
        s = self.selected_names()
        label = (", ".join(s) if 0 < len(s) <= 2
                 else (f"{len(s)} regions" if s else self._placeholder))
        self.setText(label)                               # ▾ drawn by QSS (::menu-indicator), not text


def fill_table(table: QtWidgets.QTableWidget, headers, rows):
    was_sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)                    # don't reorder mid-populate
    blocked = table.blockSignals(True)                # bulk repopulate: suppress the per-cell
    try:                                              # itemChanged storm (callers ignore it anyway)
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = _SortItem(str(val))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                table.setItem(r, c, item)
    finally:
        table.blockSignals(blocked)                   # restore prior blocking state
    table.resizeColumnsToContents()
    table.setSortingEnabled(was_sorting)              # restore (numeric-aware) sorting
    # Right-click Copy / Export on every filled table; a view with its own menu is skipped.
    install_table_export(table)


# --------------------------------------------------------------------------- #
# Generic table copy/export — one Copy (CSV) / Export affordance for every view
# --------------------------------------------------------------------------- #
def _item_text(item, col=None):
    """The text one cell copies as. ``col`` is the column for a tree item, None for a
    table/list item.

    Falls back to the tick state for checkbox-only cells (the segmentation ``✓`` column,
    the standards ``Use`` column, the lipid tree's eye) — those carry their whole meaning
    in the checkbox, so copying them as an empty string loses a column."""
    if item is None:
        return ""
    txt = item.text() if col is None else item.text(col)
    if txt:
        return txt
    chk = (item.data(QtCore.Qt.CheckStateRole) if col is None
           else item.data(col, QtCore.Qt.CheckStateRole))
    if chk is None:
        return ""
    state = QtCore.Qt.CheckState(chk)
    return {QtCore.Qt.Checked: "yes", QtCore.Qt.PartiallyChecked: "partial"}.get(state, "no")


def table_grid(table: QtWidgets.QTableWidget, *, selected_only=False):
    """Read a QTableWidget into ``(headers, rows)`` of plain strings.

    Honours the current *visual* order (a user-sorted table exports the way it
    looks) and skips hidden rows/columns (so filtered-out rows never leak into
    the file). With ``selected_only`` it returns just the highlighted rows."""
    cols = [c for c in range(table.columnCount()) if not table.isColumnHidden(c)]
    headers = []
    for c in cols:
        h = table.horizontalHeaderItem(c)
        headers.append(h.text() if h is not None else f"col{c}")
    sel_rows = None
    if selected_only:
        sel_rows = {ix.row() for ix in table.selectedIndexes()}
    rows = []
    for r in range(table.rowCount()):
        if table.isRowHidden(r):
            continue
        if sel_rows is not None and r not in sel_rows:
            continue
        rows.append([_item_text(table.item(r, c)) for c in cols])
    return headers, rows


def tree_grid(tree: QtWidgets.QTreeWidget, *, selected_only=False):
    """Read a QTreeWidget into ``(headers, rows)`` — every item is one row, in the order
    it is drawn, so a grouped roster copies the way it reads on screen."""
    cols = [c for c in range(tree.columnCount()) if not tree.isColumnHidden(c)]
    head = tree.headerItem()
    headers = [(head.text(c) if head is not None else f"col{c}") for c in cols]
    rows = []
    it = QtWidgets.QTreeWidgetItemIterator(
        tree, QtWidgets.QTreeWidgetItemIterator.NotHidden)
    while it.value() is not None:
        item = it.value()
        it += 1
        if selected_only and not item.isSelected():
            continue
        rows.append([_item_text(item, c) for c in cols])
    return headers, rows


def list_grid(lw: QtWidgets.QListWidget, *, selected_only=False):
    """Read a QListWidget into ``(headers, rows)`` — one column of visible rows."""
    rows = []
    for i in range(lw.count()):
        item = lw.item(i)
        if item is None or item.isHidden():
            continue
        if selected_only and not item.isSelected():
            continue
        rows.append([_item_text(item)])
    return ["item"], rows


def _model_grid(view, *, selected_only=False):
    """Last-resort reader for a model-backed view (QTableView / QTreeView)."""
    model = view.model()
    if model is None:
        return [], []
    cols = list(range(model.columnCount()))
    headers = [str(model.headerData(c, QtCore.Qt.Horizontal) or f"col{c}") for c in cols]
    sm = view.selectionModel()
    rows = []
    for r in range(model.rowCount()):
        if selected_only and (sm is None or not sm.isRowSelected(r, QtCore.QModelIndex())):
            continue
        rows.append([str(model.data(model.index(r, c)) or "") for c in cols])
    return headers, rows


def view_grid(view, *, selected_only=False):
    """``(headers, rows)`` for any item view — table, tree or list — so copy and export
    read the same grid whichever widget a screen happened to use."""
    if isinstance(view, QtWidgets.QTableWidget):
        return table_grid(view, selected_only=selected_only)
    if isinstance(view, QtWidgets.QTreeWidget):
        return tree_grid(view, selected_only=selected_only)
    if isinstance(view, QtWidgets.QListWidget):
        return list_grid(view, selected_only=selected_only)
    return _model_grid(view, selected_only=selected_only)


def grid_csv(headers, rows, *, sep=","):
    """``(headers, rows)`` → CSV text, quoted by the csv module (so a lipid name with a
    comma survives the round trip)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=sep, lineterminator="\n")
    if headers:
        w.writerow(headers)
    w.writerows(rows)
    return buf.getvalue()


def copy_grid(headers, rows, *, sep=","):
    """Put a grid on the clipboard as delimited text. Returns the text.

    Deliberately ``setText`` and not ``setMimeData``: a QMimeData built in Python is kept by
    the clipboard and destroyed by Qt's static teardown *after* the interpreter is gone, and
    PySide segfaults unwinding it — so any process that copied a table crashed on quit.
    Extra clipboard flavours (an HTML table for Excel) are not worth that; ``sep="\t"`` is
    the spreadsheet-friendly form instead."""
    text = grid_csv(headers, rows, sep=sep)
    QtGui.QGuiApplication.clipboard().setText(text)
    return text


def _selected_row_count(view):
    if isinstance(view, (QtWidgets.QTreeWidget, QtWidgets.QListWidget)):
        return len(view.selectedItems())
    return len({ix.row() for ix in view.selectedIndexes()})


def copy_table(view, *, selected_only=None, sep=","):
    """Copy a whole table view to the clipboard as CSV, headers included.

    ``selected_only=None`` (the default) reads the intent from the selection: two or more
    selected rows copy just those rows, anything less copies the whole visible grid.
    Clicking one row and pressing ⌘C is how you ask for *the table*, not for that one cell.

    Returns ``(n_rows, n_columns)``, or None when there was nothing to copy."""
    if selected_only is None:
        selected_only = _selected_row_count(view) > 1
    headers, rows = view_grid(view, selected_only=selected_only)
    if not rows and selected_only:                # a selection that yielded nothing
        headers, rows = view_grid(view, selected_only=False)
    if not rows:
        return None
    copy_grid(headers, rows, sep=sep)
    return len(rows), len(headers)


def _status_window(widget):
    """The nearest ancestor window that owns a status bar. Dialogs are parented to the main
    window, so a copy or export launched from one still reports where the user is looking —
    and the provenance header is found on the same object."""
    w = widget
    while w is not None:
        if hasattr(w, "statusBar"):
            return w
        w = w.parent()
    return None


def _say(widget, message):
    win = _status_window(widget)
    if win is not None:
        win.statusBar().showMessage(message)


def copy_view(view, *, selected_only=None, sep=","):
    """:func:`copy_table` plus a status-bar report of what landed on the clipboard."""
    got = copy_table(view, selected_only=selected_only, sep=sep)
    kind = "tab-separated" if sep == "\t" else "CSV"
    _say(view, f"Copied {got[0]} rows × {got[1]} columns to the clipboard ({kind})."
               if got else "Nothing to copy — the table is empty.")
    return got


def export_rows(parent, headers, rows, *, stem="table", title="Export table", empty_msg=None,
                header_lines=None):
    """Save explicit ``headers`` + ``rows`` to CSV / TSV / Excel via a file dialog. Returns the
    path or None.

    The file-writing core behind :func:`export_table`; call this directly when the data isn't in
    a QTableWidget — e.g. a self-describing export built from a result object (with extra context
    columns the visible view doesn't show) rather than the on-screen grid.

    Every export carries a leading ``# …`` audit-trail header (the dataset + software + this
    analysis); pass ``header_lines`` to override, else it's pulled from the parent window's
    provenance. CSV/TSV get the comment block; XLSX gets a separate ``Provenance`` sheet."""
    from . import filedialogs
    win = _status_window(parent)
    if not rows:
        _say(parent, empty_msg or "Nothing to export.")
        return None
    path, flt = filedialogs.get_save_file_name(
        parent, title, f"{stem}.csv",
        "CSV (*.csv);;Tab-separated (*.tsv);;Excel (*.xlsx)")
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if not ext:                                       # no suffix typed → honour the chosen filter
        ext = ".tsv" if "tsv" in flt.lower() else ".xlsx" if "xlsx" in flt.lower() else ".csv"
        path += ext
    if header_lines is None and win is not None and hasattr(win, "audit_csv_header"):
        try:                                          # dataset + software + this analysis name
            header_lines = win.audit_csv_header(analysis=str(stem).replace("_", " "))
        except Exception:  # noqa: BLE001 — a header is never worth failing an export over
            header_lines = None
    if ext == ".xlsx":
        import pandas as pd                           # lazy: keep pandas out of GUI startup
        from .. import export as _export
        _export.write_table(pd.DataFrame(rows, columns=headers), path, fmt="xlsx",
                            header_lines=header_lines)
    else:
        import csv
        sep = "\t" if ext == ".tsv" else ","
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            for ln in (header_lines or []):
                f.write(str(ln).rstrip("\n") + "\n")
            w = csv.writer(f, delimiter=sep)
            w.writerow(headers)
            w.writerows(rows)
    _say(parent, f"Wrote {path} ({len(rows)} rows)")
    return path


def export_table(parent, table, *, stem="table", title="Export table", selected_only=False):
    """Save a QTableWidget to CSV / TSV / Excel via a file dialog. Returns the path or None.

    The single primitive behind 'export everywhere': any table view offers an export with
    one call, in the same formats (and the same UTF-8-sig CSV, so Excel reads unicode lipid
    names) as the rest of the app — without each view re-implementing a DataFrame + dialog."""
    headers, rows = view_grid(table, selected_only=selected_only)
    empty = "No rows selected." if selected_only else "Nothing to export — the table is empty."
    return export_rows(parent, headers, rows, stem=stem, title=title, empty_msg=empty)


def add_copy_actions(menu, view):
    """Add **Copy table (CSV)** / **Copy selected rows** to a menu a view already builds
    itself. Views without their own menu take :func:`install_table_export` instead."""
    menu.addAction("Copy table (CSV)", lambda: copy_view(view, selected_only=False))
    act = menu.addAction("Copy selected rows", lambda: copy_view(view, selected_only=True))
    act.setEnabled(_selected_row_count(view) > 0)
    # Excel pastes comma-separated text into a single column; tabs land it in real cells.
    menu.addAction("Copy for Excel (tab-separated)",
                   lambda: copy_view(view, selected_only=False, sep="\t"))
    return menu


def install_table_export(view, parent=None, *, stem="table", title="Export table"):
    """Give any item view a right-click **Copy / Export…** menu — the one call that puts an
    export affordance on a table view.

    Idempotent, and it leaves a view that already has its own custom menu alone (those call
    :func:`add_copy_actions` from their own handler). ``parent`` is the window an export
    dialog and its status message belong to; omit it and the view's own window is used."""
    if view.property("_smile_table_menu"):
        return None
    if view.contextMenuPolicy() == QtCore.Qt.CustomContextMenu:
        return None
    view.setProperty("_smile_table_menu", True)
    view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def _menu(pos):
        menu = QtWidgets.QMenu(view)
        add_copy_actions(menu, view)
        menu.addSeparator()
        owner = parent if parent is not None else view.window()
        menu.addAction("Export table…",
                       lambda: export_table(owner, view, stem=stem, title=title))
        act = menu.addAction("Export selected rows…",
                             lambda: export_table(owner, view, stem=stem, title=title,
                                                  selected_only=True))
        act.setEnabled(bool(view.selectedIndexes()))
        menu.exec(view.viewport().mapToGlobal(pos))

    view.customContextMenuRequested.connect(_menu)
    return _menu


class _CopyKeyFilter(QtCore.QObject):
    """Application-wide ⌘C / Ctrl+C over any table, tree or list.

    Qt gives item views no copy of their own, so ⌘C over a grid of results used to leave the
    clipboard holding whatever was there before. This copies the whole visible table as CSV.
    It filters the *application* rather than each window because the Cohort screen and every
    dialog are separate top-level windows; text editors and cell editors keep their own copy."""

    def eventFilter(self, obj, ev):               # noqa: N802 (Qt API)
        if ev.type() != QtCore.QEvent.KeyPress:
            return False
        if not ev.matches(QtGui.QKeySequence.StandardKey.Copy):
            return False
        view = _view_for_copy(obj)
        if view is None:
            return False
        copy_view(view)
        return True                               # handled — don't let it bubble and copy twice


def _view_for_copy(obj):
    """The item view a key press belongs to, or None when something else should keep it."""
    if not isinstance(obj, QtWidgets.QWidget):
        return None
    if isinstance(obj, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox,
                        QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
        return None                               # incl. a cell editor open inside a table
    w = obj
    while w is not None:
        if isinstance(w, QtWidgets.QAbstractItemView):
            return w
        w = w.parentWidget()
    return None


def install_copy_shortcut(app=None):
    """Wire ⌘C / Ctrl+C to 'copy this table as CSV' for the whole application. Idempotent."""
    app = app or QtWidgets.QApplication.instance()
    if app is None or app.property("_smile_copy_filter"):
        return None
    filt = _CopyKeyFilter(app)                    # parented to the app → outlives this call
    app.installEventFilter(filt)
    app.setProperty("_smile_copy_filter", True)
    return filt


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
