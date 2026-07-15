"""Export engine — beautiful, multi-format output for every artifact the app makes.

This module is **Qt-free and pure** (so it runs headlessly and in worker threads): the
GUI collects current state into plain dicts/arrays and hands them here. It produces:

* :func:`render_ion_panel` — the signature view. A clean ion image with a small,
  **labelled intensity colour-scale** tucked into a bottom corner and a
  scale bar — so the image itself stays uncluttered. When a mean **spectrum** (or, for an
  overlay, a swatch legend) is supplied it is placed in a tidy **side margin** beside the
  image, never composited over the data. A single PNG is still fully self-describing.
* :func:`render_overlay_panel` — additive multi-ion composite with corner
  per-channel colour ramps (and an optional ROI outline).
* :func:`render_spectrum_figure` — a publication spectrum (mean / ROI / per-pixel).
* :func:`save_figure` — write any of the above to PNG / TIFF / JPEG / PDF / SVG / EPS at
  a chosen DPI.
* :func:`write_table` / :func:`write_spectra` — tables (CSV/TSV/XLSX/JSON/Markdown) and
  spectra arrays (CSV/TSV) in one call.
* :func:`build_book` — a full multi-page **PDF report** ("the book"): cover, dataset
  summary, methods + provenance + references, an ion-image gallery (each with its corner
  overlay), segmentation, and discriminating-feature statistics — one cohesive document.

Everything uses matplotlib's object-oriented API (``Figure`` + Agg/PDF canvases, no
``pyplot`` global state) so it is safe to call off the GUI thread.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import numpy as np

from .spectrum_range import signal_mz_range

from . import imaging
from . import annotations as annot
from . import palettes
from . import stylespec as _stylespec

# Output-format groups. TIFF/JPEG go through matplotlib's Pillow path; vector formats
# are written natively. Anything else falls back to PNG.
RASTER_FORMATS = ("png", "tif", "tiff", "jpg", "jpeg", "webp")
VECTOR_FORMATS = ("pdf", "svg", "eps")
IMAGE_FORMATS = RASTER_FORMATS + VECTOR_FORMATS
TABLE_FORMATS = ("csv", "tsv", "xlsx", "json", "md", "parquet")


# --------------------------------------------------------------------------- #
# Theme / style  —  the parameterized token layer (DESIGN_SPEC §0)
# --------------------------------------------------------------------------- #
# Colour is a *parameter everywhere*: the hex defaults below are examples, not locks.
# ``dark`` backs spatial figures (ion / overlay / spectrum); ``print`` (white) backs the
# analytics figures (SHAP / UMAP). The rule of thumb is a default, not a hard rule.
DEFAULT_CATEGORY_PALETTE = palettes.CATEGORY     # canonical home: smile_msi/palettes.py


@dataclass
class Style:
    """A self-contained, fully-parameterized visual theme for exported figures.

    Spec tokens (``bg``/``fg``/``muted``/``hairline`` + ``cmap``/``diverging``/``accent``/
    ``category_palette``) drive every ``render_*``; the older ``page_bg``/``page_fg``/
    ``fg_dim``/``grid`` names remain as read-only aliases so existing callers keep working.
    ``dark`` reads well over any colormap (spatial figs); ``light`` suits a white document
    (analytics)."""
    name: str = "dark"
    bg: str = "#0d1117"                          # page / figure background
    fg: str = "#e6e9ee"                          # primary ink
    muted: str = "#8a93a0"                       # secondary ink (ticks, units, values)
    hairline: tuple = (1, 1, 1, 0.14)            # gridlines / faint guides (RGBA or hex)
    grid: str = "#2a2f3a"                        # solid spine colour (kept distinct from hairline)
    accent: str | None = None                    # active-ion / marker; None ⇒ derived from cmap
    cmap: str = "viridis"                        # ion-image colormap default
    diverging: str = "bwr"                       # SHAP direction colormap default
    category_palette: tuple = DEFAULT_CATEGORY_PALETTE
    card_bg: tuple = (0.07, 0.08, 0.10, 0.72)    # glass card fill (RGBA) — PDF book corners
    card_edge: tuple = (1, 1, 1, 0.24)
    # line-weight tokens (StyleSpec-overridable; defaults = the historical literals)
    spine_w: float = 1.2                         # xy-plot left/bottom spine weight
    hairline_w: float = 1.0                      # faint grid / guide weight
    outline_w: float = 1.4                       # default ROI-outline weight

    # ---- back-compat aliases (existing render code reads these names) ----
    @property
    def page_bg(self) -> str:
        return self.bg

    @property
    def page_fg(self) -> str:
        return self.fg

    @property
    def fg_dim(self) -> str:
        return self.muted


def make_style(theme: str = "dark", accent: str | None = None, *,
               cmap: str | None = None, diverging: str | None = None,
               category_palette=None) -> Style:
    """Build a :class:`Style` for ``theme`` (``"dark"`` or ``"light"``/``"print"``/``"paper"``).
    Every colour is overridable — pass ``accent``/``cmap``/``diverging``/``category_palette``
    to override the theme defaults.

    When a :class:`~smile_msi.stylespec.StyleSpec` preset is installed via
    :func:`active_style`, its tokens (theme, colours, palette, line weights) are merged in at
    **lower precedence than the explicit arguments** here — so a per-figure choice always wins
    over the design-system default, and every ``render_*`` inherits the active look for free."""
    _ensure_fonts()
    spec = _ACTIVE
    if spec is not None and spec.theme:          # a preset may force the base theme
        theme = spec.theme
    theme = (theme or "dark").lower()
    if theme in ("light", "print", "paper"):
        s = Style(name="light", bg="#ffffff", fg="#16191d", muted="#6b7280",
                  hairline="#e6e8eb", grid="#d7dbe2",
                  card_bg=(1, 1, 1, 0.90), card_edge=(0, 0, 0, 0.22))
    else:
        s = Style()
    # active preset (below the explicit args) — a legible diff over the theme default
    if spec is not None:
        if spec.bg:
            s.bg = spec.bg
        if spec.fg:
            s.fg = spec.fg
        if spec.muted:
            s.muted = spec.muted
        if spec.grid:
            s.grid = spec.grid
        if spec.hairline is not None:
            s.hairline = tuple(spec.hairline) if isinstance(spec.hairline, list) else spec.hairline
        if spec.accent and not accent:
            s.accent = spec.accent
        if spec.image_cmap and not cmap:
            s.cmap = spec.image_cmap
        if spec.diverging_cmap and not diverging:
            s.diverging = spec.diverging_cmap
        if category_palette is None:
            cp = spec.resolved_category_palette()
            if cp:
                s.category_palette = cp
        if spec.spine_w:
            s.spine_w = _clamp_weight(spec.spine_w)
        if spec.hairline_w:
            s.hairline_w = _clamp_weight(spec.hairline_w)
        if spec.outline_w:
            s.outline_w = _clamp_weight(spec.outline_w)
    # explicit arguments always win
    if accent:
        s.accent = accent
    if cmap:
        s.cmap = cmap
    if diverging:
        s.diverging = diverging
    if category_palette:
        s.category_palette = tuple(category_palette)
    return s


def _clamp_weight(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    lo, hi = _stylespec.WEIGHT_MIN, _stylespec.WEIGHT_MAX
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------- #
# Typography — FIXED family + one named type scale (DESIGN_SPEC §0)
# --------------------------------------------------------------------------- #
# The single biggest upgrade over matplotlib's default DejaVu Sans. We *prepend* the
# Helvetica stack to the sans-serif list (rather than hard-set ``font.family``) so any
# machine missing Helvetica falls back cleanly instead of erroring.
# Helvetica first; DejaVu Sans last as a *glyph-coverage* fallback (it carries the ↑/↓/·
# /Δ/ρ glyphs Helvetica lacks). We set ``font.family`` to this explicit list so matplotlib
# does per-glyph fallback across it — a generic "sans-serif" only resolves one font and would
# render missing glyphs as tofu boxes.
_FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans",
               "DejaVu Sans"]
_fonts_ready = False


def _ensure_fonts() -> None:
    """Register the Helvetica stack as the figure font (once), with DejaVu Sans as a final
    glyph-coverage fallback. Idempotent and failure-tolerant so headless/odd matplotlib
    installs never break an export."""
    global _fonts_ready
    if _fonts_ready:
        return
    _fonts_ready = True
    try:
        import matplotlib
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        # keep only installed families (avoids per-render findfont warnings); DejaVu Sans
        # ships with matplotlib so it's always present as the glyph-coverage fallback.
        present = [f for f in _FONT_STACK if f in available]
        if "DejaVu Sans" not in present:
            present.append("DejaVu Sans")
        existing = list(matplotlib.rcParams.get("font.sans-serif", []))
        matplotlib.rcParams["font.sans-serif"] = present + [f for f in existing if f not in present]
        # explicit family list (not the generic "sans-serif") so per-glyph fallback works
        matplotlib.rcParams["font.family"] = present
    except Exception:  # noqa: BLE001
        pass


# One named type scale. Sizes are pt at a single-column (~3.4 in) figure; they scale up
# linearly with figure width and clamp so wide panels stay readable (not billboard-sized).
# Replace every literal ``fontsize=`` with ``type_pt(role, width_in)``.
TYPE_SCALE = {                 # role: (pt @ ref width, weight)
    "title":      (9.0, "bold"),
    "subtitle":   (7.5, "medium"),
    "axis":       (7.0, "normal"),
    "tick":       (6.5, "normal"),
    "annotation": (6.0, "normal"),
    "legend":     (6.5, "normal"),
}
_TYPE_REF_IN = 3.4
_TYPE_MAX_SCALE = 1.45         # cap the linear scale-up so wide figures stay clean


# The active design-system preset (a StyleSpec), installed by :func:`active_style`. ``None``
# = the app's built-in tokens. Module-global (like matplotlib's own rcParams) because a
# render is a single top-to-bottom pass; nested/parallel renders each save+restore it.
_ACTIVE = None


def type_pt(role: str, width_in: float = _TYPE_REF_IN) -> float:
    """Point size for ``role`` at a figure of physical width ``width_in`` (inches). Honours an
    active :class:`~smile_msi.stylespec.StyleSpec` (per-role pt overrides, a global size gain,
    and a custom reference width / scale cap) so a preset restyles every figure's type."""
    spec = _ACTIVE
    scale = TYPE_SCALE if spec is None else spec.resolved_type_scale(TYPE_SCALE)
    base = scale.get(role, (7.0, "normal"))[0]
    ref = _TYPE_REF_IN if (spec is None or not spec.type_ref_in) else float(spec.type_ref_in)
    maxs = _TYPE_MAX_SCALE if (spec is None or not spec.type_max_scale) else float(spec.type_max_scale)
    gain = 1.0 if spec is None else spec.resolved_gain()
    f = min(max(float(width_in or ref) / ref, 1.0), maxs)
    return round(base * f * gain, 1)


def type_weight(role: str) -> str:
    """Font weight for ``role`` (``"bold"`` / ``"medium"`` / ``"normal"``)."""
    spec = _ACTIVE
    scale = TYPE_SCALE if spec is None else spec.resolved_type_scale(TYPE_SCALE)
    return scale.get(role, (7.0, "normal"))[1]


def current_style_spec():
    """The active :class:`~smile_msi.stylespec.StyleSpec`, or ``None`` (app default)."""
    return _ACTIVE


@contextlib.contextmanager
def active_style(spec):
    """Install ``spec`` (a :class:`~smile_msi.stylespec.StyleSpec`, dict, JSON string, or preset
    name) as the active design-system bundle for the duration of the block, so every
    ``render_*`` / ``type_pt`` / ``make_style`` honours its typography, theme, palette and line
    weights — **with no change to any render body**. Restores the prior state on exit
    (nestable and exception-safe). ``active_style(None)`` (or a default spec) is a no-op, so it
    is always safe to wrap a render in it.

        with export.active_style("Poster (large type)"):
            fig = export.render_ion_panel(image=img, mz=744.55)
    """
    global _ACTIVE
    _ensure_fonts()
    resolved = _stylespec.coerce(spec)
    prev = _ACTIVE
    font_restore = None
    if resolved is not None and not resolved.is_default():
        _ACTIVE = resolved
        fam = resolved.resolved_font_family()
        if fam:
            font_restore = _apply_font_family(fam)
    try:
        yield resolved
    finally:
        _ACTIVE = prev
        if font_restore is not None:
            _restore_font_family(font_restore)


def _apply_font_family(stack):
    """Set the figure font to ``stack`` (families installed on this machine, DejaVu Sans
    appended for glyph coverage). Returns the prior rcParams to restore, or ``None`` on any
    failure (a font swap must never break an export)."""
    try:
        import matplotlib
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        present = [f for f in stack if f in available]
        if "DejaVu Sans" not in present:
            present.append("DejaVu Sans")
        prev = (list(matplotlib.rcParams.get("font.sans-serif", [])),
                matplotlib.rcParams.get("font.family"))
        matplotlib.rcParams["font.sans-serif"] = present + [f for f in prev[0] if f not in present]
        matplotlib.rcParams["font.family"] = present
        return prev
    except Exception:  # noqa: BLE001
        return None


def _restore_font_family(prev):
    try:
        import matplotlib
        sans, fam = prev
        matplotlib.rcParams["font.sans-serif"] = sans
        matplotlib.rcParams["font.family"] = fam
    except Exception:  # noqa: BLE001
        pass


def style_xy_axes(ax, style, *, title="", xlabel="", ylabel="", width_in=_TYPE_REF_IN,
                  grid_axis="y", title_loc="left"):
    """The shared xy-plot graph-styling rules (DESIGN_SPEC §0 "Graph styling"):
    drop top+right spines, keep left+bottom at weight 1.2, outward short ticks in ``muted``,
    a horizontal-only ``hairline`` grid *behind* the data, and a left-aligned bold title.
    Every analytics ``render_*`` funnels through this so they read identically."""
    ax.set_facecolor(style.page_bg)
    if title:
        ax.set_title(title, color=style.page_fg, fontsize=type_pt("title", width_in),
                     fontweight=type_weight("title"), loc=title_loc, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=style.fg_dim, fontsize=type_pt("axis", width_in),
                      fontweight=type_weight("axis"))
    if ylabel:
        ax.set_ylabel(ylabel, color=style.fg_dim, fontsize=type_pt("axis", width_in),
                      fontweight=type_weight("axis"))
    ax.tick_params(colors=style.fg_dim, labelsize=type_pt("tick", width_in),
                   length=3, width=1.0, direction="out")
    spine_w = getattr(style, "spine_w", 1.2)
    for side, sp in ax.spines.items():
        keep = side in ("left", "bottom")
        sp.set_visible(keep)
        if keep:
            sp.set_color(style.grid)
            sp.set_linewidth(spine_w)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=style.hairline, linewidth=getattr(style, "hairline_w", 1.0))
        ax.grid(False, axis=("x" if grid_axis == "y" else "y"))
    ax.set_axisbelow(True)                  # gridlines behind the data
    return ax


def accent_for_cmap(cmap: str) -> str:
    """A saturated accent colour for a colormap — the colour ~80% up its ramp, so the
    card's spectrum/marker visually belongs to the same palette as the image."""
    try:
        from matplotlib import colormaps
        name = cmap if cmap in colormaps else cmap.lower()
        cm = colormaps[name if name in colormaps else "viridis"]
        r, g, b, _ = cm(0.82)
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    except Exception:  # noqa: BLE001
        return "#5aa9e6"


# --------------------------------------------------------------------------- #
# Low-level figure helpers
# --------------------------------------------------------------------------- #
def _new_figure(width_in, height_in, dpi, facecolor="none"):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor=facecolor)
    FigureCanvasAgg(fig)                 # attach a canvas (OO API, no pyplot)
    return fig


def _corner_xy(corner, w, h, pad):
    """(x0, y0) of a card of size (w, h) tucked into ``corner`` with margin ``pad``,
    all in figure-fraction coordinates."""
    corner = (corner or "lower left").lower().replace("-", " ")
    left = "left" in corner or "west" in corner
    bottom = "lower" in corner or "bottom" in corner or "south" in corner
    x0 = pad if left else 1.0 - pad - w
    y0 = pad if bottom else 1.0 - pad - h
    return x0, y0


def _draw_scalebar(oax, *, pixel_size_um, scale_bar_um, img_w_px, corner, x_extent,
                   ink="#ffffff", stroke=(0, 0, 0, 0.55)):
    """Draw a physical scale bar (a line with short end-ticks and the length
    label centred above it). ``corner`` is the scale bar's own bottom corner — callers place
    it opposite the legend. ``ink``/``stroke`` come from :func:`annotations.pick_ink` so it's
    legible on the image it overlays (this is what fixes the old theme-coupled colour bug).
    ``x_extent`` is the image's horizontal span in figure fractions."""
    sb = annot.layout_scalebar(pixel_size_um=pixel_size_um, scale_bar_um=scale_bar_um,
                               img_w_px=img_w_px, corner=corner, x_extent=x_extent)
    if sb is None:
        return
    # no outline/shadow: the contrast ink from pick_ink carries legibility (matches the
    # live overlay's clean look). ``stroke`` is kept in the signature for callers but unused.
    oax.add_line(_line(oax, [sb.x0, sb.x1], [sb.y, sb.y], ink, 2.8))
    for xt in (sb.x0, sb.x1):                          # short vertical end-ticks
        oax.add_line(_line(oax, [xt, xt], [sb.y - sb.tick_h, sb.y + sb.tick_h], ink, 2.2))
    oax.text(sb.label_xy[0], sb.label_xy[1], sb.label, transform=oax.transAxes,
             color=ink, ha="center", va="bottom", fontsize=sb.font_pt, fontweight="bold")


def _line(ax, xs, ys, color, lw):
    from matplotlib.lines import Line2D
    ln = Line2D(xs, ys, color=color, lw=lw, transform=ax.transAxes,
                solid_capstyle="round")
    return ln


def _overlay_axes(fig, z=15):
    """A transparent full-figure axes in 0..1 coords for drawing cards/scale bars. Its
    zorder is high so cards and text composite *over* the ion image (which is at z≈0)."""
    oax = fig.add_axes([0, 0, 1, 1], zorder=z)
    oax.set_axis_off()
    oax.set_xlim(0, 1)
    oax.set_ylim(0, 1)
    oax.patch.set_alpha(0.0)
    return oax


def _draw_footer_band(fig, frac, style):
    """Fill the bottom ``frac`` of the figure with the theme's solid background colour — the
    footer band that backs the scale bar (bottom-left) and intensity legend (bottom-right),
    so they read cleanly in light or dark mode instead of floating over the tissue. The
    legend/scale bar are drawn afterwards at higher zorder, landing on this band."""
    from matplotlib.patches import Rectangle
    bax = _overlay_axes(fig, z=2)
    bax.add_patch(Rectangle((0, 0), 1.0, float(frac), transform=bax.transData,
                            facecolor=style.page_bg, edgecolor="none", zorder=1))
    return bax


def _add_side_panel(fig, *, x0, style, accent, title, subtitle, mean_spectrum, mz, legend,
                    width_in=7.0):
    """Render the right-hand margin panel: a heading, then either a mean **spectrum** (with
    the active m/z marked) for single-ion views or a swatch **legend** for colour overlays.

    Everything is placed at ``x >= x0`` (the image occupies ``0..x0``), so the panel sits
    *beside* the image and never covers the data. The content block is vertically centred
    in the margin. ``x0`` is the image's right edge in figure fractions."""
    sx0 = x0 + 0.020
    sx1 = 0.982
    sw = sx1 - sx0
    has_spec = mean_spectrum is not None
    has_title = bool(title)
    has_sub = bool(has_title and subtitle)
    has_leg = bool(legend)

    # slot heights (figure fraction); centre the whole block vertically in the margin
    TITLE = 0.046 if has_title else 0.0
    SUB = 0.032 if has_sub else 0.0
    GAP = 0.020 if (has_title and (has_spec or has_leg)) else 0.0
    SPEC = 0.26 if has_spec else 0.0
    XLAB = 0.046 if has_spec else 0.0          # room for the spectrum's m/z axis labels
    row_h = 0.040
    n_leg = min(len(legend), 16) if has_leg else 0
    LEG = row_h * n_leg if has_leg else 0.0
    block_h = TITLE + SUB + GAP + SPEC + XLAB + LEG
    cur = min(0.955, 0.5 + block_h / 2)

    oax = _overlay_axes(fig, z=15)
    if has_title:
        oax.text(sx0, cur, title, transform=fig.transFigure, color=style.fg, ha="left",
                 va="top", fontsize=type_pt("title", width_in), fontweight=type_weight("title"),
                 zorder=12)
        cur -= TITLE
        if has_sub:
            sub = subtitle if len(str(subtitle)) <= 34 else (str(subtitle)[:33] + "…")
            oax.text(sx0, cur, sub, transform=fig.transFigure, color=style.fg_dim,
                     ha="left", va="top", fontsize=type_pt("subtitle", width_in), zorder=12)
            cur -= SUB
        cur -= GAP

    # ---- mean spectrum (the former "spectra overlay", now beside the image) ----
    if has_spec:
        axis, spec = _clip_xy(*mean_spectrum)
        sax = fig.add_axes([sx0, cur - SPEC, sw, SPEC], zorder=16)
        sax.set_facecolor("none")
        if axis.size and spec.size:
            sax.fill_between(axis, spec, color=accent, alpha=0.28, linewidth=0)
            sax.plot(axis, spec, color=accent, linewidth=1.0)
            if mz is not None:
                sax.axvline(float(mz), color=style.fg, linewidth=1.0, alpha=0.85)
                j = int(np.argmin(np.abs(axis - float(mz))))
                sax.plot([axis[j]], [spec[j]], marker="o", markersize=4.5, color=style.fg,
                         markeredgecolor=accent, markeredgewidth=1.0, zorder=6)
            sax.set_ylim(0, float(np.nanmax(spec)) * 1.20 + 1e-9)
            lo, hi = signal_mz_range(axis, spec) or (float(axis.min()), float(axis.max()))
            sax.set_xlim(lo, hi)
        sax.set_yticks([])
        sax.tick_params(axis="x", colors=style.fg_dim, labelsize=type_pt("tick", width_in),
                        length=2)
        for side, sp in sax.spines.items():
            sp.set_visible(side == "bottom")
            if side == "bottom":
                sp.set_color(style.grid); sp.set_linewidth(1.0)
        sax.text(0.0, 1.02, "mean spectrum", transform=sax.transAxes, color=style.fg_dim,
                 ha="left", va="bottom", fontsize=type_pt("annotation", width_in))
        sax.set_xlabel("m/z", color=style.fg_dim, fontsize=type_pt("annotation", width_in),
                       labelpad=2)
        cur -= SPEC + XLAB

    # ---- swatch legend (colour-overlay panels) ----
    if has_leg:
        from matplotlib.patches import Rectangle
        for lab, color in legend[:n_leg]:
            oax.add_patch(Rectangle((sx0, cur - 0.026), 0.024, 0.020, transform=fig.transFigure,
                                    facecolor=color, edgecolor="none", zorder=12))
            oax.text(sx0 + 0.032, cur - 0.016, str(lab), transform=fig.transFigure, color=style.fg,
                     ha="left", va="center", fontsize=type_pt("legend", width_in), zorder=12)
            cur -= row_h


def _checker_rgba(n=4):
    """A small grey checkerboard (transparency indicator) as an (n,n,4) float array."""
    base = (np.indices((n, n)).sum(axis=0) % 2).astype(float)
    g = 0.50 + 0.32 * base                              # two greys
    rgba = np.ones((n, n, 4), float)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
    return rgba


def _draw_legend(fig, x_extent, *, rows, corner, ink="#ffffff", stroke=(0, 0, 0, 0.55),
                 subject=None, percentile_row=False):
    """Render a legend (no card): one row per ion — a right-aligned label, a
    transparency checkerboard, a gradient bar, a trailing relative-max %, and dim lo/hi
    contrast labels under the bar ends. ``rows`` is a list of :class:`annotations.LegendRow`.
    Text sits directly on the image, made legible by the contrast ``ink`` from pick_ink —
    no outline/shadow, matching the live overlay's clean look. (``stroke`` is kept in the
    signature for callers but unused.)

    When ``percentile_row`` is set (overlay channels — DESIGN_SPEC §1), the per-row lo/hi
    labels are replaced by one **shared percentile sub-row** spanning the bars:
    ``{lo}% … contrast percentile … {hi}%``."""
    if not rows:
        return
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    L = annot.layout_legend(rows, corner=corner, x_extent=x_extent, subject=subject)
    oax = _overlay_axes(fig, z=17)
    for g in L.rows:
        if g.cmap is not None:                          # single-ion: the real colormap
            cmap = g.cmap if _cmap_ok(g.cmap) else "viridis"
        else:                                           # overlay channel: black → colour
            cmap = LinearSegmentedColormap.from_list("ch", [(0, 0, 0), to_rgb(g.color or "#fff")])
        cax = fig.add_axes([g.bar.x, g.bar.y, g.bar.w, g.bar.h], zorder=18)
        cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=cmap)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_visible(False)
        if g.checker is not None:
            kax = fig.add_axes([g.checker.x, g.checker.y, g.checker.w, g.checker.h], zorder=18)
            kax.imshow(_checker_rgba(), aspect="auto", interpolation="nearest")
            kax.set_xticks([]); kax.set_yticks([])
            for sp in kax.spines.values():
                sp.set_visible(False)
        oax.text(g.label_xy[0], g.label_xy[1], g.label, transform=fig.transFigure, color=ink,
                 ha="right", va="center", fontsize=L.font_label_pt, fontweight="bold",
                 zorder=19)
        if g.max_xy is not None:
            oax.text(g.max_xy[0], g.max_xy[1], f"{g.max_pct:.0f}%", transform=fig.transFigure,
                     color=ink, ha="left", va="center", fontsize=L.font_small_pt + 0.5,
                     fontweight="bold", zorder=19)
        if not percentile_row and g.lo_xy is not None:
            oax.text(g.lo_xy[0], g.lo_xy[1], f"{g.lo:.0f}%", transform=fig.transFigure, color=ink,
                     ha="left", va="top", fontsize=L.font_small_pt, alpha=0.85, zorder=19)
        if not percentile_row and g.hi_xy is not None:
            oax.text(g.hi_xy[0], g.hi_xy[1], f"{g.hi:.0f}%", transform=fig.transFigure, color=ink,
                     ha="right", va="top", fontsize=L.font_small_pt, alpha=0.85, zorder=19)
    # one shared percentile sub-row centred under the bars (replaces per-row lo/hi labels):
    # "25% · contrast percentile · 90%" — the contrast window the channel ramps were stretched
    # to. A single centred string so it never collides with the labels/bars above it.
    if percentile_row and L.rows:
        los = [g.lo for g in L.rows if g.lo is not None]
        his = [g.hi for g in L.rows if g.hi is not None]
        if los or his:
            bot = L.rows[-1].bar
            lo_v = min(los) if los else None
            hi_v = max(his) if his else None
            parts = [(f"{lo_v:.0f}%" if lo_v is not None else None),
                     "contrast percentile",
                     (f"{hi_v:.0f}%" if hi_v is not None else None)]
            txt = "  ·  ".join(p for p in parts if p)
            cx = (L.block.x + L.block.w / 2.0) if L.block else (bot.x + bot.w / 2.0)
            oax.text(cx, bot.y - 0.006, txt, transform=fig.transFigure, color=ink,
                     ha="center", va="top", fontsize=L.font_small_pt, alpha=0.85, zorder=19)
    if L.overflow:
        oax.text(L.block.x, L.block.y - 0.004, f"+{L.overflow} more", transform=fig.transFigure,
                 color=ink, ha="left", va="top", fontsize=L.font_small_pt, zorder=19)


def _add_intensity_legend(fig, x_extent, *, corner, style, cmap, mz, label, low, high,
                          window, show_title=True, peak=None, ppm=None,
                          label_mode=annot.LABEL_MZ_PPM, ink="#ffffff",
                          stroke=(0, 0, 0, 0.55), subject=None, label_override=None):
    """Single-ion legend: one row using the image's ``cmap`` as the gradient.
    ``peak`` (relative-max %) becomes the trailing readout. ``x_extent`` is the image's
    horizontal span in figure fractions."""
    lo, hi = (window if window is not None else (low, high))
    text = annot.format_ion_label(mz, ppm=ppm, name=(label or ""), override=label_override,
                                  mode=(label_mode if show_title else annot.LABEL_MZ))
    row = annot.LegendRow(label=text, color=None, cmap=cmap, lo=lo, hi=hi,
                          max_pct=(float(peak) if peak else None), checker=False)
    _draw_legend(fig, x_extent, rows=[row], corner=corner, ink=ink, stroke=stroke,
                 subject=subject)


def _cmap_ok(cmap):
    try:
        from matplotlib import colormaps
        return cmap in colormaps or cmap.lower() in colormaps
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Cropping — zoom a panel in on a region of interest (publication-style insets)
# --------------------------------------------------------------------------- #
def _crop_bounds(crop, h, w):
    """Clamp ``crop=(r0, r1, c0, c1)`` (row/col pixel bounds, r1/c1 exclusive) to an
    ``h×w`` image. Returns the clamped tuple, or ``None`` when ``crop`` is empty or
    collapses to nothing."""
    if not crop:
        return None
    r0, r1, c0, c1 = (int(round(float(v))) for v in crop)
    r0, r1 = max(0, min(r0, h)), max(0, min(r1, h))
    c0, c1 = max(0, min(c0, w)), max(0, min(c1, w))
    if r1 - r0 < 1 or c1 - c0 < 1:
        return None
    return r0, r1, c0, c1


def _view_orient(arr):
    """Return a spatial array in the live viewer's on-screen orientation — which is a
    **no-op**, because the app forces ``imageAxisOrder == 'row-major'`` (see
    ``gui/common.py``). With row-major set, the pyqtgraph ImageView displays a ``(rows,
    cols)`` array exactly the way matplotlib ``imshow`` does (axis 0 → vertical, axis 1 →
    horizontal), so the export already matches the viewer.

    This used to ``swapaxes(0, 1)`` on the (wrong) assumption that the ImageView kept
    pyqtgraph's col-major default. That transpose flipped every export's aspect — a wide
    tissue exported as tall (and vice-versa) — relative to the screen. Kept as an identity
    pass-through so the call sites stay self-documenting and nobody re-adds a swap. Always
    crop in ``(rows, cols)`` *before* calling this."""
    return np.asarray(arr)


def _auto_mz_range(spectra):
    """Union m/z window holding signal across every trace, so an exported spectrum
    auto-crops to its peaks (skipping the empty tail past the last real peak) the same
    way the live views do. ``None`` when indeterminate — caller falls back to full axis."""
    los, his = [], []
    for item in spectra or []:
        rng = signal_mz_range(item[1], item[2])
        if rng:
            los.append(rng[0])
            his.append(rng[1])
    return (min(los), max(his)) if los else None


def _spectrum_xlim(spectra, peaks=None, pad_right=50.0):
    """X-limits for an exported spectrum figure: left edge at the first real signal, right
    edge a fixed ``pad_right`` m/z past the **last detected peak** (or, with no peak list,
    the last signal bin) — so the plot keeps a little breathing room past the final peak
    instead of running to the long acquired tail or cropping flush against it. Clamped to
    the data's actual m/z extent. ``None`` when indeterminate (caller keeps the full axis)."""
    base = _auto_mz_range(spectra)
    if base is None:
        return None
    lo, hi = base
    axis_max = None                            # widest acquired axis — never extend past data
    for item in spectra or []:
        ax = np.asarray(item[1], float)
        if ax.size:
            axis_max = float(ax.max()) if axis_max is None else max(axis_max, float(ax.max()))
    last = max(peaks) if (peaks is not None and len(peaks)) else hi
    right = float(last) + float(pad_right)
    right = max(right, hi)                     # never crop the signal we already found
    if axis_max is not None:
        right = min(right, axis_max)
    return (lo, right)


def _clip_xy(x, y):
    """Equal-length float arrays for plotting. A malformed spectrum can arrive with an m/z
    axis and intensity of different lengths (e.g. a dense dataset whose shared axis and
    matrix width disagree); matplotlib's ``plot``/``fill_between`` reject that, so trim both
    to the shorter — a clean line beats a hard crash mid-export."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(x.size, y.size)
    return x[:n], y[:n]


def _ylim_from_range(ymin, ymax, headroom):
    """(ymin, ymax) → matplotlib y-limits: floor at 0 for ordinary spectra, drop below it
    (with headroom) for signed differences. Shared by spectra_ylim and render_spectrum_figure."""
    span = (ymax - ymin) or (ymax or 1.0)
    return ((ymin - span * headroom) if ymin < 0 else 0.0, ymax + span * headroom + 1e-9)


def spectra_ylim(spectra, headroom=0.28):
    """One shared ``(lo, hi)`` y-range over a list of ``(name, axis, y[, color])`` spectra —
    0 (or below, for signed differences) up to the tallest peak across *all* of them, plus
    ``headroom``. Pass it to :func:`render_spectrum_figure` as ``ylim`` so several spectra
    exported as separate files land on one comparable intensity scale."""
    ymax = 0.0
    ymin = 0.0
    for item in spectra or []:
        y = np.asarray(item[2], float)
        if y.size:
            ymax = max(ymax, float(np.nanmax(y)))
            ymin = min(ymin, float(np.nanmin(y)))
    return _ylim_from_range(ymin, ymax, headroom)


def _dim_outside(disp, mask, keep=0.22):
    """Darken the pixels where ``mask`` is False so a cropped panel reads as a close-up of
    just the region (the publication convention of fading the surround). ``disp`` is an
    H×W×3/4 uint8 image; ``mask`` is an H×W boolean."""
    out = np.array(disp, copy=True)
    m = ~np.asarray(mask, bool)
    if m.shape == out.shape[:2]:
        out[m, :3] = (out[m, :3].astype(float) * float(keep)).astype(out.dtype)
    return out


def _ink_for(disp, subject_mask, style=None):
    """Legible ink + stroke sampled from the image *background* (the margin beside the
    subject, where the annotations actually sit) so text/scale-bar contrast on any image.
    Falls back to the whole image when no subject mask is available; when the background is
    transparent (nothing to sample) uses an ink suited to the theme's page colour."""
    default = annot.INK_LIGHT if (style is not None and style.name == "light") else annot.INK_DARK
    d = np.asarray(disp)
    if subject_mask is not None:
        m = np.asarray(subject_mask, bool)
        if m.shape == d.shape[:2] and (~m).any():
            return annot.pick_ink(d[~m], default=default)
    return annot.pick_ink(d, default=default)


# --------------------------------------------------------------------------- #
# Public: ion / overlay panels
# --------------------------------------------------------------------------- #
def render_ion_panel(image=None, *, rgb=None, mz=None, label="", title=None, subtitle=None,
                     cmap="viridis", low=0.0, high=99.0, clip=None, window=None, mean_spectrum=None,
                     pixel_size_um=None, scale_bar_um=None, theme="dark", accent=None,
                     overlay=True, overlay_corner="lower left", show_spectrum=True,
                     show_colorbar=True, show_scalebar=True, show_title=True,
                     width_in=7.0, dpi=200, legend=None, crop=None, outline_mask=None,
                     dim_outside=False, outline_color="#ffffff", outline_width=None,
                     label_mode=annot.LABEL_MZ_PPM, ppm=None, subject_mask=None,
                     auto_corner=True, label_override=None, anchor=None):
    """Render an ion image with a small labelled intensity scale and (when supplied) a
    side-margin spectrum/legend — kept off the data so the image reads cleanly.

    Pass a scalar ``image`` (colour-mapped here, honouring ``low``/``high`` percentile
    contrast) **or** a precomputed ``rgb`` (H×W×3/4 uint8, e.g. a multi-ion overlay).
    The returned ``Figure`` can be handed to :func:`save_figure` in any format.

    ``crop=(r0, r1, c0, c1)`` zooms the panel in on a region of interest. The colormap is
    computed on the **full** image first, so a cropped close-up keeps the very same contrast
    as the full view (a faithful zoom, not a re-stretch). ``outline_mask`` (a full-resolution
    boolean) is traced over the image in ``outline_color`` — the white ROI border seen in
    publications — and ``dim_outside`` fades everything outside it.
    """
    style = make_style(theme, accent)
    acc = style.accent or accent_for_cmap(cmap)
    if outline_width is None:                     # None ⇒ take the design-system outline weight
        outline_width = style.outline_w

    if rgb is not None:
        disp = np.asarray(rgb, np.ubyte)
    else:
        disp = imaging.apply_colormap(np.asarray(image, float), cmap=cmap, low=low, high=high,
                                      clip=clip, anchor=anchor)

    # ROI close-up: fade the surround, then crop both the image and the outline together so
    # the traced boundary stays pixel-aligned with the zoomed data.
    om = np.asarray(outline_mask, bool) if outline_mask is not None else None
    if om is not None and dim_outside:
        disp = _dim_outside(disp, om)
    cb = _crop_bounds(crop, *disp.shape[:2])
    if cb:
        r0, r1, c0, c1 = cb
        disp = disp[r0:r1, c0:c1]
        if om is not None:
            om = om[r0:r1, c0:c1]
    disp = _view_orient(disp)                    # match the live (pyqtgraph) viewer orientation
    if om is not None:
        om = _view_orient(om)
    h, w = disp.shape[:2]

    if title is None:
        title = f"m/z {float(mz):.4f}" if mz is not None else "Ion image"
        if subtitle is None and label:        # lipid name drops to the subtitle line
            subtitle = str(label)

    # The spectrum (single ion) or swatch legend (overlay) goes into a side margin; the
    # image itself stays uncluttered. Reserve that margin only when there's content for it.
    has_spec = bool(overlay and show_spectrum and mean_spectrum is not None)
    has_legend = bool(overlay and legend)
    has_side = has_spec or has_legend

    # Size the image area to the data's aspect so pixels stay square (no stretch), then
    # bound the long side: a tall ROI close-up would otherwise become an absurdly tall
    # figure, and a wide one would be too short for the corner chip/scale bar. Both bounds
    # scale *both* dimensions together, so squareness is preserved either way.
    img_w_in = float(width_in)
    img_h_in = img_w_in * (h / max(w, 1))
    max_h_in = img_w_in * 1.5                   # cap tall crops (shrink the whole figure)
    min_h_in = 2.2                              # floor short/wide crops (grow it — wider figure)
    if img_h_in > max_h_in:
        img_w_in *= max_h_in / img_h_in; img_h_in = max_h_in
    elif img_h_in < min_h_in:
        img_w_in *= min_h_in / img_h_in; img_h_in = min_h_in
    side_w_in = max(2.1, img_w_in * 0.40) if has_side else 0.0
    total_w_in = img_w_in + side_w_in
    img_frac = img_w_in / total_w_in            # image x-span in figure fractions

    # Reserve a bottom footer band (solid themed background) that carries the legend +
    # scale bar, so they sit below the image in a fixed corner — never crushed against a
    # narrow, tissue-filled frame — and stay legible in light or dark mode. The image keeps
    # its true size; the figure grows by the band height beneath it.
    want_legend = bool(overlay and show_colorbar and rgb is None)
    foot = annot.footer_fraction(1 if want_legend else 0, has_scalebar=bool(show_scalebar))
    total_h_in = img_h_in / (1.0 - foot) if foot > 0 else img_h_in

    # Solid themed background (light/dark) whenever annotations are shown, so the image's
    # off-tissue pixels and the footer band read as one seamless backdrop;
    # a bare image with no annotations stays transparent for compositing.
    fig = _new_figure(total_w_in, total_h_in, dpi,
                      facecolor=(style.page_bg if (has_side or foot > 0) else "none"))
    ax = fig.add_axes([0, foot, img_frac, 1.0 - foot])
    ax.imshow(disp, interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    if om is not None and om.any():                 # the ROI boundary, traced over the data
        ax.contour(om.astype(float), levels=[0.5], colors=[outline_color],
                   linewidths=outline_width, antialiased=True)
    if foot > 0:
        _draw_footer_band(fig, foot, style)

    if has_side:
        _add_side_panel(fig, x0=img_frac, style=style, accent=acc,
                        title=(title if show_title else ""), subtitle=subtitle,
                        mean_spectrum=(mean_spectrum if has_spec else None), mz=mz,
                        legend=(legend if has_legend else None), width_in=total_w_in)
        # subtle divider between image and margin (stops above the footer band)
        dax = _overlay_axes(fig, z=14)
        dax.add_line(_line(dax, [img_frac, img_frac], [max(0.05, foot + 0.01), 0.95],
                           style.grid, 1.0))

    # Annotations live in the footer band: legend bottom-right, scale bar bottom-left (the
    # diagonally-opposite corner). Honour an explicit "left" legend choice, else default
    # right. Ink comes from the theme (the band, not the image, is what's behind the text).
    legend_corner = "lower left" if (not auto_corner and "left" in str(overlay_corner or "")) \
        else "lower right"
    ink, stroke = style.fg, (0, 0, 0, 0)

    # small labelled intensity scale, in the footer (single-ion only)
    if want_legend:
        # peak anchor is the hotspot-clip percentile (the 100% reference), not the window's
        # high handle — which is now a relative-intensity %, not a percentile. With an explicit
        # shared ``anchor`` (cross-section contrast) the peak is reported against that instead,
        # so a faint section honestly reads well under 100%.
        if image is None:
            peak = None
        elif anchor is not None and anchor > 0:
            finite = np.asarray(image, float)
            finite = finite[np.isfinite(finite)]
            peak = (100.0 * float(finite.max()) / float(anchor)) if finite.size else None
        else:
            peak = imaging.relative_max(image, clip if clip is not None else high)
        _add_intensity_legend(fig, (0.0, img_frac), corner=legend_corner, style=style,
                              cmap=cmap, mz=mz, label=(label if not has_side else ""),
                              low=low, high=high, window=window, show_title=show_title,
                              peak=peak, ppm=ppm, label_mode=label_mode, ink=ink,
                              stroke=stroke, subject=None, label_override=label_override)

    oax = _overlay_axes(fig)
    if show_scalebar:
        sb_corner = "lower left" if "right" in legend_corner else "lower right"
        _draw_scalebar(oax, pixel_size_um=pixel_size_um, scale_bar_um=scale_bar_um,
                       img_w_px=w, corner=sb_corner, x_extent=(0.0, img_frac),
                       ink=ink, stroke=stroke)
    return fig


def render_cell(image, spec, path):
    """Render ONE ion panel to ``path`` — the process-pool worker for Export Studio's parallel
    render phase. ``spec`` is the picklable output of a main-thread spec builder:
    ``{"kw": <render_ion_panel kwargs minus image>, "dpi": int, "fmt": str, "style": <dict>?}``.

    ``style`` (an optional :class:`~smile_msi.stylespec.StyleSpec` dict) is applied via
    :func:`active_style` so every batch panel shares one design-system look even though each
    renders in its own spawned process. Kept top-level and Qt-free so it pickles **by
    reference** into a spawned worker; the heavy matplotlib import happens in the child. Forces
    the Agg backend so a spawned process never touches a GUI backend. Returns ``path`` on
    success (so the parent can record what was written)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    with active_style(spec.get("style")):          # None ⇒ no-op (app default)
        fig = render_ion_panel(image=image, **dict(spec.get("kw") or {}))
        try:
            save_figure(fig, path, dpi=int(spec.get("dpi", 200)), fmt=spec.get("fmt"))
        finally:
            fig.clear()        # a worker renders many cells — don't let figures accumulate
    return path


def _norm_channel(e):
    """Accept either a rich channel dict ``{mz, color, ppm?, lo?, hi?, label?}`` or a
    legacy ``(label, color)`` swatch tuple, returning a dict."""
    if isinstance(e, dict):
        return e
    lab, color = e
    mz = None
    try:
        import re
        m = re.search(r"[-+]?\d+\.\d+", str(lab))
        mz = float(m.group()) if m else None
    except Exception:  # noqa: BLE001
        mz = None
    return dict(mz=mz, color=color, label=str(lab))


def _add_channel_legend(fig, x_extent, *, entries, style, corner="lower right",
                        ink="#ffffff", stroke=(0, 0, 0, 0.55),
                        label_mode=annot.LABEL_MZ_PPM, subject=None):
    """Per-channel legend (no card): one row per ion — ``{mz}  m/z ± {ppm} ppm``
    label, a transparency checkerboard, a black→colour gradient bar, a trailing relative-max
    ``%``, and dim ``lo``/``hi`` contrast bounds under the bar ends. Each ``entries`` item is
    a dict ``{mz, color, ppm?, lo?, hi?, label?, peak?, label_override?}``."""
    rows = []
    for e in (_norm_channel(x) for x in (entries or [])):
        text = annot.format_ion_label(e.get("mz"), ppm=e.get("ppm"),
                                      name=str(e.get("label") or ""),
                                      override=e.get("label_override"), mode=label_mode)
        rows.append(annot.LegendRow(label=text, color=e.get("color"), cmap=None,
                                    lo=e.get("lo"), hi=e.get("hi"),
                                    max_pct=e.get("peak"), checker=True))
    _draw_legend(fig, x_extent, rows=rows, corner=corner, ink=ink, stroke=stroke,
                 subject=subject, percentile_row=True)


# default ROI outline (DESIGN_SPEC §1): a single-weight magenta polygon, no fill.
DEFAULT_OUTLINE_COLOR = "#c060e8"
AB_A_COLOR = "#DD8452"   # orange = group A (matches the GUI region-A colour)
AB_B_COLOR = "#4C72B0"   # blue   = group B
DEFAULT_OUTLINE_WIDTH = 1.4


def _norm_outline(o, default_color, default_width):
    """Accept ``mask`` / ``(mask, color)`` / ``(mask, color, width)`` / ``{mask, color?, width?}``
    → ``(mask2d, color, width)`` with the caller's defaults filled in."""
    if isinstance(o, dict):
        return o.get("mask"), o.get("color") or default_color, float(o.get("width") or default_width)
    if isinstance(o, (tuple, list)):
        mask = o[0]
        color = o[1] if len(o) > 1 and o[1] else default_color
        width = float(o[2]) if len(o) > 2 and o[2] else default_width
        return mask, color, width
    return o, default_color, default_width


def _draw_corner_label(ax, text, style):
    """A small region label top-left over the image, in ``fg`` with a soft shadow so it reads
    on any tissue (DESIGN_SPEC §1)."""
    if not text:
        return
    import matplotlib.patheffects as pe
    t = ax.text(0.022, 0.972, str(text), transform=ax.transAxes, ha="left", va="top",
                color=style.fg, fontsize=type_pt("subtitle", 7.0),
                fontweight=type_weight("subtitle"), zorder=30)
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=(0, 0, 0, 0.55))])


def render_overlay_panel(rgb, channels=None, *, theme="dark", pixel_size_um=None,
                         scale_bar_um=None, overlay_corner="lower left", show_scalebar=True,
                         show_legend=True, outlines=None, width_in=7.0, dpi=200, crop=None,
                         label_mode=annot.LABEL_MZ_PPM, subject_mask=None, auto_corner=True,
                         outline_color=DEFAULT_OUTLINE_COLOR, outline_width=DEFAULT_OUTLINE_WIDTH,
                         corner_label=None, **_ignore):
    """Additive multi-ion colour composite (H×W×3 uint8) shown **full-frame**.

    ``channels`` is a list of dicts ``{mz, color, ppm?, lo?, hi?, label?, peak?}`` rendered as
    per-channel colour ramps stacked beside the tissue (the scale bar takes the opposite
    corner); a shared **contrast-percentile sub-row** spans the bars. ``outlines`` is an
    optional list of region boundaries traced over the composite — each ``mask`` /
    ``(mask, color)`` / ``(mask, color, width)`` / ``{mask, color?, width?}`` — defaulting to a
    single-weight magenta polygon (``outline_color`` / ``outline_width``), no fill.
    ``corner_label`` draws a small region label top-left over the image. ``crop=(r0, r1, c0,
    c1)`` zooms in. Extra keyword arguments are accepted and ignored so this can share the
    design-kwargs dict with :func:`render_ion_panel`."""
    style = make_style(theme)
    disp = np.asarray(rgb, np.ubyte)
    cb = _crop_bounds(crop, *disp.shape[:2])
    if cb:
        disp = disp[cb[0]:cb[1], cb[2]:cb[3]]
    disp = _view_orient(disp)                    # match the live (pyqtgraph) viewer orientation
    h, w = disp.shape[:2]
    img_h_in = max(2.2, width_in * (h / max(w, 1)))

    # Reserve a bottom footer band (solid themed background) for the channel legend
    # (bottom-right) + scale bar (bottom-left), so they sit below the composite in a fixed
    # corner and stay legible in light or dark mode rather than floating over the tissue.
    n_leg = min(len(channels), annot._MAX_ROWS) if (show_legend and channels) else 0
    foot = annot.footer_fraction(n_leg, has_scalebar=bool(show_scalebar))
    total_h_in = img_h_in / (1.0 - foot) if foot > 0 else img_h_in

    fig = _new_figure(width_in, total_h_in, dpi,
                      facecolor=(style.page_bg if foot > 0 else "none"))
    ax = fig.add_axes([0, foot, 1, 1.0 - foot])
    ax.imshow(disp, interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    if foot > 0:
        _draw_footer_band(fig, foot, style)

    for o in (outlines or []):
        mask2d, color, width = _norm_outline(o, outline_color, outline_width)
        if mask2d is None:
            continue
        m = np.asarray(mask2d, bool)
        if cb:
            m = m[cb[0]:cb[1], cb[2]:cb[3]]
        m = _view_orient(m)                       # keep the ROI trace aligned with the data
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=width,
                       antialiased=True)
    if corner_label:
        _draw_corner_label(ax, corner_label, style)

    # Legend bottom-right, scale bar bottom-left (its diagonal opposite); ink from the theme
    # (the band, not the image, backs the text). Honour an explicit "left" legend choice.
    legend_corner = "lower left" if (not auto_corner and "left" in str(overlay_corner or "")) \
        else "lower right"
    ink, stroke = style.fg, (0, 0, 0, 0)

    if show_legend and channels:
        _add_channel_legend(fig, (0.0, 1.0), entries=channels, style=style, corner=legend_corner,
                            ink=ink, stroke=stroke, label_mode=label_mode, subject=None)
    if show_scalebar:
        sb_corner = "lower left" if "right" in legend_corner else "lower right"
        _draw_scalebar(_overlay_axes(fig), pixel_size_um=pixel_size_um, scale_bar_um=scale_bar_um,
                       img_w_px=w, corner=sb_corner, x_extent=(0.0, 1.0), ink=ink, stroke=stroke)
    return fig


def segmentation_rgba(label_image, colors):
    """Map a (H, W) cluster-label image (NaN = off-tissue) to an RGBA uint8 image using
    ``colors`` (``{cluster_id: hex}``). Unmapped/off-tissue pixels stay transparent."""
    from matplotlib.colors import to_rgb
    label_image = np.asarray(label_image, float)
    h, w = label_image.shape
    rgba = np.zeros((h, w, 4), np.ubyte)
    for cl, hexc in (colors or {}).items():
        r, g, b = to_rgb(hexc)
        m = label_image == cl
        rgba[m] = (int(r * 255), int(g * 255), int(b * 255), 255)
    return rgba


def render_segmentation_figure(label_image, colors, *, legend=None, title="Segmentation",
                               theme="dark", width_in=7.0, dpi=200, pixel_size_um=None,
                               scale_bar_um=None, overlay_corner="lower left", crop=None):
    """A clean segmentation map: clusters/regions in their colours with a swatch legend
    composited into a corner card (so it is self-describing like the ion panels).
    ``crop=(r0, r1, c0, c1)`` zooms the map in on a region."""
    style = make_style(theme)
    rgba = segmentation_rgba(label_image, colors)
    cb = _crop_bounds(crop, *rgba.shape[:2])
    if cb:
        rgba = rgba[cb[0]:cb[1], cb[2]:cb[3]]
    rgba = _view_orient(rgba)                     # match the live (pyqtgraph) viewer orientation
    h, w = rgba.shape[:2]
    height_in = max(2.2, width_in * (h / max(w, 1)))
    fig = _new_figure(width_in, height_in, dpi, facecolor=style.page_bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgba, interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    leg = legend or [(c, f"cluster {cl}") for cl, c in (colors or {}).items()]
    # The figure background is the opaque themed page colour, so off-tissue corners read as
    # that colour — annotate with the theme foreground (legible there) rather than ink sampled
    # from the bright cluster blobs, which left the corner text near-invisible on a dark map.
    ink, stroke = style.fg, (0, 0, 0, 0)
    # no-box title + swatch legend, drawn directly on the map (matches the ion overlay)
    _add_corner_title_legend(fig, title, [(name, color) for color, name in leg],
                             overlay_corner, ink)
    if pixel_size_um and scale_bar_um is None:      # size to the (possibly cropped) map width
        scale_bar_um = annot.nice_scalebar_um(w * float(pixel_size_um))
    if pixel_size_um and scale_bar_um:
        sb_corner = "lower left" if "right" in (overlay_corner or "lower left") else "lower right"
        _draw_scalebar(_overlay_axes(fig), pixel_size_um=pixel_size_um, scale_bar_um=scale_bar_um,
                       img_w_px=w, corner=sb_corner, x_extent=(0.0, 1.0), ink=ink, stroke=stroke)
    return fig


def _image_panel_grid(panels_rgba, titles, *, style, width_in, dpi, ncols=None,
                      suptitle=None, legend=None, pixel_sizes=None):
    """Lay a list of RGBA image panels out in a titled grid — shared by the segmentation grid
    and the ion montage. ``legend`` = list of ``(label, hexcolor)`` swatches under the title.
    ``pixel_sizes`` (optional, per-panel; ``None`` entries skip) draws a physical scale bar in
    each panel that has one — the montage passes it for the first panel only (all panels share
    one dataset), the joint-seg grid passes one per slide."""
    n = max(1, len(panels_rgba))
    if ncols is None:
        ncols = min(n, max(1, int(round(n ** 0.5 * 1.25))))
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols
    cell_w = width_in / ncols
    aspects = [im.shape[0] / max(im.shape[1], 1) for im in panels_rgba] or [1.0]
    asp = min(max(sorted(aspects)[len(aspects) // 2], 0.4), 2.2)   # median aspect, clamped
    head_in = (0.42 if suptitle else 0.0) + (0.24 if legend else 0.0)
    total_h = nrows * cell_w * asp * 1.12 + head_in + 0.1
    fig = _new_figure(width_in, total_h, dpi, facecolor=style.bg)
    top = 1.0 - head_in / total_h
    gs = fig.add_gridspec(nrows, ncols, left=0.015, right=0.985, top=top, bottom=0.02,
                          wspace=0.06, hspace=0.26)
    for i, (im, title) in enumerate(zip(panels_rgba, titles)):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.imshow(im, interpolation="nearest", aspect="equal")
        ax.set_axis_off()
        # per-panel physical scale bar (data coords, so it aligns with the image even though
        # aspect="equal" letterboxes the axes box); auto-sized to this panel's width.
        px_i = pixel_sizes[i] if (pixel_sizes and i < len(pixel_sizes)) else None
        if px_i:
            pw = im.shape[1]
            sb = annot.nice_scalebar_um(pw * float(px_i))
            if sb:
                imaging._draw_scale_bar(ax, im.shape[0], pw, px_i, sb)
        if title:
            ax.set_title(str(title), fontsize=type_pt("annotation", width_in),
                         color=style.fg, pad=2)
    if suptitle:
        fig.text(0.015, 1.0 - 0.5 * (0.42 if suptitle else 0.0) / total_h, suptitle,
                 ha="left", va="center", color=style.fg,
                 fontsize=type_pt("title", width_in), fontweight=type_weight("title"))
    if legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=col, edgecolor="none", label=str(lab)) for lab, col in legend]
        leg = fig.legend(handles=handles, loc="upper left",
                         bbox_to_anchor=(0.015, 1.0 - (0.42 if suptitle else 0.0) / total_h),
                         ncol=min(len(handles), 8), frameon=False,
                         fontsize=type_pt("legend", width_in), handlelength=1.0,
                         columnspacing=1.1, borderaxespad=0.0)
        for t in leg.get_texts():
            t.set_color(style.fg)
    return fig


def render_segmentation_grid(panels, colors, *, legend=None, title="Joint segmentation",
                             theme="dark", width_in=7.5, dpi=200, ncols=None, pixel_sizes=None):
    """A multi-panel **segmentation montage** — one tissue panel per slide, every panel painted
    from the *shared* ``colors`` map so a cluster id means the same tissue in each (the joint-
    segmentation view, as a publication figure rather than a screenshot). ``panels`` is a list of
    ``(name, label_image)``; ``colors`` maps ``{cluster_id: hex}``. Honours the active StyleSpec.
    """
    style = make_style(theme)
    rgba = [_view_orient(segmentation_rgba(lab, colors)) for _name, lab in panels]
    titles = [name for name, _lab in panels]
    leg = legend or [(f"cluster {cl}", c) for cl, c in sorted((colors or {}).items())]
    return _image_panel_grid(rgba, titles, style=style, width_in=width_in, dpi=dpi,
                             ncols=ncols, suptitle=title, legend=leg, pixel_sizes=pixel_sizes)


def render_montage_figure(images, labels, *, cmap="viridis", low=0.0, high=99.0, clip=None,
                          title="Ion montage", theme="dark", width_in=7.5, dpi=200, ncols=None,
                          pixel_size_um=None):
    """A **montage** of ion images (one panel per m/z, shared colormap) as a clean figure — the
    replacement for the on-screen ion-montage screenshot. ``images`` is a list of 2-D arrays,
    ``labels`` the per-panel captions. Each panel is contrast-clipped independently (its own
    ``low``/``high`` percentiles), matching the live montage. Honours the active StyleSpec.
    """
    style = make_style(theme, cmap=cmap)
    rgba = [_view_orient(imaging.apply_colormap(np.asarray(im, float), cmap=cmap,
                                                low=low, high=high, clip=clip)) for im in images]
    # one shared scale bar in the first panel — every m/z panel is the same slide/pixel size
    pixel_sizes = ([pixel_size_um] + [None] * (len(rgba) - 1)) if pixel_size_um else None
    return _image_panel_grid(rgba, list(labels), style=style, width_in=width_in, dpi=dpi,
                             ncols=ncols, suptitle=title, legend=None, pixel_sizes=pixel_sizes)


def render_matrix_figure(image_grid, *, row_labels, col_labels, cmap="viridis", low=0.0,
                         high=100.0, clip=99.0, row_anchors=None, theme="light", dpi=200,
                         title=None, subtitle=None, cell_in=1.9, pixel_size_um=None,
                         scale_bar_um=None):
    """An ion-row × section-column **contact sheet**: every cell is the colour-mapped ion image
    for one (ion, section), with section column headers across the top and ion row labels down
    the left — the matrix reviewers want when comparing a panel of ions across slides.

    ``image_grid[r][c]`` is the 2-D ion image for row ``r`` (ion) / column ``c`` (section), or
    ``None`` for an absent cell (rendered as a faint placeholder). When ``row_anchors[r]`` is
    given, that row's cells share one absolute intensity window (honest cross-section
    comparison via :func:`imaging.apply_colormap`'s ``anchor``); otherwise each cell is
    auto-contrasted. A single shared scale bar is drawn in the first rendered cell.
    """
    style = make_style(theme)
    n_rows = len(image_grid)
    n_cols = len(col_labels)
    if n_rows == 0 or n_cols == 0:
        fig = _new_figure(4.0, 2.0, dpi, facecolor=style.page_bg)
        fig.text(0.5, 0.5, "Nothing to render", color=style.page_fg, ha="center", va="center")
        return fig
    row_anchors = row_anchors or [None] * n_rows

    # geometry: a label gutter on the left + header band on top, then a uniform cell grid sized
    # to the first non-empty cell's aspect so pixels stay square.
    aspect = 1.0                                   # h/w of a cell's image
    for row in image_grid:
        for img in row:
            if img is not None:
                a = _view_orient(np.asarray(img))
                aspect = a.shape[0] / max(a.shape[1], 1)
                break
        else:
            continue
        break
    left_in, top_in, pad_in = 1.15, 0.42, 0.06
    cell_w = cell_in
    cell_h = cell_in * aspect
    fig_w = left_in + n_cols * (cell_w + pad_in) + pad_in
    fig_h = (top_in + n_rows * (cell_h + pad_in) + pad_in + (0.5 if title else 0.0)
             + (0.24 if (subtitle and not title) else 0.0))
    fig = _new_figure(fig_w, fig_h, dpi, facecolor=style.page_bg)

    def fx(x):                                     # inches → figure fraction
        return x / fig_w

    def fy(y):
        return y / fig_h

    if title:
        fig.text(fx(left_in), 1.0 - fy(0.28), str(title), color=style.page_fg,
                 fontsize=type_pt("title", fig_w), fontweight=type_weight("title"),
                 ha="left", va="center")
    if subtitle:                                       # e.g. the shared-contrast reference
        fig.text(fx(left_in), 1.0 - fy(0.50 if title else 0.20), str(subtitle),
                 color=style.page_fg, fontsize=type_pt("tick", fig_w),
                 ha="left", va="center", alpha=0.85)

    # column headers (sections)
    for c, label in enumerate(col_labels):
        cx = left_in + pad_in + c * (cell_w + pad_in) + cell_w / 2
        fig.text(fx(cx), 1.0 - fy((0.5 if title else 0.0) + 0.22), str(label),
                 color=style.page_fg, fontsize=type_pt("subtitle", fig_w),
                 fontweight=type_weight("subtitle"), ha="center", va="center")

    drew_scalebar = False
    for r in range(n_rows):
        # row label (ion), vertically centred on the row
        ry_top = (0.5 if title else 0.0) + top_in + r * (cell_h + pad_in)
        cy = ry_top + cell_h / 2
        fig.text(fx(0.10), 1.0 - fy(cy), str(row_labels[r]), color=style.page_fg,
                 fontsize=type_pt("tick", fig_w), ha="left", va="center", wrap=True)
        for c in range(n_cols):
            img = image_grid[r][c] if c < len(image_grid[r]) else None
            x0 = left_in + pad_in + c * (cell_w + pad_in)
            ax = fig.add_axes([fx(x0), 1.0 - fy(ry_top + cell_h), fx(cell_w), fy(cell_h)])
            ax.set_axis_off()
            if img is None:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor=style.grid, edgecolor="none", alpha=0.18))
                ax.text(0.5, 0.5, "—", color=style.fg_dim, ha="center", va="center",
                        transform=ax.transAxes, fontsize=type_pt("title", fig_w))
                continue
            rgba = imaging.apply_colormap(np.asarray(img, float), cmap=cmap, low=low, high=high,
                                          clip=clip, anchor=row_anchors[r])
            disp = _view_orient(rgba)
            ax.imshow(disp, interpolation="nearest", aspect="auto")
            if not drew_scalebar and pixel_size_um and scale_bar_um:
                _draw_scalebar(_overlay_axes(fig), pixel_size_um=pixel_size_um,
                               scale_bar_um=scale_bar_um, img_w_px=disp.shape[1],
                               corner="lower left", x_extent=(fx(x0), fx(x0 + cell_w)),
                               ink=style.fg, stroke=(0, 0, 0, 0))
                drew_scalebar = True
    return fig


def _add_corner_title_legend(fig, title, entries, corner, ink):
    """A no-box title + swatch→label legend (for segmentation maps), drawn straight on the
    map with the contrast ``ink`` from :func:`_ink_for` — no card background, matching the
    ion overlay's clean look. The colour swatches carry the only fills; text rides on ``ink``."""
    oax = _overlay_axes(fig, z=15)
    from matplotlib.patches import Rectangle
    n = min(len(entries), 14)
    row_h = 0.040
    w = 0.40
    h = 0.052 + row_h * n
    x0, y0 = _corner_xy(corner, w, h, 0.028)
    oax.text(x0, y0 + h - 0.010, title, transform=oax.transData, color=ink,
             ha="left", va="top", fontsize=type_pt("title", 7.0),
             fontweight=type_weight("title"), zorder=3)
    y = y0 + h - 0.052
    for lab, color in entries[:n]:
        oax.add_patch(Rectangle((x0, y - 0.018), 0.028, 0.022, transform=oax.transData,
                                facecolor=color, edgecolor="none", zorder=3))
        oax.text(x0 + 0.040, y - 0.007, str(lab), transform=oax.transData, color=ink,
                 ha="left", va="center", fontsize=type_pt("legend", 7.0),
                 fontweight=type_weight("subtitle"), zorder=3)
        y -= row_h


def _add_bubble_size_legend(fig, rect, *, style, max_imp, min_d, max_d, width_in):
    """Three reference bubbles (low / med / high |SHAP|) in their own axes at ``rect`` (figure
    fractions), using the *same* area↔importance mapping as the matrix so a reader can size
    bubbles honestly (DESIGN_SPEC §4). ``min_d``/``max_d`` are the bubble diameter bounds (pt)."""
    import numpy as np
    lax = fig.add_axes(rect, zorder=5)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_axis_off()
    fracs = (0.15, 0.5, 1.0)                                 # low / med / high
    ys = (0.20, 0.50, 0.80)
    for frac, yy in zip(fracs, ys):
        d = min_d + (max_d - min_d) * np.sqrt(frac)
        lax.scatter([0.22], [yy], s=d * d, facecolors="none",
                    edgecolors=style.muted, linewidths=1.0, zorder=4)
        lax.text(0.46, yy, _fmt_importance(frac * max_imp), ha="left", va="center",
                 color=style.fg_dim, fontsize=type_pt("annotation", width_in))
    lax.text(0.0, 1.02, "|SHAP|", ha="left", va="bottom", color=style.fg_dim,
             fontsize=type_pt("legend", width_in), fontweight=type_weight("subtitle"))
    return lax


def render_shap_figure(importance, direction, *, col_labels, row_mz, row_labels=None,
                       title="SHAP biomarkers", col_title=None, theme="light", dpi=200,
                       cell_in=0.34, max_diam_pt=26.0, cmap="bwr",
                       row_guides=True, size_legend=True, label_rotation=-45):
    """A publication-ready SHAP biomarker bubble plot — the matplotlib twin of the live
    pyqtgraph view. ``importance`` and ``direction`` are ``(n_cols, n_rows)`` aligned to
    ``col_labels`` (columns: regions or donors) and ``row_mz`` (ion rows, strongest first).

    Marker **area** ∝ importance (mean |SHAP|); **colour** = signed direction in [-1, 1]
    (red = high intensity marks the column's region, blue = away), with a diverging colour bar
    and a **bubble size legend** (DESIGN_SPEC §4). ``row_guides`` draws faint hairline guides
    along each ion row; ``label_rotation`` sets the column-label angle (default −45°)."""
    import numpy as np
    from matplotlib import cm, colormaps, colors as mcolors

    style = make_style(theme)
    imp = np.asarray(importance, float)
    drc = np.asarray(direction, float)
    n_cols = len(col_labels)
    n_rows = len(row_mz)
    max_imp = float(imp.max()) if imp.size else 1.0
    max_imp = max_imp or 1.0

    ylabels = (list(row_labels) if row_labels is not None
               else [f"{float(m):.3f}" for m in row_mz])
    xlabels = [str(c) for c in col_labels]
    # Size the canvas in inches so labels never spill past the frame: the left gutter grows
    # with the longest ion label (m/z + lipid annotation can be long), the bottom with the
    # rotated column labels, the right holds the colour bar + bubble size legend.
    rot = float(label_rotation)
    drop = abs(np.sin(np.radians(rot)))                     # how far rotated labels descend
    char_in = 0.5 * type_pt("tick", 5.0) / 72.0
    left_in = 0.25 + char_in * max((len(s) for s in ylabels), default=4)
    bot_in = 0.45 + max(0.35, drop) * char_in * max((len(s) for s in xlabels), default=4)
    top_in = 0.55                                          # title band
    matrix_in = max(1.6, 0.5 * n_cols)
    cbar_in = 1.45                                         # colour bar + size legend + labels
    fig_w = left_in + matrix_in + cbar_in
    fig_h = bot_in + max(1.8, cell_in * n_rows) + top_in
    width_in = fig_w
    fig = _new_figure(fig_w, fig_h, dpi, facecolor=style.page_bg)
    ax = fig.add_axes([left_in / fig_w, bot_in / fig_h,
                       matrix_in / fig_w, (fig_h - bot_in - top_in) / fig_h])
    ax.set_facecolor(style.page_bg)

    cmap = colormaps[cmap if cmap in colormaps else "bwr"]   # cm.get_cmap removed in mpl 3.9
    norm = mcolors.Normalize(-1.0, 1.0)
    # Cap the largest bubble to one grid cell so dense plots don't overlap on export, then
    # scale every bubble down from there (area ∝ importance). The cell is the tighter of the
    # row/column spacing in points; 0.9 leaves a hair of gap between the biggest neighbours.
    matrix_h_in = fig_h - bot_in - top_in
    cell_pt = 72.0 * min(matrix_in / (n_cols + 0.2), matrix_h_in / (n_rows + 0.6))
    max_d = min(max_diam_pt, 0.9 * cell_pt)
    min_d = min(3.0, max_d)                                 # never let the floor exceed the cap
    xs, ys, sizes, cols = [], [], [], []
    for ci in range(n_cols):
        for ri in range(n_rows):
            v = float(imp[ci, ri])
            xs.append(ci)
            ys.append(ri)
            diam = min_d + (max_d - min_d) * np.sqrt(max(v, 0.0) / max_imp)
            sizes.append(diam * diam)                      # scatter s is area in pt²
            cols.append(cmap(norm(float(drc[ci, ri]))))
    # scatter first so it stays collections[0]; row guides are added after but sit *behind* it
    # via zorder (one faint hairline per ion row — DESIGN_SPEC §4).
    ax.scatter(xs, ys, s=sizes, c=cols, edgecolors=(0.16, 0.16, 0.16, 0.55),
               linewidths=0.5, zorder=3)
    if row_guides:
        ax.hlines(range(n_rows), -0.6, n_cols - 0.4, color=style.hairline, linewidth=1.0,
                  zorder=0)

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.invert_yaxis()                                       # strongest ion at the top
    xha = "right" if rot > 0 else ("left" if rot < 0 else "center")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xlabels, rotation=rot, ha=xha, fontsize=type_pt("tick", width_in),
                       color=style.fg_dim)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ylabels, fontsize=type_pt("tick", width_in), color=style.fg_dim)
    if row_labels is None:
        ax.set_ylabel("m/z", fontsize=type_pt("axis", width_in), color=style.fg_dim)
    if col_title:
        ax.set_xlabel(col_title, fontsize=type_pt("axis", width_in), color=style.fg_dim)
    ax.tick_params(colors=style.fg_dim, length=0)
    ax.grid(False)
    for side, sp in ax.spines.items():
        keep = side in ("left", "bottom")
        sp.set_visible(keep)
        if keep:
            sp.set_color(style.grid)
            sp.set_linewidth(1.2)
    ax.set_title(title, fontsize=type_pt("title", width_in), color=style.page_fg,
                 fontweight=type_weight("title"), loc="left", pad=8)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # the right gutter holds the diverging colour bar (upper) + bubble size legend (lower)
    gutter_x = (left_in + matrix_in + 0.18) / fig_w
    cbar_y0 = (bot_in + (0.42 if size_legend else 0.0) * matrix_h_in) / fig_h
    cbar_h = ((0.58 if size_legend else 1.0) * matrix_h_in) / fig_h
    cax = fig.add_axes([gutter_x, cbar_y0, 0.16 / fig_w, cbar_h])
    _style_diverging_cbar(fig, sm, cax, style, "direction  (blue ↓ low · red ↑ high)", width_in)
    if size_legend:
        _add_bubble_size_legend(fig, [gutter_x, bot_in / fig_h, (cbar_in - 0.45) / fig_w,
                                      0.34 * matrix_h_in / fig_h], style=style, max_imp=max_imp,
                                min_d=min_d, max_d=max_d, width_in=width_in)
    return fig


def render_shap_histogram_figure(mz, importance_mean, *, importance_std=None,
                                 direction=None, row_labels=None,
                                 title="SHAP biomarker importance",
                                 value_label="Global SHAP importance score", theme="light", dpi=200,
                                 cmap="bwr", cell_in=0.36, bar_frac=0.74,
                                 orientation="v", show_values=False):
    """A publication-ready per-category biomarker bar chart. ``orientation="v"`` (default,
    after Farrow et al. Fig. S151) lays one **vertical bar per species along the X axis**;
    ``orientation="h"`` re-lays them as **horizontal bars** so the lipid IDs read left-to-right
    (DESIGN_SPEC §3). **Length** = mean SHAP importance (``mean(|SHAP|)``) across donor samples,
    an **error bar** = the SD across donors, and the **bar fill colour** = the mean Spearman
    direction (intensity vs SHAP) on a diverging scale with a colour bar (red = high intensity
    marks this region, blue = away). Species are ordered strongest-first.

    ``mz`` / ``importance_mean`` are aligned 1-D sequences (already top-N, strongest first).
    ``importance_std`` (error bars) and ``direction`` (bar colour) are optional. ``show_values``
    prints each bar's value at its end."""
    import numpy as np
    from matplotlib import cm, colormaps, colors as mcolors

    style = make_style(theme)
    mz = np.asarray(mz, float)
    mean = np.asarray(importance_mean, float)
    n = len(mean)
    std = (np.asarray(importance_std, float) if importance_std is not None else np.zeros(n))
    drc = (np.asarray(direction, float) if direction is not None else None)
    horiz = str(orientation).lower().startswith("h")

    cats = (list(row_labels) if row_labels is not None
            else [f"{float(m):.4f}" for m in mz])
    cmap_obj = colormaps[cmap if cmap in colormaps else "bwr"]
    norm = mcolors.Normalize(-1.0, 1.0)
    cbar_in = 1.3 if drc is not None else 0.25
    top_in = 0.55                                          # title band

    # Manual inch layout. Horizontal: the left gutter grows with the longest lipid ID and the
    # plot height with the species count. Vertical: width grows with the species count and the
    # bottom gutter with the longest (rotated 45°) label.
    char_in = 0.5 * type_pt("tick", 5.0) / 72.0
    max_label = max((len(s) for s in cats), default=4)
    if horiz:
        left_in = 0.35 + char_in * max_label
        bot_in = 0.62                                      # value-axis label + ticks
        plot_w = 3.4
        plot_h = max(1.8, max(0.30, cell_in) * n + 0.4)
    else:
        left_in = 0.92                                     # y-axis label + tick numbers
        bot_in = 0.55 + 0.72 * char_in * max_label
        plot_w = max(2.4, cell_in * n)
        plot_h = 2.9
    fig_w = left_in + plot_w + cbar_in
    fig_h = bot_in + plot_h + top_in
    width_in = fig_w
    fig = _new_figure(fig_w, fig_h, dpi, facecolor=style.page_bg)
    ax = fig.add_axes([left_in / fig_w, bot_in / fig_h, plot_w / fig_w, plot_h / fig_h])

    _draw_shap_histogram_bars(ax, dict(mz=mz, mean=mean, std=std, direction=drc), style=style,
                              cmap_obj=cmap_obj, norm=norm, row_labels=cats,
                              value_label=value_label, bar_frac=bar_frac, orientation=orientation,
                              show_values=show_values, width_in=width_in)
    ax.set_title(title, fontsize=type_pt("title", width_in), color=style.page_fg,
                 fontweight=type_weight("title"), loc="left", pad=8)

    if drc is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cax = fig.add_axes([(left_in + plot_w + 0.22) / fig_w, bot_in / fig_h,
                            0.16 / fig_w, plot_h / fig_h])
        _style_diverging_cbar(fig, sm, cax, style, _SPEARMAN_CBAR_LABEL, width_in)
    return fig


def _style_diverging_cbar(fig, sm, cax, style, label, width_in):
    """A slim diverging colour bar styled to spec: hairline border, muted rotated label, red
    (high) at top → blue (low) at bottom (DESIGN_SPEC §0 "Legends & colorbars")."""
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label, fontsize=type_pt("legend", width_in), color=style.fg_dim)
    cbar.ax.tick_params(colors=style.fg_dim, labelsize=type_pt("annotation", width_in))
    cbar.outline.set_edgecolor(style.grid)
    cbar.outline.set_linewidth(1.0)
    return cbar


# the diverging colour bar's label — spelled out so the Spearman provenance is explicit
# wherever a bar chart is rendered (live figure or the multi-panel "all regions" export).
_SPEARMAN_CBAR_LABEL = ("mean Spearman ρ across donors\n"
                        "ion intensity vs SHAP score  (blue ↓ low · red ↑ high)")


def _fmt_importance(v):
    """Compact label for a bar-end value (DESIGN_SPEC §3 ``show_values``)."""
    a = abs(float(v))
    if a == 0:
        return "0"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def _draw_shap_histogram_bars(ax, bars, *, style, cmap_obj, norm, row_labels=None,
                              value_label="Global SHAP importance score", bar_frac=0.74,
                              orientation="v", show_values=False, width_in=_TYPE_REF_IN):
    """Draw one category's importance bar chart into ``ax``. ``orientation="v"`` (default) lays
    species along the X axis with bar **height** = mean importance; ``"h"`` lays them down the
    Y axis as **horizontal** bars (lipid IDs read left-to-right — DESIGN_SPEC §3). SD shows as
    error bars; bar fill = mean Spearman direction. ``show_values`` prints each bar's value at
    its end (muted, annotation size). Shared by the single-figure and multi-panel renderers."""
    import numpy as np
    mean = np.asarray(bars["mean"], float)
    std = np.asarray(bars.get("std"), float) if bars.get("std") is not None else np.zeros(len(mean))
    drc = np.asarray(bars.get("direction"), float) if bars.get("direction") is not None else None
    n = len(mean)
    cats = (list(row_labels) if row_labels is not None
            else [f"{float(m):.4f}" for m in bars["mz"]])
    idx = np.arange(n)
    colors = ([cmap_obj(norm(float(v))) for v in drc] if drc is not None else style.fg_dim)
    horiz = str(orientation).lower().startswith("h")
    tick_pt = type_pt("tick", width_in)
    ann_pt = type_pt("annotation", width_in)
    has_err = bool(np.any(std > 0))
    pad = (float(np.nanmax(mean)) if mean.size else 1.0) * 0.012 or 0.01
    edge = (0.16, 0.16, 0.16, 0.55)

    if horiz:
        ax.barh(idx, mean, height=bar_frac, color=colors, edgecolor=edge, linewidth=0.5, zorder=3)
        if has_err:
            ax.errorbar(mean, idx, xerr=std, fmt="none", ecolor=style.muted, elinewidth=0.8,
                        capsize=2.5, capthick=0.8, zorder=4)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(left=0)
        ax.invert_yaxis()                                  # strongest species at the top
        ax.set_yticks(idx)
        ax.set_yticklabels(cats, fontsize=tick_pt, color=style.fg_dim)
        ax.set_xlabel(value_label, fontsize=type_pt("axis", width_in), color=style.fg_dim)
        grid_axis = "x"
        if show_values:
            for i, v in enumerate(mean):
                ax.text(v + (std[i] if has_err else 0.0) + pad, i, _fmt_importance(v),
                        ha="left", va="center", fontsize=ann_pt, color=style.fg_dim, zorder=5)
    else:
        ax.bar(idx, mean, width=bar_frac, color=colors, edgecolor=edge, linewidth=0.5, zorder=3)
        if has_err:
            ax.errorbar(idx, mean, yerr=std, fmt="none", ecolor=style.muted, elinewidth=0.8,
                        capsize=2.5, capthick=0.8, zorder=4)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_xticks(idx)
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=tick_pt, color=style.fg_dim)
        ax.set_ylabel(value_label, fontsize=type_pt("axis", width_in), color=style.fg_dim)
        grid_axis = "y"
        if show_values:
            for i, v in enumerate(mean):
                ax.text(i, v + (std[i] if has_err else 0.0) + pad, _fmt_importance(v),
                        ha="center", va="bottom", fontsize=ann_pt, color=style.fg_dim, zorder=5)

    ax.set_facecolor(style.page_bg)
    ax.tick_params(colors=style.fg_dim, length=0, labelsize=tick_pt)
    ax.grid(True, axis=grid_axis, color=style.hairline, linewidth=1.0)
    ax.grid(False, axis=("y" if grid_axis == "x" else "x"))
    ax.set_axisbelow(True)
    for side, sp in ax.spines.items():
        keep = side in ("left", "bottom")
        sp.set_visible(keep)
        if keep:
            sp.set_color(style.grid)
            sp.set_linewidth(1.2)


def render_shap_histogram_grid(items, *, theme="light", dpi=200, cmap="bwr", ncols=None,
                               suptitle=None, value_label="Global SHAP importance score",
                               orientation="v", show_values=False):
    """Render every region's importance histogram at once as one multi-panel figure — the
    "all regions" SHAP export. ``items`` is a list of dicts, one per category::

        {"title": str, "mz": [...], "mean": [...], "std": [...],
         "direction": [...], "row_labels": [...]}

    Each panel is a :func:`render_shap_histogram_figure`-style bar chart (own importance
    scale + ion rows); a single diverging colour bar for the mean Spearman direction is
    shared across panels. Falls back to a one-row empty frame if ``items`` is empty."""
    import numpy as np
    from matplotlib import cm, colormaps, colors as mcolors
    from matplotlib.figure import Figure

    style = make_style(theme)
    items = [it for it in (items or []) if it is not None and len(it.get("mean", [])) > 0]
    n = len(items)
    if ncols is None:
        ncols = 1 if n <= 1 else (2 if n <= 4 else 3)
    ncols = max(1, min(ncols, max(1, n)))
    nrows = int(np.ceil(max(n, 1) / ncols))

    cmap_obj = colormaps[cmap if cmap in colormaps else "bwr"]
    norm = mcolors.Normalize(-1.0, 1.0)

    horiz = str(orientation).lower().startswith("h")
    # size the canvas so each panel has room for its ion labels + bars; constrained layout
    # then divides it evenly (manual inch math doesn't compose cleanly across a grid).
    char_in = 0.5 * type_pt("tick", 5.0) / 72.0
    max_label = max((max((len(s) for s in (it.get("row_labels") or ["x"])), default=6)
                     for it in items), default=6) if items else 6
    max_species = max((len(it["mean"]) for it in items), default=4)
    if horiz:                                               # labels left, bars stacked down
        panel_w = 1.0 + char_in * max_label + 2.2
        panel_h = 0.8 + max(1.8, 0.30 * max_species)
    else:
        panel_w = 0.9 + max(2.0, 0.34 * max_species)        # width grows with the X-axis bars
        panel_h = 2.6 + 0.72 * char_in * max_label          # bottom gutter for rotated labels
    fig_w = ncols * panel_w + 1.5                            # +colour-bar gutter
    fig_h = nrows * panel_h + (0.5 if suptitle else 0.15)
    width_in = panel_w
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=style.page_bg, layout="constrained")
    axes = np.atleast_2d(fig.subplots(nrows, ncols, squeeze=False))

    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.set_visible(False)                            # blank the unused grid cells
            continue
        it = items[k]
        _draw_shap_histogram_bars(ax, it, style=style, cmap_obj=cmap_obj, norm=norm,
                                  row_labels=it.get("row_labels"), value_label=value_label,
                                  orientation=orientation, show_values=show_values,
                                  width_in=width_in)
        ax.set_title(str(it.get("title", "")), fontsize=type_pt("subtitle", width_in),
                     color=style.page_fg, fontweight=type_weight("title"), loc="left", pad=6)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    cbar.set_label(_SPEARMAN_CBAR_LABEL, fontsize=type_pt("legend", width_in), color=style.fg_dim)
    cbar.ax.tick_params(colors=style.fg_dim, labelsize=type_pt("annotation", width_in))
    cbar.outline.set_edgecolor(style.grid)
    cbar.outline.set_linewidth(1.0)
    if suptitle:
        fig.suptitle(suptitle, fontsize=type_pt("title", fig_w), color=style.page_fg,
                     fontweight=type_weight("title"))
    return fig


# --------------------------------------------------------------------------- #
# Public: spectra
# --------------------------------------------------------------------------- #
def render_spectrum_figure(spectra, *, peaks=None, active_mz=None, title="Mass spectrum",
                           xlabel="m/z", ylabel="intensity", theme="dark", accent="#5ad1c4",
                           width_in=9.0, height_in=3.4, dpi=200, fill=True, mz_range=None,
                           pad_right=50.0, headroom=0.28, draw_style="line", diff_labels=None,
                           ylim=None, annotate_top=5, show_active=True, caption=None,
                           k_ticks=True):
    """A clean publication spectrum. ``spectra`` is a list of ``(name, axis, y[, color])``;
    optional ``peaks`` (m/z list) are ticked just *below* the x-axis (pass ``peaks=None`` to
    omit them) and ``active_mz`` is marked.

    ``mz_range`` forces an exact ``(lo, hi)`` x-window; left ``None`` the axis auto-crops to
    the first peak and stops ``pad_right`` m/z past the last one. ``headroom`` is the fraction
    of empty space left above the tallest peak (so it — and the legend — aren't flush to the
    top frame).

    ``draw_style`` is ``"line"`` (a filled profile trace, the default) or ``"sticks"`` (a
    centroid-style impulse plot — a thin vertical line per m/z rising from the baseline, the
    classic stacked-spectra look). ``diff_labels`` — when the spectrum is a signed difference
    ``mean(A) − mean(B)``, pass ``(a_name, b_name)`` to draw a zero baseline and label which
    direction belongs to which region (``↑ higher in A`` / ``↓ higher in B``).

    ``annotate_top`` labels the N tallest peaks of the primary trace with their m/z (centred
    above the apex, overlaps suppressed); set ``0`` to disable. ``show_active`` draws the
    ``active_mz`` stick full-height in ``accent`` with a filled apex dot. ``k_ticks`` formats
    intensity ticks compactly (5k / 1.2M). ``caption`` adds a muted footer note."""
    import matplotlib.transforms as mtransforms
    style = make_style(theme, accent)
    fig = _new_figure(width_in, height_in, dpi, facecolor=style.page_bg)
    # left margin (0.105) leaves room for the y tick labels + axis label so big intensity
    # counts can't overrun the frame and clip off the left edge.
    ax = fig.add_axes([0.105, (0.235 if caption else 0.17), 0.87, (0.69 if caption else 0.74)])
    ax.set_facecolor(style.page_bg)
    palette = list(style.category_palette)
    sticks = str(draw_style).lower() in ("sticks", "stick", "centroid")
    single = len(spectra or []) == 1
    ymax = 0.0
    ymin = 0.0                                  # tracks below-zero excursions (difference spectra)
    primary = None                              # (axis, y) of the first trace, for peak labelling
    for i, item in enumerate(spectra or []):
        name = item[0]
        axis, y = _clip_xy(item[1], item[2])
        explicit = len(item) > 3 and item[3]
        # a lone stick spectrum draws in the foreground ink (DESIGN_SPEC §2); overlays cycle
        # the category palette so each trace stays distinct.
        color = item[3] if explicit else ((style.page_fg if (sticks and single) else palette[i % len(palette)]))
        if i == 0:
            primary = (axis, y)
        if sticks:                              # centroid-style impulses from the baseline
            ax.vlines(axis, 0.0, y, color=color, linewidth=1.0, label=str(name))
        else:
            if fill and i == 0:
                ax.fill_between(axis, y, color=color, alpha=0.18, linewidth=0)
            ax.plot(axis, y, color=color, linewidth=1.1, label=str(name))
        if y.size:
            ymax = max(ymax, float(np.nanmax(y)))
            ymin = min(ymin, float(np.nanmin(y)))
    if mz_range is None:                       # auto-crop to the peaks unless told otherwise
        mz_range = _spectrum_xlim(spectra, peaks, pad_right)
    if mz_range:
        ax.set_xlim(*mz_range)
    if ylim is not None:                       # caller-forced range (e.g. one shared scale)
        ax.set_ylim(*ylim)
    else:
        # floor at 0 for ordinary spectra; drop below it (with headroom) for signed differences
        ax.set_ylim(*_ylim_from_range(ymin, ymax, headroom))
    # active-ion marker: the selected stick drawn full height in the accent, with a filled
    # accent dot at its apex on the primary trace (DESIGN_SPEC §2).
    acc = style.accent or accent
    if active_mz is not None and show_active:
        ax.axvline(float(active_mz), color=acc, linewidth=1.4, alpha=0.95, zorder=5)
        if primary is not None and primary[0].size:
            pax, py = primary
            j = int(np.argmin(np.abs(pax - float(active_mz))))
            ax.plot([pax[j]], [py[j]], marker="o", markersize=5.0, color=acc,
                    markeredgecolor=style.page_bg, markeredgewidth=0.8, zorder=6, clip_on=False)
    elif active_mz is not None:
        ax.axvline(float(active_mz), color=style.page_fg, linewidth=1.0, alpha=0.6)
    # compact intensity ticks: "5k" / "1.2M" so counts stay narrow and never push the y-label
    # off the left of the figure. Falls back to a shared ×10ⁿ multiplier when k-ticks are off.
    if k_ticks:
        ax.yaxis.set_major_formatter(_k_tick_formatter())
    else:
        try:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            ot = ax.yaxis.get_offset_text()
            ot.set_color(style.fg_dim); ot.set_fontsize(type_pt("annotation", width_in))
        except Exception:  # noqa: BLE001 — an axis without a ScalarFormatter keeps its own ticks
            pass
    # top-N peak labels on the primary trace: m/z centred just above each apex, overlaps culled.
    if annotate_top and primary is not None and primary[0].size and not diff_labels:
        for mz_pk, h_pk in _top_peak_labels(primary[0], primary[1], int(annotate_top),
                                            xlim=ax.get_xlim()):
            ax.annotate(f"{mz_pk:.2f}", xy=(mz_pk, h_pk), xytext=(0, 2.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=type_pt("annotation", width_in), color=style.page_fg,
                        zorder=7, clip_on=False)
    if diff_labels:                            # a declared signed difference: always say which
        a_name, b_name = str(diff_labels[0]), str(diff_labels[1])
        # zero baseline at the data origin (the actual A = B reference); the direction labels are
        # pinned to the top/bottom corners so they stay legible no matter where 0 lands — a
        # strictly-positive difference (0 at the floor) or a small difference overlaid on a tall
        # spectrum (0 crushed low). The arrows + the baseline carry the above/below meaning.
        ax.axhline(0.0, color=style.page_fg, linewidth=0.8, alpha=0.55, zorder=1)
        bbox = dict(facecolor=style.page_bg, edgecolor="none", alpha=0.7, pad=1.5)
        ax.text(0.012, 0.965, f"↑ higher in {a_name}", transform=ax.transAxes, ha="left",
                va="top", fontsize=type_pt("annotation", width_in), color=style.page_fg,
                zorder=6, bbox=bbox)
        ax.text(0.012, 0.035, f"↓ higher in {b_name}", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=type_pt("annotation", width_in), color=style.page_fg,
                zorder=6, bbox=bbox)
        if ylabel == "intensity":
            ylabel = "Δ intensity"
    if peaks is not None and len(peaks):
        # tick the peaks just beneath the x-axis (x in data coords, y in axes fraction), pointing
        # up at the baseline; clip_on=False lets them sit in the margin below the plotted area.
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot(np.asarray(peaks, float), np.full(len(peaks), -0.035), transform=trans,
                linestyle="none", marker="^", markersize=5, color=acc, alpha=0.75,
                clip_on=False, label="peaks")
    _style_axes(ax, style, title, xlabel, ylabel, width_in=width_in)
    if spectra and len(spectra) > 1:
        leg = ax.legend(loc="upper right", fontsize=type_pt("legend", width_in), frameon=False)
        for t in leg.get_texts():
            t.set_color(style.page_fg)
    if caption:                                # optional muted footer note ("n = 1 ROI · …")
        fig.text(0.105, 0.045, str(caption), ha="left", va="bottom", color=style.fg_dim,
                 fontsize=type_pt("annotation", width_in))
    return fig


def _style_axes(ax, style, title="", xlabel="", ylabel="", width_in=_TYPE_REF_IN):
    """Spectrum/xy axis styling — delegates to the shared :func:`style_xy_axes` token rules."""
    style_xy_axes(ax, style, title=title, xlabel=xlabel, ylabel=ylabel,
                  width_in=width_in, grid_axis="y")


def _k_tick_formatter():
    """A compact intensity tick formatter — ``5000`` → ``5k``, ``1.2e6`` → ``1.2M`` — so
    intensity counts stay narrow and never push the axis label off the figure (DESIGN_SPEC §2)."""
    from matplotlib.ticker import FuncFormatter

    def _fmt(v, _pos):
        a = abs(v)
        if a >= 1e6:
            return f"{v / 1e6:g}M"
        if a >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"
    return FuncFormatter(_fmt)


def _top_peak_labels(axis, y, n, *, xlim, min_sep_frac=0.05):
    """The ``n`` tallest local maxima of ``(axis, y)``, greedily de-overlapped: sweep peaks
    high→low and keep one only if it sits ≥ ``min_sep_frac`` of the x-span from every kept
    peak. Returns ``[(mz, height), …]`` for top-N peak labelling (DESIGN_SPEC §2)."""
    axis = np.asarray(axis, float)
    y = np.asarray(y, float)
    if axis.size < 3 or n <= 0:
        return []
    # local maxima (strictly greater than both neighbours)
    interior = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if interior.size == 0:
        interior = np.array([int(np.argmax(y))])
    order = interior[np.argsort(y[interior])[::-1]]
    span = (float(xlim[1]) - float(xlim[0])) if xlim else (float(axis.max() - axis.min()) or 1.0)
    sep = abs(span) * float(min_sep_frac)
    kept = []
    for i in order:
        mz = float(axis[i])
        if any(abs(mz - m) < sep for m, _ in kept):
            continue
        kept.append((mz, float(y[i])))
        if len(kept) >= n:
            break
    return kept


def _scale_matrix(M, scale):
    """Min-max scale ``M`` to [0,1] along ``scale`` (``"col"`` per feature, ``"row"`` per sample,
    ``"none"`` global), NaNs → 0 (absent → floor). Mirrors the on-screen cohort heatmap so the
    exported figure matches the live view."""
    M = np.asarray(M, float)
    if scale == "none":
        finite = M[np.isfinite(M)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        out = (M - lo) / (hi - lo) if hi > lo else np.zeros_like(M)
        return np.nan_to_num(out, nan=0.0)
    axis = 0 if scale == "col" else 1
    out = np.full_like(M, np.nan)
    it = range(M.shape[1]) if axis == 0 else range(M.shape[0])
    for k in it:
        vec = M[:, k] if axis == 0 else M[k, :]
        ok = np.isfinite(vec)
        if not ok.any():
            sl = np.zeros_like(vec)
        else:
            lo, hi = float(np.nanmin(vec)), float(np.nanmax(vec))
            sl = (vec - lo) / (hi - lo) if hi > lo else ok.astype(float)
        if axis == 0:
            out[:, k] = sl
        else:
            out[k, :] = sl
    return np.nan_to_num(out, nan=0.0)


def render_heatmap_figure(matrix, *, row_labels=None, col_labels=None, cmap="viridis",
                          scale="col", group_labels=None, group_colors=None,
                          title="Sample × feature heatmap", xlabel="feature m/z", ylabel="",
                          theme="light", width_in=7.5, height_in=None, dpi=200,
                          colorbar_label="scaled intensity", max_col_ticks=24, caption=None):
    """A publication **heatmap** of a scalar ``matrix`` (rows = samples, cols = features),
    rendered as a clean :class:`~matplotlib.figure.Figure` (not a screenshot). ``scale`` sets
    the per-``"col"``/``"row"``/``"none"`` min-max normalisation used for display (default
    per-feature, matching the on-screen cohort heatmap). Honours the active
    :class:`~smile_msi.stylespec.StyleSpec`.

    ``group_labels`` (one per row) draws a thin colour strip down the left edge + a legend, so
    a sample × feature map shows which group each row belongs to. ``group_colors`` maps a group
    name → hex (defaults to the category palette). Column ticks are thinned to ``max_col_ticks``
    so a wide feature axis stays readable.
    """
    M = np.atleast_2d(np.asarray(matrix, float))
    nrow, ncol = M.shape
    disp = _scale_matrix(M, scale)
    style = make_style(theme, cmap=cmap)

    if height_in is None:                             # grow with the number of samples
        height_in = min(9.5, max(2.6, 0.22 * nrow + 1.9))

    has_grp = group_labels is not None and len(group_labels) == nrow
    groups = list(dict.fromkeys(group_labels)) if has_grp else []
    if has_grp:
        if group_colors:
            gcol = {g: group_colors.get(g, style.category_palette[i % len(style.category_palette)])
                    for i, g in enumerate(groups)}
        else:
            gcol = {g: style.category_palette[i % len(style.category_palette)]
                    for i, g in enumerate(groups)}

    fig = _new_figure(width_in, height_in, dpi, facecolor=style.bg)
    # layout fractions: [group strip][heatmap][colorbar]
    left = 0.055 + (0.14 if row_labels is not None else 0.0)
    strip_w = 0.018 if has_grp else 0.0
    cbar_w = 0.022
    gap = 0.012
    top = 0.90 if title else 0.965
    bot = 0.20 if (col_labels is not None or caption) else 0.09
    hm_x = left + strip_w + (gap if has_grp else 0.0)
    hm_w = 0.965 - cbar_w - gap - hm_x
    ax = fig.add_axes([hm_x, bot, hm_w, top - bot])
    im = ax.imshow(disp, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest", origin="upper")

    ax.set_yticks([])                             # sample labels are placed below (left of any strip)
    # col labels (features) — thinned
    if col_labels is not None and ncol:
        cstep = max(1, ncol // max(1, max_col_ticks))
        idx = list(range(0, ncol, cstep))
        ax.set_xticks(idx)
        ax.set_xticklabels([str(col_labels[i]) for i in idx], rotation=90,
                           fontsize=type_pt("tick", width_in))
    else:
        ax.set_xticks([])
    ax.tick_params(colors=style.muted, length=2)
    for sp in ax.spines.values():
        sp.set_color(style.hairline)
    if xlabel:
        ax.set_xlabel(xlabel, color=style.muted, fontsize=type_pt("axis", width_in))
    if ylabel:
        ax.set_ylabel(ylabel, color=style.muted, fontsize=type_pt("axis", width_in))

    # group colour strip on its own axes (left of the heatmap) + a legend
    label_ax = ax
    if has_grp:
        sax = fig.add_axes([left, bot, strip_w, top - bot])
        strip = np.array([[_hex_to_rgb(gcol[group_labels[r]])] for r in range(nrow)])
        sax.imshow(strip, aspect="auto", interpolation="nearest", origin="upper")
        sax.set_xticks([])
        for sp in sax.spines.values():
            sp.set_visible(False)
        label_ax = sax                            # sample labels go left of the strip, not under it
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=gcol[g], edgecolor="none", label=str(g)) for g in groups]
        leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
                        ncol=min(len(groups), 4), frameon=False,
                        fontsize=type_pt("legend", width_in), handlelength=1.0,
                        columnspacing=1.2, borderaxespad=0.0)
        for t in leg.get_texts():
            t.set_color(style.fg)

    # sample (row) labels on the leftmost axes — the strip when present, else the heatmap
    if row_labels is not None:
        rstep = max(1, nrow // 40)
        ticks = list(range(0, nrow, rstep))
        label_ax.set_yticks(ticks)
        label_ax.set_yticklabels([str(row_labels[i]) for i in ticks],
                                 fontsize=type_pt("tick", width_in))
        label_ax.tick_params(axis="y", colors=style.muted, length=0)

    # colorbar
    cax = fig.add_axes([0.965 - cbar_w, bot, cbar_w, top - bot])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_edgecolor(style.hairline)
    cax.tick_params(colors=style.muted, labelsize=type_pt("tick", width_in), length=2)
    cb.set_label(colorbar_label, color=style.muted, fontsize=type_pt("axis", width_in))

    if title:
        fig.text(left, 0.955, title, color=style.fg, ha="left", va="center",
                 fontsize=type_pt("title", width_in), fontweight=type_weight("title"))
    if caption:
        fig.text(left, 0.03, caption, color=style.muted, ha="left", va="center",
                 fontsize=type_pt("annotation", width_in))
    return fig


def _hex_to_rgb(c):
    """Hex/`#rrggbb` (or already-RGB tuple) → an (r,g,b) float triple in [0,1]."""
    if isinstance(c, (tuple, list)):
        return tuple(float(x) for x in c[:3])
    c = str(c).lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def render_volcano_figure(log2_fc, p_value, q_value, *, mz=None, labels=None,
                          a_label="A", b_label="B", title="Differential abundance",
                          subtitle=None, q_sig=0.05, fc_line=None, annotate_top=8,
                          annotate_by="mz", theme="light", width_in=6.4, height_in=5.2,
                          dpi=200, a_color=None, b_color=None, caption=None):
    """A publication **volcano plot** — effect size (log₂ fold-change, B/A) vs significance
    (−log₁₀ p) — rendered as a clean :class:`~matplotlib.figure.Figure` (not a screenshot of the
    live plot). Honours the active :class:`~smile_msi.stylespec.StyleSpec` (fonts / weights /
    theme) like every other ``render_*``.

    Colour encodes **direction** (orange = higher in ``a_label``, blue = higher in ``b_label``);
    **significance** (``q_value ≤ q_sig``, Benjamini–Hochberg FDR) is shown by emphasis —
    significant ions are opaque, larger and ringed, the rest keep the hue but fade back — so a
    small cohort where nothing clears FDR still reads as data, not a broken all-grey plot (the
    same honest encoding as the on-screen view). A dashed line marks the exact FDR p-threshold
    (the largest p among the q≤``q_sig`` hits); when nothing is significant it falls back to a
    faint uncorrected p=0.05 reference.

    ``annotate_top`` labels the N most-significant ions (by q, then |fold-change|) with their m/z
    (``annotate_by="mz"``) or lipid name (``annotate_by="label"``, needs ``labels``), overlap-
    suppressed. ``fc_line`` (if set) draws fold-change guide verticals at ±``fc_line``.
    """
    fc = np.asarray(log2_fc, float)
    p = np.asarray(p_value, float)
    q = np.asarray(q_value, float)
    n = min(len(fc), len(p), len(q))
    fc, p, q = fc[:n], p[:n], q[:n]
    mz = np.asarray(mz, float)[:n] if mz is not None else np.full(n, np.nan)
    labels = list(labels)[:n] if labels is not None else [None] * n

    ok = np.isfinite(fc) & np.isfinite(p)                 # tested features only
    fc, p, q, mz = fc[ok], p[ok], q[ok], mz[ok]
    labels = [labels[i] for i in range(len(ok)) if ok[i]]
    y = -np.log10(np.clip(p, 1e-12, 1.0))
    sig = np.isfinite(q) & (q <= q_sig)

    style = make_style(theme)
    a_col = a_color or AB_A_COLOR                         # orange = higher in A
    b_col = b_color or AB_B_COLOR                         # blue   = higher in B

    fig = _new_figure(width_in, height_in, dpi, facecolor=style.bg)
    top = 0.90 if (subtitle or title) else 0.96
    bot = 0.19 if caption else 0.16
    ax = fig.add_axes([0.12, bot, 0.85, top - bot])
    style_xy_axes(ax, style, xlabel="log₂ fold-change (%s / %s)" % (b_label, a_label),
                  ylabel="−log₁₀ p", width_in=width_in, grid_axis="y",
                  title="")

    # reference guides: x=0 (no change), the FDR p-threshold (dashed), optional ±fc_line
    ax.axvline(0.0, color=style.grid, lw=getattr(style, "spine_w", 1.2) * 0.7, zorder=1)
    if sig.any():
        p_thr = float(np.nanmax(p[sig]))
        ax.axhline(-np.log10(max(p_thr, 1e-12)), color=style.muted,
                   ls="--", lw=1.0, zorder=1)
    else:
        ax.axhline(-np.log10(q_sig), color=style.hairline, ls=":", lw=0.9, zorder=1)
    if fc_line:
        for xv in (-abs(fc_line), abs(fc_line)):
            ax.axvline(xv, color=style.hairline, ls=":", lw=0.9, zorder=1)

    # points: colour by direction, emphasis by significance
    up = fc > 0
    for mask, base in ((~up, a_col), (up, b_col)):
        for s, alpha, size, ring in ((mask & ~sig, 0.42, 16, None),
                                     (mask & sig, 0.95, 34, style.bg)):
            if s.any():
                ax.scatter(fc[s], y[s], s=size, c=base, alpha=alpha, linewidths=0.6,
                           edgecolors=(ring if ring else "none"), zorder=3)

    # annotate the most-significant ions
    ntop = int(annotate_top or 0)
    if ntop > 0 and len(fc):
        order = np.lexsort((-np.abs(fc), np.where(np.isfinite(q), q, np.inf)))
        chosen, placed = order[:ntop], []
        xmax = float(np.nanmax(np.abs(fc))) or 1.0
        for i in chosen:
            if annotate_by == "label" and labels[i]:
                txt = str(labels[i])
            elif np.isfinite(mz[i]):
                txt = f"{mz[i]:.4f}"
            else:
                continue
            xi, yi = float(fc[i]), float(y[i])
            if any(abs(xi - px) < 0.45 and abs(yi - py) < 0.30 for px, py in placed):
                continue                                  # crude overlap suppression
            placed.append((xi, yi))
            right = xi > 0.55 * xmax                      # near the right edge → label leftwards
            ax.annotate(txt, (xi, yi), xytext=((-4 if right else 4), 4),
                        textcoords="offset points", ha=("right" if right else "left"),
                        fontsize=type_pt("annotation", width_in), color=style.fg,
                        zorder=4, annotation_clip=False)

    # title / subtitle (left-aligned, above the axes) + direction hint + caption
    if title:
        fig.text(0.12, 0.955, title, color=style.fg, ha="left", va="center",
                 fontsize=type_pt("title", width_in), fontweight=type_weight("title"))
    sub = subtitle if subtitle is not None else f"{a_label}  vs  {b_label}"
    if sub:
        fig.text(0.12, 0.918, sub, color=style.muted, ha="left", va="center",
                 fontsize=type_pt("subtitle", width_in))
    # tiny direction key under the x-axis extremes
    ax.text(0.0, -0.14, f"← higher in {a_label}", transform=ax.transAxes, ha="left",
            va="top", color=a_col, fontsize=type_pt("annotation", width_in),
            fontweight="bold")
    ax.text(1.0, -0.14, f"higher in {b_label} →", transform=ax.transAxes, ha="right",
            va="top", color=b_col, fontsize=type_pt("annotation", width_in), fontweight="bold")
    if caption:
        fig.text(0.12, 0.03, caption, color=style.muted, ha="left", va="center",
                 fontsize=type_pt("annotation", width_in))
    return fig


# --------------------------------------------------------------------------- #
# Public: saving figures in any format
# --------------------------------------------------------------------------- #
def save_figure(fig, path, dpi=200, fmt=None, transparent=False):
    """Write ``fig`` to ``path``; the format follows the extension unless ``fmt`` is set.
    Supports PNG/TIFF/JPEG/WebP (raster) and PDF/SVG/EPS (vector)."""
    fmt = (fmt or os.path.splitext(str(path))[1].lstrip(".") or "png").lower()
    if fmt not in IMAGE_FORMATS:
        fmt = "png"
    # Guarantee the filename actually carries the format's extension. matplotlib writes the
    # bytes in ``fmt`` regardless of ``path``, so a name whose trailing ".xxx" isn't a real
    # image format — e.g. an ion label like "947.546PG46", where os.path.splitext reads the
    # m/z's decimal as the extension ".546PG46" — would otherwise land on disk typed
    # ".546pg46". Append the real extension unless it's already there. (Only for string
    # paths; file-like targets, e.g. BytesIO, carry no name and are left untouched.)
    if isinstance(path, str) and os.path.splitext(path)[1].lower().lstrip(".") != fmt:
        path = path + "." + fmt
    save_kw = dict(dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    if fmt in ("jpg", "jpeg", "tif", "tiff", "webp"):
        # Pillow path: no alpha, give it a real background so the glass card composites.
        fig.savefig(path, format=("jpeg" if fmt == "jpg" else "tiff" if fmt == "tif" else fmt),
                    pil_kwargs=({"compression": "tiff_lzw"} if fmt in ("tif", "tiff") else None),
                    **{**save_kw, "facecolor": _opaque(fig.get_facecolor())})
    elif fmt in VECTOR_FORMATS:
        fig.savefig(path, format=fmt, transparent=transparent, **{k: v for k, v in save_kw.items() if k != "dpi"} | {"dpi": dpi})
    else:
        fig.savefig(path, format="png", transparent=transparent, **save_kw)
    _close(fig)
    return path


def _opaque(rgba):
    """Drop alpha for formats that don't support it (JPEG/TIFF) — white if the figure
    was transparent."""
    try:
        r, g, b, a = rgba
        if a < 1.0:
            return (1, 1, 1)
        return (r, g, b)
    except Exception:  # noqa: BLE001
        return "white"


def _close(fig):
    try:
        fig.clf()
    except Exception:  # noqa: BLE001
        pass


def quick_export_image(image=None, path="ion.png", *, dpi=200, **panel_kwargs):
    """Render an ion panel and save it in one call (convenience for the GUI)."""
    rgb = panel_kwargs.pop("rgb", None)
    fig = render_ion_panel(image, rgb=rgb, dpi=dpi, **panel_kwargs)
    return save_figure(fig, path, dpi=dpi)


# --------------------------------------------------------------------------- #
# Public: tables & spectra data
# --------------------------------------------------------------------------- #
def write_table(df, path, fmt=None, header_lines=None):
    """Write a pandas DataFrame as CSV/TSV/XLSX/JSON/Markdown/Parquet (by extension or
    explicit ``fmt``). CSV/TSV use UTF-8-sig so Excel reads unicode lipid names.

    ``header_lines`` is an optional audit-trail block (a list of pre-formatted ``# …``
    comment lines from :meth:`provenance.Provenance.csv_header_lines`): for CSV/TSV it is
    prepended verbatim (standard ``comment='#'`` parsers skip it); for XLSX it becomes a
    separate ``Provenance`` sheet. Other formats ignore it (the bundle's ``methods.md``
    and the PDF book carry the full record for those)."""
    fmt = (fmt or os.path.splitext(str(path))[1].lstrip(".") or "csv").lower()
    if fmt == "tsv":
        _write_delimited(df, path, "\t", header_lines)
    elif fmt == "xlsx":
        _write_xlsx(df, path, header_lines)
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2)
    elif fmt in ("md", "markdown"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(_to_markdown_table(df))
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        _write_delimited(df, path, ",", header_lines)
    return path


def _write_delimited(df, path, sep, header_lines):
    """Write a CSV/TSV, optionally prepending a ``# …`` audit-trail header. Keeps the
    UTF-8-sig BOM (Excel) and writes both header + table through one handle so the BOM is
    emitted exactly once."""
    if not header_lines:
        df.to_csv(path, sep=sep, index=False, encoding="utf-8-sig")
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        for ln in header_lines:
            f.write(str(ln).rstrip("\n") + "\n")
        df.to_csv(f, sep=sep, index=False)


def _write_xlsx(df, path, header_lines):
    """Write an XLSX; when an audit block is supplied, add a second ``Provenance`` sheet
    holding the same record (the ``# `` markers stripped for readability)."""
    if not header_lines:
        df.to_excel(path, index=False)
        return
    import pandas as pd
    with pd.ExcelWriter(path) as xw:
        df.to_excel(xw, index=False, sheet_name="Data")
        rows = [str(ln).lstrip("#").strip() for ln in header_lines]
        pd.DataFrame({"Provenance & methods": rows}).to_excel(
            xw, index=False, sheet_name="Provenance")


def _to_markdown_table(df):
    """A GitHub-flavoured Markdown table without the optional ``tabulate`` dependency."""
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join("" if v is None else str(v) for v in r.tolist()) + " |")
    return "\n".join([head, sep, *rows]) + "\n"


def write_spectra(spectra, path, fmt=None):
    """Write one or more spectra ``[(name, axis, y), ...]`` to a table. A single spectrum
    becomes ``m/z,intensity``; several are aligned on a shared m/z axis when possible,
    else written long-form (``name,m/z,intensity``)."""
    import pandas as pd
    fmt = (fmt or os.path.splitext(str(path))[1].lstrip(".") or "csv").lower()
    spectra = [s for s in (spectra or []) if s is not None]
    if not spectra:
        raise ValueError("no spectra to write")
    if len(spectra) == 1:
        axis, y = np.asarray(spectra[0][1]), np.asarray(spectra[0][2])
        df = pd.DataFrame({"m/z": axis, "intensity": y})
        return write_table(df, path, fmt)
    axes = [np.asarray(s[1]) for s in spectra]
    same = all(a.shape == axes[0].shape and np.allclose(a, axes[0]) for a in axes)
    if same:
        df = pd.DataFrame({"m/z": axes[0]})
        for s in spectra:
            df[str(s[0])] = np.asarray(s[2])
    else:
        rows = []
        for s in spectra:
            for m, v in zip(np.asarray(s[1]), np.asarray(s[2])):
                rows.append({"spectrum": str(s[0]), "m/z": m, "intensity": v})
        df = pd.DataFrame(rows)
    return write_table(df, path, fmt)


# --------------------------------------------------------------------------- #
# Public: data bundle (the report's numbers as a tight folder of CSVs)
# --------------------------------------------------------------------------- #
# Each entry: option key → (filename, one-line description, mock columns, mock rows). The
# mock rows feed :func:`bundle_preview` so the GUI can show *exactly which data points* a
# bundle will contain before anything is written — no real data needed.
BUNDLE_SPEC = {
    "features": ("features.csv", "Annotated feature table (SMART-style: id, adduct, m/z bin)",
                 ["mz", "lipid", "adduct", "ppm_window", "intensity_lo", "intensity_hi"],
                 [[744.5548, "PC 34:1", "[M+H]+", "±5 ppm", "0%", "99%"],
                  [810.5291, "PE 40:6", "[M-H]-", "±5 ppm", "2%", "98%"]]),
    "region_stats": ("region_intensities.csv", "Per-region intensity of each feature",
                     ["region", "mz", "lipid", "n_px", "mean", "median", "std"],
                     [["Fascicle", 744.5548, "PC 34:1", 1820, 312.4, 287.0, 96.1],
                      ["Epineurium", 744.5548, "PC 34:1", 940, 88.2, 71.5, 40.3]]),
    "statistics": ("statistics.csv", "Discriminating features between two regions",
                   ["mz", "best_lipid", "AUC", "q_value", "log2_fc"],
                   [[463.2843, "LPA 18:1", 0.82, 1.3e-3, 1.07],
                    [888.6233, "ST 24:1", 0.18, 4.0e-4, -1.0]]),
    "spectra": ("spectra.csv", "Raw mean / skyline spectra — large (opt-in)",
                ["m/z", "mean spectrum", "skyline (max)"],
                [[400.0021, 12.3, 88.0], [400.0510, 9.1, 77.4]]),
    "methods": ("methods.md", "Methods, provenance, references + file manifest", None, None),
}

DEFAULT_BUNDLE_OPTIONS = {"features": True, "region_stats": True, "statistics": True,
                          "spectra": False, "methods": True}   # raw spectra are opt-in


def _bundle_options(document, options):
    """Resolve the effective bundle toggles. Explicit ``options`` win; otherwise fall back
    to the document's book ``sections`` with the new-key defaults (so older callers keep
    working, and the big raw-spectra dump stays off unless asked for)."""
    if options is not None:
        return {**DEFAULT_BUNDLE_OPTIONS, **options}
    sec = document.get("sections") or {}
    return {**DEFAULT_BUNDLE_OPTIONS,
            "region_stats": sec.get("stats", True), "statistics": sec.get("stats", True),
            "methods": sec.get("methods", True)}


def build_data_bundle(document, folder, options=None):
    """Write the report's underlying data as a tight folder of CSV/Markdown files — the
    machine-readable companion to :func:`build_book`. Reworked to ship only what is reusable
    as supplementary data (per the SMART / MIAMSIE reporting standards) instead of dumping
    everything: a SMART-style ``features.csv``, per-region ``region_intensities.csv``,
    discriminating ``statistics.csv``, an opt-in raw ``spectra.csv``, and one compact
    ``methods.md`` that folds in the dataset summary, provenance, references and a file
    manifest (the old verbose ``provenance.json`` + ``README.txt`` + ``dataset_summary.csv``
    are gone). ``options`` is the toggle dict (see :data:`DEFAULT_BUNDLE_OPTIONS`)."""
    import pandas as pd

    os.makedirs(folder, exist_ok=True)
    opt = _bundle_options(document, options)
    written = []

    fdf = document.get("feature_table")
    if opt.get("features") and fdf is not None and len(fdf):
        write_table(fdf, os.path.join(folder, "features.csv"), fmt="csv")
        written.append(("features.csv", "annotated feature table"))

    rdf = document.get("region_stats")
    if opt.get("region_stats") and rdf is not None and len(rdf):
        write_table(rdf, os.path.join(folder, "region_intensities.csv"), fmt="csv")
        written.append(("region_intensities.csv", "per-region feature intensities"))

    stats = document.get("stats")
    if opt.get("statistics") and stats is not None and stats.get("df") is not None and len(stats["df"]):
        write_table(stats["df"], os.path.join(folder, "statistics.csv"), fmt="csv")
        written.append(("statistics.csv", "discriminating features"))

    if opt.get("spectra") and document.get("spectra"):
        try:
            write_spectra(document["spectra"], os.path.join(folder, "spectra.csv"), fmt="csv")
            written.append(("spectra.csv", "raw mean / skyline spectra"))
        except ValueError:
            pass

    if opt.get("methods"):
        try:
            _write_bundle_methods(document, os.path.join(folder, "methods.md"), written)
            written.append(("methods.md", "methods, provenance & references"))
        except Exception:  # noqa: BLE001 — methods export is best-effort
            pass
    return folder


def _provenance_sections(prov, *, include_method=False):
    """Ordered ``(heading, [lines])`` provenance sections shared by the bundle
    methods file and the PDF data-book. Returns line *content* only — each caller
    applies its own heading style / bullets / indentation / pagination. The four
    sections, in order, are Input files, Processing steps, Software, References
    (each present only when the corresponding ``Provenance`` field is non-empty).
    ``include_method`` appends the input fingerprint method to each input line (the
    book includes it; the bundle does not)."""
    out = []
    if getattr(prov, "inputs", None):
        lines = []
        for i in prov.inputs:
            base = (f"{i.get('name', '?')} ({i.get('bytes', 0):,} bytes, "
                    f"sha256 {str(i.get('sha256', ''))[:16]}…")
            lines.append(base + (f" {i.get('method', '')})" if include_method else ")"))
        out.append(("Input files", lines))
    if getattr(prov, "steps", None):
        lines = []
        for k, s in enumerate(prov.steps, 1):
            p = ", ".join(f"{a}={b}" for a, b in (s.get("params") or {}).items())
            lines.append(f"{k}. {s.get('step', '')}" + (f" — {p}" if p else ""))
        out.append(("Processing steps", lines))
    if getattr(prov, "environment", None):
        out.append(("Software", [f"{k}: {v}" for k, v in prov.environment.items()]))
    refs = prov.references_used() if hasattr(prov, "references_used") else []
    if refs:
        out.append(("References", [f"{i}. {r}" for i, r in enumerate(refs, 1)]))
    return out


def _write_bundle_methods(document, path, written):
    """One readable methods file: a Methods paragraph, the dataset summary, provenance
    (inputs/steps/software), references, and a manifest of the other files in the bundle —
    everything the old provenance.json + README + dataset_summary.csv carried, in one place."""
    prov = document.get("provenance")
    lines = [f"# {document.get('title', 'MSI analysis')} — supplementary data", "", BRAND, ""]

    mp = document.get("methods_paragraph")
    if mp is None and prov is not None:
        try:
            mp = prov.methods_paragraph()
        except Exception:  # noqa: BLE001
            mp = None
    if mp:
        lines += ["## Methods", "", str(mp), ""]

    meta = document.get("meta") or {}
    if meta:
        lines += ["## Dataset", ""] + [f"- **{k}:** {v}" for k, v in meta.items()] + [""]

    if prov is not None:
        try:
            for head, body in _provenance_sections(prov, include_method=False):
                # Input files / Software render as Markdown bullets; numbered
                # Processing steps and References carry their own "N." prefix.
                if head in ("Input files", "Software"):
                    body = [f"- {ln}" for ln in body]
                lines += [f"## {head}", ""] + body + [""]
        except Exception:  # noqa: BLE001
            pass

    if written:
        lines += ["## Files in this bundle", ""]
        lines += [f"- `{fn}` — {desc}" for fn, desc in written]
        lines.append("- `methods.md` — this file")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def bundle_preview(options=None, *, max_rows=2):
    """A short, human-readable preview of what a data bundle will contain for the given
    ``options`` — the file list, each file's purpose, and a *mock* column/row sample (not
    real data) so the user sees the exact data points they'll get before exporting."""
    opt = {**DEFAULT_BUNDLE_OPTIONS, **(options or {})}
    out = []
    for key, (fname, desc, cols, rows) in BUNDLE_SPEC.items():
        if not opt.get(key):
            continue
        out.append(f"📄 {fname} — {desc}")
        if cols:
            out.append("    " + " · ".join(str(c) for c in cols))
            for r in (rows or [])[:max_rows]:
                out.append("    " + " · ".join(str(v) for v in r))
        else:
            out.append("    (Methods paragraph · dataset summary · provenance · references · manifest)")
        out.append("")
    if not out:
        return "Nothing selected — the bundle would be empty."
    return "\n".join(out).rstrip()


# --------------------------------------------------------------------------- #
# Public: the full PDF book
# --------------------------------------------------------------------------- #
BRAND = "SMILE MSI · Spatial Mass Imaging of Lipid Environments"


def build_book(document, path, *, theme="dark", dpi=200, page=(8.27, 11.69)):
    """Assemble the multi-page PDF report. ``document`` is a plain dict:

    ``title``/``subtitle``/``meta`` (cover), ``dataset_lines`` (summary bullets),
    ``methods_paragraph``/``provenance`` (methods + traceability),
    ``panels`` (list of ``render_ion_panel`` kwargs), ``spectra`` (``[(name,axis,y)]``),
    ``segmentation`` (``{image, colors, legend, title}``), ``stats``
    (``{df, a_label, b_label, title}``), ``sections`` (which to include).

    When ``items`` is present (the curated Report-tab list) the body is rendered from it
    **in order** instead of the fixed sections above — cover + summary + methods still come
    from the document. Each item is ``{type, title?, caption?, ...payload}``; see
    :func:`_book_items`.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    style = make_style(theme)
    pw, ph = page
    sections = document.get("sections") or {}

    def want(key, default=True):
        return sections.get(key, default)

    with PdfPages(path) as pdf:
        # 1) cover
        _book_cover(pdf, document, style, pw, ph, dpi)
        # 2) dataset summary
        if want("summary") and document.get("dataset_lines"):
            _book_text_page(pdf, "Dataset summary", document["dataset_lines"], style, pw, ph,
                            dpi, page_no=2, bullets=True)
        # 3) methods & provenance
        if want("methods"):
            _book_methods_pages(pdf, document, style, pw, ph, dpi)
        if document.get("items"):
            # curated, ordered report-item list (Report tab)
            _book_items(pdf, document, style, pw, ph, dpi)
        else:
            # fixed-section book assembled from live state (no curated list)
            if want("gallery") and document.get("panels"):
                _book_gallery(pdf, document["panels"], style, pw, ph, dpi)
            if want("spectra") and document.get("spectra"):
                _book_spectra_page(pdf, document["spectra"], style, pw, ph, dpi)
            if want("segmentation") and document.get("segmentation"):
                _book_segmentation_page(pdf, document["segmentation"], style, pw, ph, dpi)
            if want("stats") and document.get("stats"):
                _book_stats_page(pdf, document["stats"], style, pw, ph, dpi)
        _set_pdf_metadata(pdf, document)
    return path


def roc_report_pdf(path, stats_df, curves, *, a_label="Region A", b_label="Region B",
                   title="ROI ROC report", note="", theme="dark", page=(11.0, 8.5), dpi=200):
    """A focused PDF of ROC curves for an A-vs-B ROI comparison.

    Page 1 is a volcano overview of *every* tested ion (effect size vs −log10 q), with a
    summary table of the chosen ions. The following pages lay out one ROC curve per
    chosen ion in a grid — each reproduces the on-screen view: the chance diagonal plus
    the ion's curve, shaded to its area (the AUC), bowing up (blue, enriched in B) or
    down (orange, enriched in A).

    ``curves`` is a list of dicts ``{mz, lipid, fpr, tpr, auc, q, log2_fc, higher}`` where
    ``higher`` is the enriched region's label (the GUI builds these via
    :func:`smile_msi.spatial.roc_curve`). Landscape page so the grid breathes.
    """
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    style = make_style(theme)
    pw, ph = page
    A_COLOR, B_COLOR = AB_A_COLOR, AB_B_COLOR        # orange = A, blue = B (matches the GUI)
    curves = list(curves or [])

    with PdfPages(path) as pdf:
        # --- page 1: volcano overview + summary table of the chosen ions ---
        fig = _new_figure(pw, ph, dpi, facecolor=style.page_bg)
        fig.text(0.06, 0.95, title, color=style.page_fg, fontsize=16, fontweight="bold",
                 ha="left", va="center")
        fig.text(0.06, 0.915, f"{a_label}  vs  {b_label}", color=style.fg_dim, fontsize=10,
                 ha="left", va="center")
        if note:
            fig.text(0.94, 0.93, note, color=style.fg_dim, fontsize=8, ha="right", va="center")
        ax = fig.add_axes([0.40, 0.10, 0.55, 0.74])
        ax.set_facecolor(style.page_bg)
        if stats_df is not None and len(stats_df):
            x = (stats_df["AUC"].to_numpy() - 0.5) * 2
            q = np.clip(stats_df["q_value"].to_numpy(), 1e-12, 1)
            y = -np.log10(q)
            sig = q <= 0.05
            colors = [(B_COLOR if xi > 0 else A_COLOR) if s else "#9aa3af"
                      for xi, s in zip(x, sig)]
            ax.scatter(x, y, c=colors, s=16, edgecolor=style.page_bg, linewidth=0.3)
            ax.axhline(-np.log10(0.05), color=style.grid, ls="--", lw=0.8)
            ax.axvline(0, color=style.grid, lw=0.8)
            ax.set_xlabel(f"← more in {a_label}      effect size 2·(AUC−0.5)      more in {b_label} →",
                          color=style.page_fg, fontsize=8.5)
            ax.set_ylabel("−log10 q (FDR)", color=style.page_fg, fontsize=9)
        ax.set_title("Discrimination overview", color=style.page_fg, fontsize=11, loc="left")
        ax.tick_params(colors=style.page_fg, labelsize=8, length=3)
        for sp in ax.spines.values():
            sp.set_color(style.grid)
        # summary table of the ions whose ROC curves follow
        tax = fig.add_axes([0.04, 0.10, 0.32, 0.74]); tax.set_axis_off()
        if curves:
            cell_rows = [[f"{c['mz']:.4f}", (c.get("lipid") or "—")[:22], f"{c['auc']:.3f}",
                          f"↑ {c.get('higher', '')}"] for c in curves]
            tbl = tax.table(cellText=cell_rows, colLabels=["m/z", "lipid", "AUC", "enriched"],
                            loc="upper left", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(7.0)
            for (ri, _ci), cell in tbl.get_celld().items():
                cell.set_edgecolor(style.grid)
                cell.set_facecolor(style.page_bg)
                cell.get_text().set_color(style.page_fg)
                if ri == 0:
                    cell.get_text().set_fontweight("bold")
        _footer(fig, style, page_no=1)
        pdf.savefig(fig, facecolor=style.page_bg); _close(fig)

        # --- ROC grid pages: 3×2 curves per landscape page ---
        ncol, nrow = 3, 2
        per_page = ncol * nrow
        n_pages = max(1, (len(curves) + per_page - 1) // per_page)
        for pi in range(n_pages):
            chunk = curves[pi * per_page:(pi + 1) * per_page]
            fig = _new_figure(pw, ph, dpi, facecolor=style.page_bg)
            fig.text(0.06, 0.955, f"ROC curves — {a_label} vs {b_label}", color=style.page_fg,
                     fontsize=13, fontweight="bold", ha="left", va="center")
            for i, c in enumerate(chunk):
                r, cc = divmod(i, ncol)
                w, h = 0.27, 0.36
                x0 = 0.06 + cc * 0.315
                y0 = 0.50 - r * 0.42
                ax = fig.add_axes([x0, y0, w, h])
                ax.set_facecolor(style.page_bg)
                higher_b = c["auc"] >= 0.5
                color = B_COLOR if higher_b else A_COLOR
                ax.plot([0, 1], [0, 1], color=style.grid, ls="--", lw=0.8)
                ax.fill_between(c["fpr"], c["tpr"], color=color, alpha=0.20)
                ax.plot(c["fpr"], c["tpr"], color=color, lw=1.8)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                lab = (c.get("lipid") or f"m/z {c['mz']:.4f}")[:26]
                ax.set_title(f"{lab}\nAUC {c['auc']:.3f}  ·  q {c.get('q', float('nan')):.1e}",
                             color=style.page_fg, fontsize=7.5)
                ax.tick_params(colors=style.page_fg, labelsize=6, length=2)
                for sp in ax.spines.values():
                    sp.set_color(style.grid)
            fig.text(0.5, 0.045, f"false-positive rate ({a_label})", color=style.page_fg,
                     fontsize=9, ha="center")
            fig.text(0.012, 0.5, f"true-positive rate ({b_label})", color=style.page_fg,
                     fontsize=9, va="center", rotation="vertical")
            _footer(fig, style, page_no=pi + 2)
            pdf.savefig(fig, facecolor=style.page_bg); _close(fig)

        try:
            d = pdf.infodict()
            d["Title"], d["Author"], d["Creator"] = title, "SMILE MSI", BRAND
        except Exception:  # noqa: BLE001
            pass
    return path


def _set_pdf_metadata(pdf, document):
    try:
        d = pdf.infodict()
        d["Title"] = document.get("title", "MSI analysis report")
        d["Author"] = "SMILE MSI"
        d["Subject"] = "MALDI mass spectrometry imaging analysis"
        d["Creator"] = BRAND
    except Exception:  # noqa: BLE001
        pass


def _page_fig(style, pw, ph, dpi):
    fig = _new_figure(pw, ph, dpi, facecolor=style.page_bg)
    return fig


def _footer(fig, style, page_no=None):
    fig.text(0.06, 0.035, BRAND, color=style.fg_dim, fontsize=7.0, ha="left", va="center")
    if page_no is not None:
        fig.text(0.94, 0.035, str(page_no), color=style.fg_dim, fontsize=8.0,
                 ha="right", va="center")
    fig.add_artist(_hline(fig, 0.06, 0.94, 0.058, style.grid))


def _hline(fig, x0, x1, y, color, lw=0.8):
    from matplotlib.lines import Line2D
    return Line2D([x0, x1], [y, y], color=color, lw=lw, transform=fig.transFigure)


def _book_cover(pdf, document, style, pw, ph, dpi):
    fig = _page_fig(style, pw, ph, dpi)
    acc = accent_for_cmap(document.get("cmap", "viridis"))
    fig.add_artist(_hline(fig, 0.12, 0.88, 0.74, acc, lw=2.5))
    fig.text(0.12, 0.80, document.get("title", "MSI analysis report"), color=style.page_fg,
             fontsize=27, fontweight="bold", ha="left", va="bottom")
    sub = document.get("subtitle", "")
    if sub:
        fig.text(0.12, 0.685, sub, color=style.fg_dim, fontsize=13, ha="left", va="top")
    # key/value metadata block
    meta = document.get("meta") or {}
    y = 0.55
    for k, v in meta.items():
        fig.text(0.12, y, f"{k}", color=style.fg_dim, fontsize=10.5, ha="left", va="top")
        fig.text(0.34, y, f"{v}", color=style.page_fg, fontsize=10.5, ha="left", va="top")
        y -= 0.038
    fig.text(0.12, 0.10, BRAND, color=acc, fontsize=10, ha="left", va="center", fontweight="bold")
    fig.text(0.12, 0.072, "Generated locally · MIT-licensed · reproducible analysis record",
             color=style.fg_dim, fontsize=8, ha="left", va="center")
    pdf.savefig(fig, facecolor=style.page_bg)
    _close(fig)


def _wrap(text, width):
    import textwrap
    out = []
    for para in str(text).split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return out


def _book_text_page(pdf, heading, lines, style, pw, ph, dpi, page_no=None, bullets=False,
                    wrap_width=92):
    """Paginate ``lines`` (a list of strings, or a single string) under ``heading``."""
    if isinstance(lines, str):
        lines = _wrap(lines, wrap_width)
    else:
        flat = []
        for ln in lines:
            flat.extend(_wrap(ln, wrap_width) if len(str(ln)) > wrap_width else [str(ln)])
        lines = flat
    top, bottom, line_h = 0.88, 0.09, 0.026
    per_page = int((top - bottom) / line_h)
    chunks = [lines[i:i + per_page] for i in range(0, max(len(lines), 1), per_page)] or [[]]
    for ci, chunk in enumerate(chunks):
        fig = _page_fig(style, pw, ph, dpi)
        head = heading if ci == 0 else f"{heading} (cont.)"
        fig.text(0.06, 0.93, head, color=style.page_fg, fontsize=16, fontweight="bold",
                 ha="left", va="center")
        y = top
        for ln in chunk:
            prefix = "•  " if (bullets and ln.strip()) else ""
            fig.text(0.07 if not bullets else 0.065, y, prefix + ln, color=style.page_fg,
                     fontsize=9.6, ha="left", va="top", family="DejaVu Sans")
            y -= line_h
        _footer(fig, style, page_no if page_no and ci == 0 else None)
        pdf.savefig(fig, facecolor=style.page_bg)
        _close(fig)


def _book_methods_pages(pdf, document, style, pw, ph, dpi):
    blocks = []
    prov = document.get("provenance")
    mp = document.get("methods_paragraph")
    if mp is None and prov is not None:
        try:
            mp = prov.methods_paragraph()
        except Exception:  # noqa: BLE001
            mp = None
    if mp:
        blocks.append(("Methods", mp))
    if prov is not None:
        try:
            prov_lines, ref_block = [], None
            for head, body in _provenance_sections(prov, include_method=True):
                if head == "References":
                    ref_block = body
                elif head == "Input files":
                    prov_lines += ["Input files:"] + [f"  - {ln}" for ln in body] + [""]
                elif head == "Processing steps":
                    prov_lines += ["Processing steps:"] + [f"  {ln}" for ln in body] + [""]
                elif head == "Software":
                    prov_lines += ["Software:"] + [f"  - {ln}" for ln in body]
            if prov_lines:
                blocks.append(("Provenance & traceability", prov_lines))
            if ref_block:
                blocks.append(("References", ref_block))
        except Exception:  # noqa: BLE001
            pass
    for i, (head, body) in enumerate(blocks):
        _book_text_page(pdf, head, body, style, pw, ph, dpi, page_no=(3 if i == 0 else None))


def _book_gallery(pdf, panels, style, pw, ph, dpi, per_page=2, heading="Ion images"):
    """Image panels two-up; each rendered through the shared panel renderer so the gallery
    carries the same side-margin spectrum + bottom-corner intensity scale as the
    standalone image export (the image data itself stays uncovered)."""
    chunks = [panels[i:i + per_page] for i in range(0, len(panels), per_page)]
    for ci, chunk in enumerate(chunks):
        fig = _page_fig(style, pw, ph, dpi)
        head = heading if ci == 0 else f"{heading} (cont.)"
        fig.text(0.06, 0.955, head, color=style.page_fg, fontsize=16,
                 fontweight="bold", ha="left", va="center")
        # full-width vertical slots; each holds one image + its side panel composite
        slots = [(0.07, 0.52, 0.86, 0.40), (0.07, 0.085, 0.86, 0.40)][:len(chunk)]
        for slot, pk in zip(slots, chunk):
            _embed_panel(fig, slot, pk, style, dpi=dpi)
        _footer(fig, style)
        pdf.savefig(fig, facecolor=style.page_bg)
        _close(fig)


def _embed_panel(parent_fig, rect, panel_kwargs, style, dpi=200):
    """A gallery entry inside ``rect``. For an ion panel: the image (rendered through the
    shared renderer, so it keeps the clean look + bottom-corner intensity scale) with the
    mean **spectrum directly below it** on a labelled m/z axis. For a colour-overlay panel
    (``kind == "overlay"``): the additive composite rendered full-frame, no spectrum. Both
    carry an optional user title + caption above the image; the image data stays uncovered."""
    x0, y0, w, h = rect
    pk = dict(panel_kwargs)
    pk.setdefault("theme", style.name)       # match the book's theme/resolution when the
    pk.setdefault("dpi", dpi)                 # caller didn't pin them onto the panel kwargs
    pk.setdefault("width_in", 7.0)
    acc = style.accent or accent_for_cmap(pk.get("cmap", "viridis"))

    if pk.get("kind") == "overlay":
        # additive multi-ion composite: render full-frame (its own corner legend), no spectrum
        ofig = render_overlay_panel(
            pk["rgb"], pk.get("channels"), crop=pk.get("crop"), outlines=pk.get("outlines"),
            show_legend=pk.get("show_legend", True), **(pk.get("design") or {}))
        arr = _fig_to_raster(ofig)
        has_spec = False
        default_title = "Colour overlay"
    else:
        has_spec = (pk.get("overlay", True) and pk.get("show_spectrum", True)
                    and pk.get("mean_spectrum") is not None)
        # image: render with the side spectrum OFF (it goes below) but keep the corner
        # intensity scale; the title above carries the lipid name, so the chip shows m/z only.
        img_pk = dict(pk)
        img_pk["show_spectrum"] = False
        img_pk["mean_spectrum"] = None
        img_pk["label"] = ""
        for k in ("legend", "kind", "caption"):
            img_pk.pop(k, None)
        arr = _render_panel_raster(img_pk)
        mz, label = pk.get("mz"), pk.get("label", "")
        default_title = (f"m/z {float(mz):.4f}   {label}".strip()
                         if mz is not None else (label or "Ion image"))

    # optional caption (≤2 wrapped lines) sits between the title row and the image
    cap_lines = _wrap(pk.get("caption"), 64)[:2] if pk.get("caption") else []
    spec_h = 0.11 if has_spec else 0.0       # spectrum panel height (figure fraction)
    title_h = 0.022                          # title row above the image
    cap_h = 0.016 * len(cap_lines)
    gap = 0.014 if has_spec else 0.0
    img_area_h = max(h - spec_h - title_h - cap_h - gap, 0.05)

    ih, iw = arr.shape[:2]
    fig_w, fig_h = parent_fig.get_figwidth(), parent_fig.get_figheight()
    img_top = y0 + spec_h + gap
    slot_aspect = (w * fig_w) / (img_area_h * fig_h)
    img_aspect = iw / max(ih, 1)
    if img_aspect > slot_aspect:             # image wider → limit by width
        dw = w
        dh = w * fig_w / fig_h / img_aspect
    else:
        dh = img_area_h
        dw = img_area_h * fig_h / fig_w * img_aspect
    ax = parent_fig.add_axes([x0 + (w - dw) / 2, img_top + (img_area_h - dh), dw, dh])
    ax.set_facecolor(style.page_bg)
    ax.imshow(arr, aspect="auto")
    ax.set_axis_off()

    # title (+ caption) above the image
    title = pk.get("title") or default_title
    parent_fig.text(x0, img_top + img_area_h + cap_h + 0.004, title, color=style.page_fg,
                    fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    cy = img_top + img_area_h + cap_h - 0.002
    for ln in cap_lines:
        parent_fig.text(x0, cy, ln, color=style.fg_dim, fontsize=7.6, ha="left", va="top")
        cy -= 0.015

    if has_spec:
        _embed_spectrum_strip(parent_fig, (x0, y0, w, spec_h), pk, style, acc)


def _render_panel_raster(pk):
    """Render one ion panel to an RGBA array via the shared renderer, so an embedded
    gallery image is pixel-identical to the standalone image export."""
    return _fig_to_raster(render_ion_panel(**pk))


def _fig_to_raster(fig):
    """Rasterize a finished figure to an RGBA array and close it (shared by the gallery
    embedders so an embedded panel matches its standalone export pixel-for-pixel)."""
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba()).copy()
    _close(fig)
    return arr


def _embed_spectrum_strip(fig, rect, pk, style, acc):
    """The mean spectrum below a gallery image, on a normal labelled m/z axis with the
    active m/z marked. Reserves the rect's bottom for the x-axis ticks + label."""
    x0, y0, w, h = rect
    axis, spec = _clip_xy(*pk["mean_spectrum"])
    sax = fig.add_axes([x0, y0 + h * 0.36, w, h * 0.64])   # bottom 36% holds the x labels
    sax.set_facecolor(style.page_bg)
    if axis.size and spec.size:
        sax.fill_between(axis, spec, color=acc, alpha=0.18, linewidth=0)
        sax.plot(axis, spec, color=acc, linewidth=0.9)
        if pk.get("mz") is not None:
            sax.axvline(float(pk["mz"]), color=style.page_fg, linewidth=0.9, alpha=0.7)
        sax.set_ylim(0, float(np.nanmax(spec)) * 1.15 + 1e-9)
        sax.set_xlim(float(axis.min()), float(axis.max()))
    sax.set_yticks([])
    sax.tick_params(axis="x", colors=style.page_fg, labelsize=7.0, length=2)
    sax.set_xlabel("m/z", color=style.page_fg, fontsize=7.5, labelpad=1)
    for side, sp in sax.spines.items():
        sp.set_visible(side == "bottom")
        sp.set_color(style.grid)
    sax.text(0.0, 1.03, "mean spectrum", transform=sax.transAxes, color=style.fg_dim,
             fontsize=6.8, ha="left", va="bottom")
    if pk.get("mz") is not None:
        sax.text(1.0, 1.03, f"m/z {float(pk['mz']):.4f}", transform=sax.transAxes,
                 color=style.page_fg, fontsize=7.2, fontweight="bold", ha="right", va="bottom")


def _draw_caption(fig, caption, style, y=0.93):
    """A single dimmed caption line under a full-page heading (report-item captions)."""
    if not caption:
        return
    lines = _wrap(caption, 110)
    if not lines:
        return
    txt = lines[0] + ("…" if len(lines) > 1 else "")
    fig.text(0.06, y, txt, color=style.fg_dim, fontsize=9.0, ha="left", va="center")


def _book_spectra_page(pdf, spectra, style, pw, ph, dpi, per_page=5, heading="Spectra",
                       caption=None):
    """Each spectrum is its own short panel, stacked up to ``per_page`` per page (so a
    column of traces reads cleanly instead of one tall overlaid plot). Paginates when
    there are more, and keeps the panel height fixed so a partial last page matches."""
    palette = ["#5aa9e6", "#f4a259", "#8ac926", "#e15759", "#b07aa1", "#76b7b2"]
    spectra = [s for s in (spectra or []) if s is not None]
    if not spectra:
        return
    chunks = [spectra[i:i + per_page] for i in range(0, len(spectra), per_page)]
    top, bottom = 0.90, 0.085
    row_h = (top - bottom) / per_page                 # fixed row height regardless of count
    for ci, chunk in enumerate(chunks):
        fig = _page_fig(style, pw, ph, dpi)
        head = heading if ci == 0 else f"{heading} (cont.)"
        fig.text(0.06, 0.955, head, color=style.page_fg, fontsize=16, fontweight="bold",
                 ha="left", va="center")
        if ci == 0:
            _draw_caption(fig, caption, style)
        for ri, s in enumerate(chunk):
            gi = ci * per_page + ri
            axis, y = _clip_xy(s[1], s[2])
            color = s[3] if len(s) > 3 else palette[gi % len(palette)]
            ry0 = top - (ri + 1) * row_h
            ax = fig.add_axes([0.10, ry0 + row_h * 0.22, 0.84, row_h * 0.62])
            ax.set_facecolor(style.page_bg)
            ymax = float(np.nanmax(y)) if y.size else 0.0
            ax.fill_between(axis, y, color=color, alpha=0.16, linewidth=0)
            ax.plot(axis, y, color=color, linewidth=1.0)
            ax.set_ylim(0, ymax * 1.28 + 1e-9)        # headroom matches the figure export
            xlim = _spectrum_xlim([(s[0], axis, y)])  # crop to signal, ~50 m/z past the last peak
            if xlim:
                ax.set_xlim(*xlim)
            elif axis.size:
                ax.set_xlim(float(axis.min()), float(axis.max()))
            is_last = (ri == len(chunk) - 1)          # only the bottom panel labels the m/z axis
            _style_axes(ax, style, "", "m/z" if is_last else "", "intensity")
            ax.text(0.006, 0.94, str(s[0]), transform=ax.transAxes, color=style.page_fg,
                    fontsize=9.0, fontweight="bold", ha="left", va="top")
        _footer(fig, style)
        pdf.savefig(fig, facecolor=style.page_bg)
        _close(fig)


def _book_segmentation_page(pdf, seg, style, pw, ph, dpi, caption=None):
    fig = _page_fig(style, pw, ph, dpi)
    fig.text(0.06, 0.955, seg.get("title", "Segmentation"), color=style.page_fg, fontsize=16,
             fontweight="bold", ha="left", va="center")
    _draw_caption(fig, caption, style)
    colors = seg.get("colors") or {}
    # match the live (pyqtgraph) viewer orientation; reuse the canonical label→RGBA helper
    rgba = _view_orient(segmentation_rgba(seg["image"], colors))
    ax = fig.add_axes([0.07, 0.30, 0.86, 0.60])
    ax.imshow(rgba, interpolation="nearest", aspect="equal", origin="upper")
    ax.set_axis_off()
    # legend
    legend = seg.get("legend") or [(c, f"cluster {cl}") for cl, c in colors.items()]
    y = 0.24
    fig.text(0.07, y + 0.02, "Regions", color=style.page_fg, fontsize=11, fontweight="bold",
             ha="left", va="top")
    from matplotlib.patches import Rectangle
    x = 0.07
    for color, name in legend[:18]:
        fig.add_artist(Rectangle((x, y - 0.02), 0.018, 0.018, transform=fig.transFigure,
                                 facecolor=color, edgecolor="none"))
        fig.text(x + 0.026, y - 0.011, str(name), color=style.page_fg, fontsize=8.5,
                 ha="left", va="center")
        x += 0.205
        if x > 0.8:
            x = 0.07; y -= 0.035
    _footer(fig, style)
    pdf.savefig(fig, facecolor=style.page_bg)
    _close(fig)


def _book_stats_page(pdf, stats, style, pw, ph, dpi, caption=None):
    df = stats.get("df")
    a_label, b_label = stats.get("a_label", "Group A"), stats.get("b_label", "Group B")
    fig = _page_fig(style, pw, ph, dpi)
    fig.text(0.06, 0.955, stats.get("title", "Discriminating features"), color=style.page_fg,
             fontsize=16, fontweight="bold", ha="left", va="center")
    _draw_caption(fig, caption, style)
    if df is None or len(df) == 0:
        _footer(fig, style)
        pdf.savefig(fig, facecolor=style.page_bg); _close(fig)
        return
    d = df.copy()
    if "AUC" in d.columns:
        d["discrim"] = (d["AUC"] - 0.5).abs()
        d = d.sort_values("discrim", ascending=False)
    top = d.head(18)
    # diverging bar chart of AUC-0.5 (orange = A, blue = B)
    ax = fig.add_axes([0.30, 0.40, 0.63, 0.50])
    ax.set_facecolor(style.page_bg)
    if "AUC" in top.columns:
        vals = (top["AUC"] - 0.5).values[::-1]
        labels = [_short_lipid(r) for _, r in list(top.iterrows())[::-1]]
        colors = [AB_B_COLOR if v >= 0 else AB_A_COLOR for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=7.0, color=style.page_fg)
        ax.axvline(0, color=style.grid, linewidth=0.8)
        ax.set_xlabel(f"← more in {a_label}      AUC − 0.5      more in {b_label} →",
                      color=style.page_fg, fontsize=8.5)
        ax.tick_params(colors=style.page_fg, length=0)
        for side, sp in ax.spines.items():
            sp.set_visible(side == "bottom"); sp.set_color(style.grid)
    # compact table beneath
    cols = [c for c in ("mz", "best_lipid", "AUC", "q_value", "log2_fc") if c in top.columns]
    if cols:
        cell_rows = []
        for _, r in top.head(12).iterrows():
            cell_rows.append([_fmt_cell(r[c]) for c in cols])
        tax = fig.add_axes([0.06, 0.08, 0.88, 0.26]); tax.set_axis_off()
        tbl = tax.table(cellText=cell_rows, colLabels=cols, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.0)
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_edgecolor(style.grid)
            cell.set_facecolor(style.page_bg)
            cell.get_text().set_color(style.page_fg)
            if ri == 0:
                cell.get_text().set_fontweight("bold")
    _footer(fig, style)
    pdf.savefig(fig, facecolor=style.page_bg)
    _close(fig)


def _book_feature_table_page(pdf, item, style, pw, ph, dpi, per_page=26):
    """A paginated feature table — the curated 'feature list' report item. ``item`` carries
    ``records`` (list of row dicts), an optional ``title`` heading and ``caption``."""
    records = list(item.get("records") or [])
    heading = item.get("title") or "Feature table"
    prefer = ["mz", "lipid", "best_lipid", "name", "class", "adduct", "ppm", "confidence",
              "rel_intensity", "snr"]
    cols = []
    if records:
        present = list(records[0].keys())
        cols = [c for c in prefer if c in present] or present[:6]
    chunks = [records[i:i + per_page] for i in range(0, max(len(records), 1), per_page)] or [[]]
    for ci, chunk in enumerate(chunks):
        fig = _page_fig(style, pw, ph, dpi)
        head = heading if ci == 0 else f"{heading} (cont.)"
        fig.text(0.06, 0.955, head, color=style.page_fg, fontsize=16, fontweight="bold",
                 ha="left", va="center")
        if ci == 0:
            _draw_caption(fig, item.get("caption"), style)
        if chunk and cols:
            cell_rows = [[_fmt_cell(r.get(c)) for c in cols] for r in chunk]
            tax = fig.add_axes([0.06, 0.07, 0.88, 0.84]); tax.set_axis_off()
            tbl = tax.table(cellText=cell_rows, colLabels=cols, loc="upper center", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(7.2); tbl.scale(1, 1.25)
            for (ri, _ci), cell in tbl.get_celld().items():
                cell.set_edgecolor(style.grid)
                cell.set_facecolor(style.page_bg)
                cell.get_text().set_color(style.page_fg)
                if ri == 0:
                    cell.get_text().set_fontweight("bold")
        elif not records:
            fig.text(0.06, 0.88, "(no features)", color=style.fg_dim, fontsize=10,
                     ha="left", va="top")
        _footer(fig, style)
        pdf.savefig(fig, facecolor=style.page_bg)
        _close(fig)


def _book_note_page(pdf, item, style, pw, ph, dpi):
    """A free-text note / section heading the user added between figures."""
    heading = item.get("heading") or item.get("title") or "Note"
    body = item.get("text") or item.get("caption") or ""
    _book_text_page(pdf, heading, body, style, pw, ph, dpi)


def _book_items(pdf, document, style, pw, ph, dpi):
    """Render the curated report-item list (the Report tab) **in order**. Consecutive image
    panels (ion / colour-overlay) pack two-up on gallery pages; every other item type gets
    its own page via the matching ``_book_*`` renderer."""
    items = document.get("items") or []
    i, n = 0, len(items)
    while i < n:
        t = (items[i].get("type") or "").lower()
        if t in ("ion", "overlay"):
            run = []
            while i < n and (items[i].get("type") or "").lower() in ("ion", "overlay"):
                it = items[i]; i += 1
                pk = dict(it.get("panel") or {})
                if it.get("title"):
                    pk["title"] = it["title"]
                if it.get("caption"):
                    pk["caption"] = it["caption"]
                run.append(pk)
            heading = "Images" if any(p.get("kind") == "overlay" for p in run) else "Ion images"
            _book_gallery(pdf, run, style, pw, ph, dpi, heading=heading)
            continue
        it = items[i]; i += 1
        if t == "spectrum":
            _book_spectra_page(pdf, it.get("spectra") or [], style, pw, ph, dpi,
                               heading=it.get("title") or "Spectrum", caption=it.get("caption"))
        elif t == "segmentation":
            seg = dict(it.get("segmentation") or {})
            if it.get("title"):
                seg["title"] = it["title"]
            _book_segmentation_page(pdf, seg, style, pw, ph, dpi, caption=it.get("caption"))
        elif t == "stats":
            st = dict(it.get("stats") or {})
            if it.get("title"):
                st["title"] = it["title"]
            _book_stats_page(pdf, st, style, pw, ph, dpi, caption=it.get("caption"))
        elif t == "features":
            _book_feature_table_page(pdf, it, style, pw, ph, dpi)
        elif t == "note":
            _book_note_page(pdf, it, style, pw, ph, dpi)


def _short_lipid(row):
    for k in ("best_lipid", "lipid", "mz"):
        if k in row and str(row[k]) not in ("", "nan", "None"):
            v = row[k]
            return (f"{v:.4f}" if k == "mz" else str(v))[:22]
    return "?"


def _fmt_cell(v):
    try:
        f = float(v)
        return f"{f:.4f}" if abs(f) < 1000 else f"{f:.1f}"
    except (ValueError, TypeError):
        return "" if v is None else str(v)
