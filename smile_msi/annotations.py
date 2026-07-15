"""Shared layout + label model for ion-image annotations (intensity legend + scale bar).

This module is the single source of truth so the **export** renderer (matplotlib, in
:mod:`smile_msi.export`) and the **live** renderer (pyqtgraph, in
:mod:`smile_msi.gui.ionannotations`) lay out and label annotations identically — what you
see on screen is what you export (WYSIWYG).

It is intentionally **pure**: numpy only, no Qt and no matplotlib, so it imports cheaply and
runs headless. It computes *geometry in image-frame fractions* and *resolved label text*;
each renderer maps the fractions into its own coordinate system and draws the primitives.

Coordinate convention for layout fractions: origin **bottom-left**, x increases right,
y increases up, both in [0, 1] over the frame's ``x_extent`` span (the image region). This
matches :func:`smile_msi.export._overlay_axes` (a 0..1 figure overlay).

Occupancy / corner helpers work in **array space** (imshow convention): row 0 is the top,
col 0 is the left; "lower" means a high row index (visual bottom), "left" a low col index.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ----- label content modes ------------------------------------------------- #
LABEL_MZ_PPM = "mz_ppm"     # "463.2463 m/z ± 10 ppm"  (default; matches SCiLS)
LABEL_MZ = "mz"             # "463.2463 m/z"
LABEL_NAME = "mz_name"      # "463.2463 m/z · PC 34:1"
LABEL_MODES = (LABEL_MZ_PPM, LABEL_MZ, LABEL_NAME)
# (display label, mode) pairs for the GUI label-content dropdowns — one source shared by
# the Export hub and Studio dialogs so the two never drift.
LABEL_CHOICES = [("m/z ± ppm", LABEL_MZ_PPM), ("m/z only", LABEL_MZ),
                 ("m/z + lipid name", LABEL_NAME)]


def format_ion_label(mz, *, ppm=None, name="", override=None, mode=LABEL_MZ_PPM) -> str:
    """The one formatter used by the live view, the export engine, and the export dialog.

    A non-empty ``override`` (a per-feature rename) always wins — that's how a wrong lipid
    gets corrected or a label trimmed to taste. Otherwise the text is built from ``mode``:

    * ``LABEL_MZ_PPM`` → ``"463.2463 m/z ± 10 ppm"`` (drops the ± part when ``ppm`` is None)
    * ``LABEL_MZ``     → ``"463.2463 m/z"``
    * ``LABEL_NAME``   → ``"463.2463 m/z · PC 34:1"`` (drops the name when empty)
    """
    if override and str(override).strip():
        return str(override).strip()
    if mz is None:
        return str(name or "intensity")
    base = f"{float(mz):.4f} m/z"
    if mode == LABEL_MZ_PPM and ppm is not None:
        return f"{base} ± {float(ppm):g} ppm"
    if mode == LABEL_NAME:
        nm = str(name or "").strip()
        return f"{base} · {nm}" if nm else base
    return base


# ----- data carriers -------------------------------------------------------- #
@dataclass
class LegendRow:
    """One ion's legend entry. ``color`` drives a black→colour ramp (overlay channels);
    when ``cmap`` is given instead (single-ion view) the named colormap is the gradient.
    ``lo``/``hi`` are the contrast-window percentiles shown under the bar ends; ``max_pct``
    is the trailing relative-max (e.g. ``194``) — omitted when None."""
    label: str
    color: str | None = None
    cmap: str | None = None
    lo: float | None = None
    hi: float | None = None
    max_pct: float | None = None
    checker: bool = True


@dataclass
class Rect:
    """A rectangle in image-frame fractions (origin bottom-left)."""
    x: float
    y: float
    w: float
    h: float


@dataclass
class RowGeom:
    """Resolved per-row placement (all fractions). Text anchors carry their alignment so
    a renderer can place ``ha``/``va`` correctly without recomputing widths."""
    label: str
    label_xy: tuple            # right-aligned, vertically centred on the bar
    bar: Rect
    color: str | None
    cmap: str | None
    checker: Rect | None
    max_pct: float | None
    max_xy: tuple | None       # left-aligned, vertically centred
    lo: float | None
    hi: float | None
    lo_xy: tuple | None        # left-aligned, top-anchored (under the bar's left end)
    hi_xy: tuple | None        # right-aligned, top-anchored (under the bar's right end)


@dataclass
class LegendLayout:
    rows: list = field(default_factory=list)   # list[RowGeom], top row first
    block: Rect = None                          # bounding box of the whole legend
    overflow: int = 0                           # rows not shown (capped)
    font_label_pt: float = 8.5
    font_small_pt: float = 7.0


@dataclass
class ScalebarLayout:
    x0: float
    x1: float
    y: float
    tick_h: float              # half-height of the end ticks (fraction)
    label: str
    label_xy: tuple            # centred above the line
    font_pt: float = 8.5


# ----- geometry constants (tuned; fractions of frame unless noted) ---------- #
_MX = 0.022                    # horizontal margin from the frame edge
_MY = 0.030                    # vertical margin from the frame edge
_BAR_W = 0.115                 # gradient bar width  (of frame width)
_BAR_H = 0.020                 # gradient bar height (of frame height)
_CHECK_W = 0.017               # checkerboard swatch width
_GAP_LABEL = 0.009             # label → checker
_GAP_CHECK = 0.004             # checker → bar
_GAP_TRAIL = 0.007             # bar → trailing %
_TRAIL_W = 0.040               # room reserved for the trailing "194%"
_UNDER_OFF = 0.006             # gap below the bar for lo/hi text
_UNDER_H = 0.020               # height the lo/hi text occupies below a bar
_ROW_GAP = 0.018               # vertical gap between successive rows
_CHAR_W = 0.0118               # ~width of one label char (of frame width) at font_label
_MAX_ROWS = 8
_FONT_LABEL = 8.5
_FONT_SMALL = 7.0


def _label_width(text: str) -> float:
    return max(0.04, _CHAR_W * len(str(text)))


def layout_legend(rows, *, corner="lower right", x_extent=(0.0, 1.0),
                  subject: "Rect | None" = None) -> LegendLayout:
    """Stack ``rows`` (list[LegendRow]) into ``corner``, returning per-row fractional
    placement. No card/background is emitted — the renderer draws text directly on the
    image (with a contrasting ink). When ``subject`` is given the block is nudged so it
    sits beside the tissue (toward the emptier side) rather than over it where possible."""
    rows = list(rows or [])
    overflow = max(0, len(rows) - _MAX_ROWS)
    rows = rows[:_MAX_ROWS]
    out = LegendLayout(rows=[], overflow=overflow,
                       font_label_pt=_FONT_LABEL, font_small_pt=_FONT_SMALL)
    if not rows:
        out.block = Rect(0, 0, 0, 0)
        return out

    x_lo, x_hi = x_extent
    span = x_hi - x_lo
    label_w = max(_label_width(r.label) for r in rows) * span
    bar_w = _BAR_W * span
    check_w = _CHECK_W * span
    trail_w = _TRAIL_W * span if any(r.max_pct is not None for r in rows) else 0.0
    gl, gc, gt = _GAP_LABEL * span, _GAP_CHECK * span, _GAP_TRAIL * span
    block_w = label_w + gl + check_w + gc + bar_w + (gt + trail_w if trail_w else 0.0)

    mx = _MX * span
    left = "left" in (corner or "lower right").lower()
    block_x0 = (x_lo + mx) if left else (x_hi - mx - block_w)

    # columns left → right within the block
    label_anchor_x = block_x0 + label_w           # right edge of the (right-aligned) label
    check_x0 = label_anchor_x + gl
    bar_x0 = check_x0 + check_w + gc
    trail_x = bar_x0 + bar_w + gt

    n = len(rows)
    row_pitch = _BAR_H + _UNDER_H + _ROW_GAP
    lower = "upper" not in (corner or "lower right").lower()
    y_base = (_MY + _UNDER_H) if lower else (1.0 - _MY - row_pitch * n)

    for i, r in enumerate(rows):
        bar_y0 = y_base + (n - 1 - i) * row_pitch          # row 0 at the top
        cy = bar_y0 + _BAR_H / 2.0
        checker = Rect(check_x0, bar_y0, check_w, _BAR_H) if r.checker else None
        out.rows.append(RowGeom(
            label=r.label, label_xy=(label_anchor_x, cy),
            bar=Rect(bar_x0, bar_y0, bar_w, _BAR_H), color=r.color, cmap=r.cmap,
            checker=checker, max_pct=r.max_pct,
            max_xy=((trail_x, cy) if r.max_pct is not None else None),
            lo=r.lo, hi=r.hi,
            lo_xy=((bar_x0, bar_y0 - _UNDER_OFF) if r.lo is not None else None),
            hi_xy=((bar_x0 + bar_w, bar_y0 - _UNDER_OFF) if r.hi is not None else None),
        ))
    out.block = Rect(block_x0, y_base - _UNDER_H, block_w, row_pitch * n)
    return out


def footer_fraction(n_legend_rows=1, *, has_scalebar=True) -> float:
    """Fraction of the figure height to reserve as a bottom **footer band** — a solid,
    theme-coloured strip beneath the image that carries the intensity legend (bottom-right)
    and the scale bar (bottom-left). Putting the annotations in their own band rather than
    floating them over the tissue keeps them in a fixed corner and legible in light or dark
    mode regardless of the image's width/shape (a narrow, tissue-filled frame used to crush
    the legend against the data — this is the fix).

    Sized to contain ``n_legend_rows`` stacked legend rows and/or a scale bar with a small
    margin, in the same figure fractions the layout functions use (so the legend/scale bar
    drawn at their natural bottom-of-figure positions land inside the band). Returns ``0.0``
    when there's nothing to place (no legend rows and no scale bar)."""
    n = max(0, min(int(n_legend_rows or 0), _MAX_ROWS))
    row_pitch = _BAR_H + _UNDER_H + _ROW_GAP
    # top of the legend block: y_base up through the top row's bar, + room for its trailing %
    legend_top = ((_MY + _UNDER_H) + (n - 1) * row_pitch + _BAR_H + 0.012) if n > 0 else 0.0
    # top of the scale-bar caption: bar line + label offset + a line of text
    scalebar_top = ((_MY + 0.015) + 0.018 + 0.020) if has_scalebar else 0.0
    top = max(legend_top, scalebar_top)
    if top <= 0:
        return 0.0
    return float(min(0.5, top + 0.030))            # add a top margin below the image; cap


_NICE_STEPS = (1.0, 2.0, 5.0)
_SCALEBAR_MAX_FRAC = 0.42          # longest a bar may be drawn, as a fraction of the image width


def nice_scalebar_um(physical_width_um, *, target_frac=0.2, max_frac=_SCALEBAR_MAX_FRAC):
    """A '1-2-5 × 10ⁿ' round scale-bar length (µm) for an image whose full width is
    ``physical_width_um``. Aims for ~``target_frac`` of the width and is guaranteed not to
    exceed ``max_frac`` — so the bar always draws at its true length (never clamped, so the
    caption never lies). Returns None for a non-positive width. Single source of truth for
    every renderer's auto length (live overlay, export, crop studio)."""
    if not physical_width_um or physical_width_um <= 0:
        return None
    target = float(physical_width_um) * target_frac
    ceiling = float(physical_width_um) * max_frac
    decade = 10.0 ** math.floor(math.log10(target)) if target > 0 else 1.0
    candidates = [s * d for d in (decade / 10.0, decade, decade * 10.0) for s in _NICE_STEPS]
    fits = [v for v in candidates if 0 < v <= ceiling]
    if not fits:
        return None
    return min(fits, key=lambda v: abs(v - target))


def format_scalebar_label(scale_bar_um) -> str:
    """Human label for a bar length: µm, switching to mm at/above 1000µm (SCiLS-style)."""
    um = float(scale_bar_um)
    return f"{um / 1000.0:g}mm" if um >= 1000.0 else f"{um:g}µm"


def layout_scalebar(*, pixel_size_um, scale_bar_um, img_w_px, corner="lower left",
                    x_extent=(0.0, 1.0), subject: "Rect | None" = None) -> "ScalebarLayout | None":
    """Geometry for a physical scale bar in the bottom corner *opposite* the legend.
    Returns None without a known pixel size / bar length. ``img_w_px`` is the image's pixel
    width so the bar's fractional length is physically correct."""
    if not (pixel_size_um and scale_bar_um and img_w_px):
        return None
    x_lo, x_hi = x_extent
    span = x_hi - x_lo
    n_px = float(scale_bar_um) / float(pixel_size_um)
    frac = (n_px / float(img_w_px)) * span
    if frac > _SCALEBAR_MAX_FRAC * span:
        # the requested bar is too long to draw at its true length — snap to a nice round
        # length that fits and relabel, so the bar and its caption never disagree (the old
        # code clamped the line but kept the original label, drawing a bar that lied).
        fitted = nice_scalebar_um(float(img_w_px) * float(pixel_size_um))
        if not fitted:
            return None
        scale_bar_um = fitted
        n_px = scale_bar_um / float(pixel_size_um)
        frac = (n_px / float(img_w_px)) * span
    if frac <= 0:
        return None
    mx = 0.035 * span
    # bar opposite the legend corner: legend on the right → bar on the left
    on_left = "left" in (corner or "lower left").lower()
    if on_left:
        x0 = x_lo + mx
        x1 = x0 + frac
    else:
        x1 = x_hi - mx
        x0 = x1 - frac
    y = _MY + 0.015
    label = format_scalebar_label(scale_bar_um)
    return ScalebarLayout(x0=x0, x1=x1, y=y, tick_h=0.012, label=label,
                          label_xy=((x0 + x1) / 2.0, y + 0.018), font_pt=8.5)


# ----- subject (tissue) detection + placement ------------------------------- #
def _occ_bool(occupancy) -> "np.ndarray | None":
    a = np.asarray(occupancy)
    if a.ndim != 2 or a.size == 0:
        return None
    if a.dtype != bool:
        a = np.isfinite(a) & (a != 0) if np.issubdtype(a.dtype, np.floating) else a.astype(bool)
    return a if a.any() else None


def detect_subject_bbox(occupancy) -> "Rect | None":
    """Bounding box of the tissue footprint, in image-frame fractions (origin bottom-left,
    y up). ``occupancy`` is an H×W boolean/finite mask of where pixels exist."""
    a = _occ_bool(occupancy)
    if a is None:
        return None
    h, w = a.shape
    rows = np.where(a.any(axis=1))[0]
    cols = np.where(a.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    # array row 0 = top → fraction y is flipped so y increases up
    return Rect(x=c0 / w, y=1.0 - r1 / h, w=(c1 - c0) / w, h=(r1 - r0) / h)


def empty_corner(occupancy, default="lower right") -> str:
    """Pick the bottom/side corner with the most empty margin beside the subject, so the
    legend lands in a margin rather than over the tissue. Returns one of
    ``"lower left" | "lower right" | "upper left" | "upper right"``."""
    a = _occ_bool(occupancy)
    if a is None:
        return default
    h, w = a.shape
    rows = np.where(a.any(axis=1))[0]
    cols = np.where(a.any(axis=0))[0]
    top = int(rows[0])                       # empty rows above subject
    bottom = h - 1 - int(rows[-1])           # empty rows below subject
    leftm = int(cols[0])                     # empty cols left of subject
    rightm = w - 1 - int(cols[-1])           # empty cols right of subject
    vert = "lower" if bottom >= top else "upper"
    horiz = "right" if rightm >= leftm else "left"
    return f"{vert} {horiz}"


def opposite_corner(corner: str) -> str:
    """The diagonally-opposite corner (legend ↔ scale bar)."""
    c = (corner or "lower right").lower()
    vert = "upper" if "lower" in c else "lower"
    horiz = "left" if "right" in c else "right"
    return f"{vert} {horiz}"


# ----- legibility ----------------------------------------------------------- #
INK_DARK = ("#ffffff", (0.0, 0.0, 0.0, 0.55))      # white ink for a dark backing
INK_LIGHT = ("#101418", (1.0, 1.0, 1.0, 0.70))     # near-black ink for a light backing


def pick_ink(patch, default=INK_DARK):
    """Choose legible ink + a contrasting stroke from the mean luminance of ``patch`` (the
    image region the annotation sits over). Returns ``(ink_hex, stroke_rgba)``:
    white-on-dark with a dark stroke, or near-black-on-light with a light stroke. Robust to
    grayscale, RGB, or RGBA (transparent pixels are ignored), and to 0–1 or 0–255 ranges.
    When the patch is empty/fully transparent (no luminance to judge), returns ``default`` —
    callers pass a theme-appropriate ink so a transparent margin still reads."""
    a = np.asarray(patch, dtype=float)
    if a.ndim == 2 and a.shape[-1] in (3, 4):
        a = a[:, None, :]                       # a flat list of RGB(A) pixels → (K,1,C)
    if a.ndim == 3:
        rgb = a[..., :3]
        if a.shape[-1] >= 4:
            alpha = a[..., 3]
            amax = alpha.max() if alpha.size else 0.0
            keep = alpha > (0.06 * (255.0 if amax > 1.5 else 1.0))
        else:
            keep = np.ones(a.shape[:2], bool)
        lum = rgb.mean(axis=2)
    else:
        lum = a
        keep = np.isfinite(lum)
    vals = lum[keep & np.isfinite(lum)] if lum.size else np.array([])
    if not vals.size:                           # nothing to judge (empty / transparent)
        return default
    scale = 255.0 if float(vals.max()) > 1.5 else 1.0
    m = float(vals.mean()) / scale
    return INK_DARK if m < 0.5 else INK_LIGHT
