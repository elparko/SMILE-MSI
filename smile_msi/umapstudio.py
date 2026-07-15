"""UMAP Studio engine — turn a 2-D embedding into a **publication-quality** figure.

The single-slide :func:`~smile_msi.multivariate.embedding` and the cohort
:func:`~smile_msi.multivariate.pooled_embedding` give us the *coordinates*; this module
turns millions of those points into a figure that reads like a journal panel (e.g. Farrow
et al. 2025, Sci. Adv. 11, eadu3730, Fig 4 — a kidney lipid atlas embedding coloured by
donor and by tissue structure).

**Why a dedicated renderer.** A normal scatter plot (one marker per point) cannot draw
millions of pixels: they overplot into an opaque blob with no density texture, and most
plotting back-ends choke long before that. The atlas "painterly" look is instead
**categorical density compositing**: the plane is binned into an output-pixel grid, and
each output pixel's

* **colour** is the *count-weighted blend* of the category colours that landed in it (so
  overlapping donors mix into an intermediate hue), and
* **alpha** is its *total count*, histogram-equalised (``how='eq_hist'``) so both the sparse
  halo and the dense core show gradient instead of the core saturating into one flat patch.

This is the recipe the ``datashader`` library popularised (``count_cat`` + ``tf.shade``); we
implement it in ~a screenful of NumPy/SciPy so it works in the lean frozen build with no
heavy new dependency (``datashader`` pulls numba/dask/xarray). Provenance note: datashader
is the *style* we reproduce, **not** a documented part of the cited paper's method — the
paper names only umap-learn + scikit-learn for the embedding itself.

The module is **pure**: NumPy + SciPy + Matplotlib's object-oriented API only (no Qt, no
pyplot global state), so it renders headlessly and in worker threads. The live editor
(:mod:`smile_msi.gui.umapstudiodialog`) builds the *same* :class:`Figure` for its on-screen
preview as it exports — what you see is what you save (WYSIWYG), exactly like the ion-image
:mod:`smile_msi.annotations` legend model.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field, asdict

import numpy as np

# Alpha-mapping modes (how a pixel's total count becomes its opacity).
HOW_EQ_HIST = "eq_hist"     # rank/histogram-equalised — both halo and core show gradient
HOW_LINEAR = "linear"       # alpha ∝ count (the dense core saturates flat)
HOW_LOG = "log"             # alpha ∝ log1p(count)
HOW_MODES = (HOW_EQ_HIST, HOW_LINEAR, HOW_LOG)

# Rendering style. ``density`` is the atlas categorical-density compositing (right for millions
# of pooled pixels); ``scatter`` draws true coloured dots + legend (the crisp look of the live
# Cohort UMAP plot — right for region/sample means and modest pixel counts).
MODE_DENSITY = "density"
MODE_SCATTER = "scatter"
MODES = (MODE_DENSITY, MODE_SCATTER)

BACKGROUNDS = ("white", "black", "transparent")

# Axis treatments. Embedding coordinates are not metric, so ticks are meaningless — the
# defaults strip them. ``corner`` keeps an L of labelled spines; ``off`` removes everything.
AXIS_CORNER = "corner"
AXIS_LABELED = "labeled"
AXIS_OFF = "off"
AXIS_STYLES = (AXIS_CORNER, AXIS_LABELED, AXIS_OFF)

LAYOUTS = {                 # name -> (subplot_mosaic string, panel count)
    "single": ("A", 1),
    "two": ("AB", 2),
    "four": ("AB\nCD", 4),
}


# --------------------------------------------------------------------------- #
# Palettes — one fixed colour per category, reused across every panel
# --------------------------------------------------------------------------- #
# Hand-picked categorical sets for *few* categories (the FTU/region case ≤ ~10). Okabe-Ito
# is the colourblind-safe reference set; tab10/Set2 match Matplotlib/seaborn conventions.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#F0E442", "#000000"]
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
SET2 = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
        "#ffd92f", "#e5c494", "#b3b3b3"]

PALETTES = ("auto", "glasbey", "tab10", "okabe_ito", "set2")


def _glasbey_like(n: int) -> list:
    """``n`` maximally-distinct hex colours for *many* categories (e.g. 28 donors), when a
    fixed 8–10 colour set would recycle and make neighbours indistinguishable.

    Uses ``colorcet.glasbey_category10`` when the optional ``colorcet`` extra is installed
    (the reference perceptually-spaced sequence), else falls back to a golden-ratio hue walk
    with alternating lightness/saturation — not as optimal as glasbey but still well spread.
    """
    try:                                       # optional: the real glasbey sequence
        import colorcet as cc

        seq = cc.glasbey_category10
        if n <= len(seq):
            return [c if isinstance(c, str) else _rgb_to_hex(c) for c in seq[:n]]
    except Exception:  # noqa: BLE001 — colorcet absent or API drift → procedural fallback
        pass
    out = []
    h = 0.0
    for i in range(n):
        h = (h + 0.61803398875) % 1.0          # golden-ratio hue spacing → low collision
        light = 0.62 if i % 2 == 0 else 0.46    # alternate value so adjacent hues separate
        sat = 0.78 if i % 3 else 0.60
        out.append(_rgb_to_hex(colorsys.hsv_to_rgb(h, sat, light)))
    return out


def _rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(c) * 255)) for c in rgb[:3])
    return "#%02x%02x%02x" % (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def categorical_palette(n: int, name: str = "auto") -> list:
    """``n`` hex colours for ``n`` categories from the named palette.

    ``auto`` picks tab10 for ≤10 categories and a glasbey-like sequence above that — the
    right default for "few FTU classes" vs "many donors". Fixed sets cycle (with a warning
    left to the caller) only if explicitly chosen for more categories than they hold.
    """
    name = (name or "auto").lower()
    if name == "auto":
        return TAB10[:n] if n <= len(TAB10) else _glasbey_like(n)
    if name == "glasbey":
        return _glasbey_like(n)
    base = {"tab10": TAB10, "okabe_ito": OKABE_ITO, "set2": SET2}.get(name, TAB10)
    return [base[i % len(base)] for i in range(n)]


def build_color_key(categories, palette: str = "auto") -> dict:
    """Map each *unique* category (first-seen order) to a fixed hex colour. Build this once
    and pass the same dict to every panel/legend so a donor/FTU is the same colour
    everywhere — the consistency the multi-panel atlas figure depends on."""
    uniq = list(dict.fromkeys(np.asarray(categories, dtype=object).tolist()))
    cols = categorical_palette(len(uniq), palette)
    return {u: cols[i] for i, u in enumerate(uniq)}


# --------------------------------------------------------------------------- #
# The rasterizer — categorical / continuous density compositing
# --------------------------------------------------------------------------- #
def _bin_indices(coords, width, height, x_range, y_range):
    """Per-point output-pixel (ix, iy) with **iy=0 at the bottom** (so the RGBA is drawn with
    ``origin='lower'``), plus a validity mask for points inside the range."""
    x = np.asarray(coords[:, 0], dtype=float)
    y = np.asarray(coords[:, 1], dtype=float)
    (x0, x1), (y0, y1) = x_range, y_range
    sx = (width - 1) / (x1 - x0) if x1 > x0 else 0.0
    sy = (height - 1) / (y1 - y0) if y1 > y0 else 0.0
    ix = np.rint((x - x0) * sx).astype(np.intp)
    iy = np.rint((y - y0) * sy).astype(np.intp)
    ok = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    return ix, iy, ok


def _alpha_from_counts(total, how: str, min_alpha: float) -> np.ndarray:
    """Map a per-pixel total count → opacity in ``[min_alpha, 1]`` (0 where empty).

    ``eq_hist`` ranks the occupied pixels and spreads them evenly across the alpha range —
    the key to seeing both faint outliers and dense cores. ``linear``/``log`` are the
    simpler ramps (linear lets the core saturate flat, which is usually what you *don't*
    want for an atlas)."""
    total = np.asarray(total, dtype=float)
    a = np.zeros_like(total)
    pos = total > 0
    if not pos.any():
        return a
    t = total[pos]
    if how == HOW_LINEAR:
        v = t / t.max()
    elif how == HOW_LOG:
        v = np.log1p(t) / np.log1p(t.max())
    else:                                       # eq_hist (default)
        ranks = np.argsort(np.argsort(t))       # 0..m-1, ties broken arbitrarily (fine here)
        v = (ranks + 1.0) / float(len(t))
    a[pos] = float(min_alpha) + (1.0 - float(min_alpha)) * v
    return a


def _spread_stack(stack2d_list, spread_px: int):
    """Fatten each per-category count grid by a square max-filter so lone points survive
    print downscaling (datashader's ``dynspread`` idea). Spreading the *counts* (not the
    final image) keeps the colour blend correct."""
    if spread_px <= 0:
        return stack2d_list
    from scipy.ndimage import maximum_filter

    k = 2 * int(spread_px) + 1
    return [maximum_filter(g, size=k, mode="constant") for g in stack2d_list]


def _composite(color_rgb, alpha, width, height, background: str) -> np.ndarray:
    """Alpha-composite the per-pixel colour over the background → an ``(H, W, 4)`` uint8
    RGBA with **row 0 = bottom**. White/black backgrounds yield an opaque image; the
    ``transparent`` background keeps per-pixel alpha so it can layer over anything."""
    bg = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0)}.get(background, (1.0, 1.0, 1.0))
    a = alpha.reshape(height, width, 1)
    col = color_rgb.reshape(height, width, 3)
    out = np.asarray(bg).reshape(1, 1, 3) * (1.0 - a) + col * a
    if background == "transparent":
        out_a = alpha.reshape(height, width, 1)
    else:
        out_a = np.ones((height, width, 1), dtype=float)
    rgba = np.concatenate([np.clip(out, 0, 1), out_a], axis=2)
    return (rgba * 255.0 + 0.5).astype(np.uint8)


def shade_categorical(coords, categories, color_key, *, width=700, height=700,
                      x_range=None, y_range=None, how=HOW_EQ_HIST, min_alpha=0.18,
                      spread_px=0, background="white") -> np.ndarray:
    """Render points coloured by a **categorical** label into an ``(H, W, 4)`` RGBA raster.

    ``color_key`` maps each category to a hex colour (build it once via
    :func:`build_color_key`). Output pixel colour is the count-weighted blend of the
    categories present; opacity follows ``how`` (see :func:`_alpha_from_counts`). The
    returned array is drawn with ``imshow(origin='lower', extent=[x0,x1,y0,y1])``.
    """
    from matplotlib.colors import to_rgb

    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    cats = np.asarray(categories, dtype=object)
    x_range = x_range or _nice_range(coords[:, 0])
    y_range = y_range or _nice_range(coords[:, 1])
    ix, iy, ok = _bin_indices(coords, width, height, x_range, y_range)

    keys = list(color_key.keys())
    code = {k: i for i, k in enumerate(keys)}
    colors = np.array([to_rgb(color_key[k]) for k in keys], dtype=float)  # (C,3)
    flat = (iy * width + ix)[ok]
    cat_codes = np.array([code.get(c, -1) for c in cats[ok]], dtype=np.intp)

    grids = []                                  # one (H*W,) count grid per category
    for c in range(len(keys)):
        sel = cat_codes == c
        grids.append(np.bincount(flat[sel], minlength=width * height).astype(float)
                     if sel.any() else np.zeros(width * height))
    if spread_px > 0:
        grids = [g.reshape(height, width) for g in grids]
        grids = [g.ravel() for g in _spread_stack(grids, spread_px)]

    counts = np.vstack(grids) if grids else np.zeros((1, width * height))   # (C, H*W)
    total = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        color = (colors.T @ counts) / np.where(total > 0, total, 1.0)        # (3, H*W)
    color_rgb = color.T                                                      # (H*W, 3)
    alpha = _alpha_from_counts(total, how, min_alpha)
    return _composite(color_rgb, alpha, width, height, background)


def shade_continuous(coords, values, *, cmap="viridis", width=700, height=700,
                     x_range=None, y_range=None, how=HOW_EQ_HIST, min_alpha=0.18,
                     spread_px=0, background="white", clip=(2.0, 98.0),
                     vmin=None, vmax=None):
    """Render points coloured by a **continuous** value (a lipid/feature intensity, a SHAP
    importance) into an ``(H, W, 4)`` RGBA raster. Each output pixel takes the *mean* value
    of the points in it, mapped through ``cmap``; opacity follows the point count via ``how``.

    Returns ``(rgba, (vlo, vhi))`` so a colourbar can be drawn with the true value window
    (the ``clip`` percentiles, unless ``vmin``/``vmax`` are given)."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float)
    x_range = x_range or _nice_range(coords[:, 0])
    y_range = y_range or _nice_range(coords[:, 1])
    ix, iy, ok = _bin_indices(coords, width, height, x_range, y_range)
    flat = (iy * width + ix)[ok]
    v = values[ok]

    sum_v = np.bincount(flat, weights=v, minlength=width * height)
    cnt = np.bincount(flat, minlength=width * height).astype(float)
    if spread_px > 0:
        sum_v = _spread_stack([sum_v.reshape(height, width)], spread_px)[0].ravel()
        cnt = _spread_stack([cnt.reshape(height, width)], spread_px)[0].ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, sum_v / np.where(cnt > 0, cnt, 1.0), 0.0)

    finite = v[np.isfinite(v)]
    if vmin is None or vmax is None:
        lo, hi = (np.percentile(finite, clip) if finite.size else (0.0, 1.0))
    else:
        lo, hi = float(vmin), float(vmax)
    if hi <= lo:
        hi = lo + 1e-9
    cm = colormaps[cmap] if cmap in colormaps else colormaps["viridis"]
    norm = Normalize(vmin=lo, vmax=hi)
    color_rgb = cm(norm(mean))[:, :3]
    alpha = _alpha_from_counts(cnt, how, min_alpha)
    return _composite(color_rgb, alpha, width, height, background), (float(lo), float(hi))


# --------------------------------------------------------------------------- #
# The scatter renderer — crisp coloured dots (the live-plot look), for modest N
# --------------------------------------------------------------------------- #
def scatter_categorical(ax, coords, categories, color_key, *, size=14.0, alpha=0.85):
    """Draw points coloured by a **categorical** label as a true scatter — one colour per
    category, the crisp "dots + legend" look of the live Cohort UMAP plot. Best for region/
    sample means and modest pixel counts (millions of points overplot — use density there).
    The dots are ``rasterized`` so a vector PDF/SVG stays small while text/axes stay vector."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    cats = np.asarray(categories, dtype=object)
    for k, col in color_key.items():
        m = cats == k
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=float(size), c=col, alpha=float(alpha),
                       edgecolors="none", linewidths=0, rasterized=True)


def scatter_continuous(ax, coords, values, *, cmap="viridis", size=14.0, alpha=0.85,
                       clip=(2.0, 98.0), vmin=None, vmax=None):
    """Draw points coloured by a **continuous** value (lipid intensity) as a scatter, mapped
    through ``cmap``. Returns the ``(vlo, vhi)`` value window so a colourbar matches."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if vmin is None or vmax is None:
        lo, hi = (np.percentile(finite, clip) if finite.size else (0.0, 1.0))
    else:
        lo, hi = float(vmin), float(vmax)
    if hi <= lo:
        hi = lo + 1e-9
    ax.scatter(coords[:, 0], coords[:, 1], s=float(size), c=v, cmap=cmap, vmin=lo, vmax=hi,
               alpha=float(alpha), edgecolors="none", linewidths=0, rasterized=True)
    return (float(lo), float(hi))


def _nice_range(v, pad=0.04):
    """A finite (lo, hi) range padded a touch, robust to all-equal or non-finite input."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return (lo - 1.0, lo + 1.0)
    m = (hi - lo) * pad
    return (lo - m, hi + m)


def shared_ranges(coords, pad=0.04):
    """The ``(x_range, y_range)`` covering all points — locked across panels so every panel
    of a multi-panel figure is directly comparable (the same cloud, different colouring)."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    return _nice_range(coords[:, 0], pad), _nice_range(coords[:, 1], pad)


# --------------------------------------------------------------------------- #
# Data carrier + figure spec
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingData:
    """The inputs a figure needs: 2-D ``coords`` plus the channels you can colour by.

    ``categorical`` maps a channel name (e.g. "Sample", "Group", "Region") to a per-point
    label array; ``continuous`` maps a name (e.g. "744.55 m/z") to a per-point float array.
    ``features``/``feature_mz`` optionally retain the per-pixel feature matrix so the editor
    can *add* a continuous channel for any feature on demand (see :meth:`set_feature_channel`)
    without recomputing the embedding."""
    coords: np.ndarray
    method: str = "UMAP"
    categorical: dict = field(default_factory=dict)
    continuous: dict = field(default_factory=dict)
    features: np.ndarray | None = None
    feature_mz: np.ndarray | None = None
    feature_names: list | None = None
    counts: dict | None = None

    def channels(self) -> list:
        return list(self.categorical) + list(self.continuous)

    def is_categorical(self, name: str) -> bool:
        return name in self.categorical

    def feature_label(self, j: int) -> str:
        """Display label for feature column ``j`` — ``"744.5481 m/z"`` plus a lipid name when
        one is known (so the channel combo reads like the rest of the app)."""
        mz = float(self.feature_mz[j])
        base = f"{mz:.4f} m/z"
        if self.feature_names and j < len(self.feature_names) and self.feature_names[j]:
            return f"{base} · {self.feature_names[j]}"
        return base

    def set_feature_channel(self, j: int) -> str:
        """Add (or refresh) the continuous channel for feature column ``j`` and return its
        name, so the editor can colour by any lipid/feature intensity. Cheap — just a column
        slice of the retained matrix."""
        if self.features is None or self.feature_mz is None:
            raise ValueError("no feature matrix retained for continuous colour-by")
        name = self.feature_label(j)
        self.continuous[name] = np.asarray(self.features[:, j], dtype=float)
        return name


@dataclass
class Annotation:
    """A draggable on-figure mark in **data coordinates** so it stays glued to the cloud as
    the figure is rebuilt/exported. ``kind`` is cosmetic: ``label`` (cluster/free text),
    ``panel`` (a big A/B/C/D corner letter), or ``title``."""
    text: str
    x: float
    y: float
    panel: int = 0
    kind: str = "label"
    fontsize: float = 12.0
    color: str = "#111111"
    bold: bool = False
    ha: str = "center"
    va: str = "center"


@dataclass
class PanelSpec:
    """One panel: which channel it colours by, its title, corner letter, and (for a
    continuous channel) the colormap. ``title_xy``/``letter_xy`` are optional axes-fraction
    positions set by dragging the title / A–D letter in the editor (``None`` = default spot)."""
    color_by: str = ""
    title: str = ""
    panel_letter: str = ""
    cmap: str = "viridis"
    title_xy: tuple | None = None
    letter_xy: tuple | None = None


@dataclass
class UMAPStudioSpec:
    """A complete, serialisable description of the figure — everything the editor's controls
    set. Round-trips to a dict for session persistence (``session['_umap_studio_spec']``)."""
    panels: list = field(default_factory=lambda: [PanelSpec()])
    layout: str = "single"
    mode: str = MODE_DENSITY            # density (atlas cloud) or scatter (crisp dots)
    palette: str = "auto"
    # per-channel, per-category colour overrides {channel: {category: '#rrggbb'}} — picked in
    # the editor's "Category colours" section; win over ``palette`` (see render_figure).
    color_overrides: dict = field(default_factory=dict)
    how: str = HOW_EQ_HIST
    min_alpha: float = 0.18
    spread_px: int = 0
    point_size: float = 14.0            # scatter: marker area (pts²)
    point_alpha: float = 0.85           # scatter: per-dot opacity
    background: str = "white"
    resolution: int = 700
    axis_style: str = AXIS_CORNER
    legend: bool = True
    legend_loc: str = "right"
    # per-channel legend anchor (figure fraction, upper-left) set by dragging the legend
    legend_xy: dict = field(default_factory=dict)
    dpi: int = 600
    annotations: list = field(default_factory=list)
    # recorded embedding parameters (provenance + re-embed presets)
    metric: str = "cosine"
    n_neighbors: int = 15
    min_dist: float = 0.1
    seed: int = 0

    def n_panels(self) -> int:
        return LAYOUTS.get(self.layout, LAYOUTS["single"])[1]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["panels"] = [asdict(p) for p in self.panels]
        d["annotations"] = [asdict(a) for a in self.annotations]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UMAPStudioSpec":
        d = dict(d or {})
        pk = set(PanelSpec.__dataclass_fields__)
        ak = set(Annotation.__dataclass_fields__)
        d["panels"] = [PanelSpec(**{k: v for k, v in p.items() if k in pk})
                       for p in d.get("panels", [])] or [PanelSpec()]
        d["annotations"] = [Annotation(**{k: v for k, v in a.items() if k in ak})
                            for a in d.get("annotations", [])]
        known = {f for f in cls.__dataclass_fields__}            # ignore unknown future keys
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# Figure assembly — the same Figure the live editor previews and exports
# --------------------------------------------------------------------------- #
def _style_axis(ax, style, xlabel, ylabel, fg="#222222", border=None, width_in=3.4):
    """Minimal embedding axes (DESIGN_SPEC §5): a faint 1px hairline panel border framing the
    plane, no ticks, with UMAP-1/UMAP-2 corner labels (``corner``) or full labels (``labeled``)."""
    from . import export as _export
    ax.set_aspect("equal")
    if style == AXIS_OFF:
        ax.set_axis_off()
        return
    ax.set_xticks([])
    ax.set_yticks([])
    # 1px hairline border around the whole panel (all four sides), so the plane reads as a card
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(border if border is not None else fg)
        sp.set_linewidth(1.0)
    if style == AXIS_LABELED:
        ax.set_xlabel(xlabel, color=fg, fontsize=_export.type_pt("axis", width_in))
        ax.set_ylabel(ylabel, color=fg, fontsize=_export.type_pt("axis", width_in))
    else:                                       # corner: tiny axis names, no box
        ax.set_xlabel(xlabel, color=fg, fontsize=_export.type_pt("tick", width_in), labelpad=2)
        ax.set_ylabel(ylabel, color=fg, fontsize=_export.type_pt("tick", width_in), labelpad=2)


def render_figure(data: EmbeddingData, spec: UMAPStudioSpec, *, color_keys=None, fig=None,
                  resize=True):
    """Build the publication :class:`matplotlib.figure.Figure` for ``spec`` over ``data``.

    Panels share one locked x/y range (so they're comparable) and, for any categorical
    channel, one colour key (so a category is the same colour in every panel + legend).
    ``color_keys`` lets a caller pin keys (e.g. the editor keeps them stable while you tweak
    other controls); otherwise they're built from ``spec.palette``. Pass ``fig`` to draw into
    an existing figure (the live editor reuses one embedded canvas instead of churning a new
    one each tweak); omit it to get a fresh Agg-backed figure. Returns the Figure — the editor
    embeds its canvas for preview and hands the same figure to
    :func:`smile_msi.export.save_figure`, so preview and export are identical (WYSIWYG).

    ``resize`` (default True) sets the figure to the computed publication size. Pass
    ``resize=False`` when ``fig`` is a **live embedded canvas** (the on-screen editor preview):
    such a figure is sized by its Qt widget, and forcing an absolute figsize would shrink the
    Agg buffer below the widget — Matplotlib's ``paintEvent`` then reads out-of-bounds backing
    memory for the uncovered strip and paints sheared "garbage dots" there. With ``resize=False``
    we keep the widget-driven size and lay the mosaic + legend out as fractions of that width."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    coords = np.asarray(data.coords, dtype=float).reshape(-1, 2)
    x_range, y_range = shared_ranges(coords)
    mosaic, n_slots = LAYOUTS.get(spec.layout, LAYOUTS["single"])
    panels = list(spec.panels)[:n_slots]
    while len(panels) < n_slots:                # pad so the mosaic always fills
        panels.append(PanelSpec(color_by=(panels[0].color_by if panels else "")))

    # Resolve each panel to a channel that actually exists on THIS embedding. A remembered
    # feature/lipid colour-by from a previous embedding (never re-materialised here) would
    # otherwise hit the empty branch and render a *blank* panel — fall back to the first real
    # channel so the figure always shows the cloud the user generated.
    fallback = _first_renderable_channel(data)
    color_by = [c if (data.is_categorical(c) or c in data.continuous) else fallback
                for c in (p.color_by for p in panels)]

    # one shared colour key per distinct categorical channel used by the panels
    color_keys = dict(color_keys or {})
    for ch in color_by:
        if ch and data.is_categorical(ch) and ch not in color_keys:
            color_keys[ch] = build_color_key(data.categorical[ch], spec.palette)
    # User-picked per-category colours win over the palette (applied last so they also
    # override caller-pinned keys, e.g. the live editor's stable keys). Only categories
    # that exist on this embedding are touched, so a stale override can't inject a phantom.
    for ch, overrides in (getattr(spec, "color_overrides", None) or {}).items():
        if ch in color_keys and isinstance(overrides, dict):
            for cat, hexcol in overrides.items():
                if cat in color_keys[ch] and hexcol:
                    color_keys[ch][cat] = hexcol

    from . import export as _export
    _export._ensure_fonts()                     # Helvetica family, shared with every render_*
    bg_is_dark = spec.background == "black"
    fg = "#f0f0f0" if bg_is_dark else "#16191d"
    page = "black" if bg_is_dark else "white"
    border = (1, 1, 1, 0.16) if bg_is_dark else "#e6e8eb"   # 1px hairline panel border (spec §5)
    legend_channels = _distinct_cat_channels(color_by, data) if spec.legend else []
    n_cols = (1 if spec.layout == "single" else 2)
    n_rows = mosaic.count("\n") + 1

    # Reserve a real right margin (in figure fractions) for legends/colorbars, sized in
    # inches, and confine the panel mosaic to the left of it — so legends sit *inside* the
    # canvas and a plain savefig (no bbox_inches='tight') never clips them.
    panel_in = 3.4
    has_cont = _has_continuous(color_by, data)
    needs_margin = spec.legend and (bool(legend_channels) or has_cont)
    margin_in = 1.95 if needs_margin else 0.35
    fig_w = panel_in * n_cols + margin_in
    fig_h = panel_in * n_rows + 0.4
    if fig is None:
        fig = Figure(figsize=(fig_w, fig_h), dpi=spec.dpi, facecolor=page)
        FigureCanvasAgg(fig)                    # OO canvas for headless render/export
    else:
        fig.clear()                             # reuse the editor's embedded canvas figure
        if resize:
            fig.set_size_inches(fig_w, fig_h)
        else:                                   # embedded live canvas owns its own size
            fig_w, fig_h = (float(v) for v in fig.get_size_inches())
        fig.set_facecolor(page)
    # Reserve the legend/colorbar margin as a fraction of the ACTUAL figure width, clamped so a
    # very narrow preview can't drive the panel to zero/negative width (never bites at the fixed
    # publication size, where the ratio is ~0.36 at most).
    gs_right = 1.0 - min(margin_in / fig_w, 0.6)
    axd = fig.subplot_mosaic(mosaic, gridspec_kw=dict(
        left=0.06, right=gs_right - 0.02, top=0.93, bottom=0.07, wspace=0.12, hspace=0.14))
    letters = list("ABCD")
    res = int(spec.resolution)
    xlabel, ylabel = f"{data.method} 1", f"{data.method} 2"

    is_scatter = (spec.mode == MODE_SCATTER)
    draggables = []                             # (artist, kind, key) — editor drag, persists in spec
    panel_axes = []
    cont_items = []                             # (cmap, vlo, vhi, channel) for colorbars
    for i, p in enumerate(panels):
        ax = axd[letters[i]]
        ax.set_facecolor(page)
        panel_axes.append(ax)
        ch = color_by[i]
        if ch and data.is_categorical(ch):
            if is_scatter:
                scatter_categorical(ax, coords, data.categorical[ch], color_keys[ch],
                                    size=spec.point_size, alpha=spec.point_alpha)
            else:
                rgba = shade_categorical(coords, data.categorical[ch], color_keys[ch],
                                         width=res, height=res, x_range=x_range, y_range=y_range,
                                         how=spec.how, min_alpha=spec.min_alpha,
                                         spread_px=spec.spread_px, background=spec.background)
                ax.imshow(rgba, origin="lower",
                          extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                          aspect="equal", interpolation="nearest")
        elif ch and ch in data.continuous:
            if is_scatter:
                vlo, vhi = scatter_continuous(ax, coords, data.continuous[ch], cmap=p.cmap,
                                              size=spec.point_size, alpha=spec.point_alpha)
            else:
                rgba, (vlo, vhi) = shade_continuous(coords, data.continuous[ch], cmap=p.cmap,
                                                    width=res, height=res, x_range=x_range,
                                                    y_range=y_range, how=spec.how,
                                                    min_alpha=spec.min_alpha,
                                                    spread_px=spec.spread_px,
                                                    background=spec.background)
                ax.imshow(rgba, origin="lower",
                          extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                          aspect="equal", interpolation="nearest")
            cont_items.append((p.cmap, vlo, vhi, ch))
        # scatter sets its own data limits; lock every panel to the shared range so panels stay
        # directly comparable (and an empty panel still shows the framed plane).
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        _style_axis(ax, spec.axis_style, xlabel, ylabel, fg=fg, border=border, width_in=panel_in)
        if spec.layout != "single" and i < n_slots:
            lx0, ly0 = p.letter_xy if p.letter_xy else (0.02, 0.98)
            t = ax.text(lx0, ly0, p.panel_letter or letters[i], transform=ax.transAxes,
                        ha="left", va="top", fontsize=_export.type_pt("title", panel_in) + 1.5,
                        fontweight="bold", color=fg, zorder=21, picker=True)
            draggables.append((t, "letter", i))
        if p.title:                              # an ax.text (not set_title) so it can be dragged
            # panel title top-left, semibold (DESIGN_SPEC §5); below the letter when one is shown
            has_letter = (spec.layout != "single" and i < n_slots)
            default_xy = (0.14, 0.965) if has_letter else (0.02, 0.965)
            tx0, ty0 = p.title_xy if p.title_xy else default_xy
            t = ax.text(tx0, ty0, p.title, transform=ax.transAxes, ha="left", va="top",
                        color=fg, fontsize=_export.type_pt("title", panel_in), fontweight="semibold",
                        zorder=21, picker=True)
            draggables.append((t, "title", i))

    # Stack legends (categorical) then colorbars (continuous) down the reserved right margin.
    # Each item's vertical slot is sized **proportional to its content** (a 28-donor legend
    # needs far more room than a 5-class one), so a long legend can't overrun a fixed slot and
    # collide with the next. Everything stays within [gs_right, 1.0] so a plain savefig never
    # clips it. Wide legends spill into 2 columns to stay short.
    if needs_margin:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        items = [("legend", ch) for ch in legend_channels] + \
                [("cbar", it) for it in cont_items]
        ncols = {ch: (2 if len(color_keys[ch]) > 16 else 1) for ch in legend_channels}
        top, bottom = 0.95, 0.05
        # weight ≈ visual height: a legend's rows (per column) + title; a colorbar a fixed share
        weights = []
        for kind, payload in items:
            if kind == "legend":
                rows = -(-len(color_keys[payload]) // ncols[payload])   # ceil rows per column
                weights.append(rows + 1.6)
            else:
                weights.append(6.0)
        wsum = sum(weights) or 1.0
        avail = top - bottom
        lx = gs_right + 0.015
        y = top
        for (kind, payload), wt in zip(items, weights):
            h = avail * wt / wsum
            if kind == "legend":
                ch = payload
                handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                                  markerfacecolor=c, markeredgecolor="none", label=str(k))
                           for k, c in color_keys[ch].items()]
                anchor = (spec.legend_xy or {}).get(ch) or (lx, y)   # dragged spot, else auto
                leg = fig.legend(handles=handles, title=ch, loc="upper left",
                                 bbox_to_anchor=tuple(anchor), bbox_transform=fig.transFigure,
                                 frameon=False, fontsize=_export.type_pt("legend", panel_in),
                                 title_fontsize=_export.type_pt("subtitle", panel_in),
                                 ncol=ncols[ch], labelcolor=fg, markerscale=1.1,
                                 borderaxespad=0.0, handletextpad=0.3, columnspacing=0.8,
                                 labelspacing=0.3)
                leg.get_title().set_color(fg)
                leg.set_picker(True)
                draggables.append((leg, "legend", ch))
            else:
                cmap, vlo, vhi, ch = payload
                cax = fig.add_axes([lx, y - h * 0.82, 0.02, h * 0.66])
                cb = fig.colorbar(ScalarMappable(norm=Normalize(vlo, vhi), cmap=cmap), cax=cax)
                cb.set_label(ch, color=fg, fontsize=_export.type_pt("legend", panel_in))
                cb.ax.tick_params(colors=fg, labelsize=_export.type_pt("annotation", panel_in))
                cb.outline.set_edgecolor(border)
                cb.outline.set_linewidth(1.0)
            y -= h

    # annotations (data coords) on their assigned panel; keep (artist, ann) so the live
    # editor can drag/rename them without a full rebuild.
    ann_artists = []
    for k, ann in enumerate(spec.annotations):
        idx = max(0, min(int(ann.panel), len(panel_axes) - 1))
        ax = panel_axes[idx]
        t = ax.text(ann.x, ann.y, ann.text, ha=ann.ha, va=ann.va, color=ann.color,
                    fontsize=ann.fontsize, fontweight=("bold" if ann.bold else "normal"),
                    zorder=22, clip_on=False, picker=True)
        ann_artists.append((t, ann))
        draggables.append((t, "annotation", k))

    fig._umap_color_keys = color_keys           # stash so the editor can reuse stable keys
    fig._umap_ranges = (x_range, y_range)
    fig._umap_panel_axes = panel_axes
    fig._umap_annotation_artists = ann_artists
    fig._umap_draggables = draggables           # (artist, kind, key) for the live editor's drag
    return fig


def _first_renderable_channel(data) -> str:
    """The first channel that actually exists on ``data`` — categorical preferred (it gives a
    legend), else continuous. Used as the safety fallback so a panel never renders blank."""
    cats = list(data.categorical)
    if cats:
        return cats[0]
    conts = list(data.continuous)
    return conts[0] if conts else ""


def _distinct_cat_channels(channels, data) -> list:
    seen = []
    for ch in channels:
        if ch and data.is_categorical(ch) and ch not in seen:
            seen.append(ch)
    return seen


def _has_continuous(channels, data) -> bool:
    return any(ch in data.continuous for ch in channels)
