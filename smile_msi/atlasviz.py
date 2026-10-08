"""Atlas-style visuals — dot mosaics and "splitter" movies of a granularity tree.

Two looks borrowed from the "lipizones" lipid atlas of the mouse brain (Fusar Bassini,
La Manno et al., bioRxiv 2025, doi:10.1101/2025.10.13.682018):

* **Dot mosaic** (:func:`dot_mosaic`) — every pixel is drawn as its own round dot on a pure
  black background instead of as a square cell. The dots are sized in data units (a fraction
  of the pixel pitch) so they scale with the figure, and the gaps between them hide the blocky
  acquisition grid: a 50 µm raster reads as a field of points rather than a mosaic of tiles.
  It paints either an ion image (colour-mapped intensity) or a label image (one colour per
  segment, e.g. the tree colours of :func:`smile_msi.spatial.segment_colors`).
* **Splitter movie** (:func:`splitter_frames`, :func:`splitter_movie`) — walk a
  :class:`~smile_msi.spatial.Hierarchy` from coarse to fine (``k = 2, 3, 4, 6, …``) and fade
  every pixel from its colour at one level to its colour at the next, so territories visibly
  divide into their sub-territories. Because the tree colours are hierarchical (a segment keeps
  its hue family as ``k`` grows), a split reads as one colour blossoming into related shades.
  The fade is done in **OKLab** (Ottosson 2020), a perceptually uniform space, so intermediate
  frames pass through sensible colours instead of the muddy midpoints an sRGB blend gives.

Pure and headless: NumPy, SciPy and Matplotlib's object-oriented API only (``Figure`` + Agg
canvas, no ``pyplot`` global state, no Qt), so it is safe to call from worker threads and
scripts. GIF output goes through Pillow (a Matplotlib dependency); MP4 needs the optional
``imageio-ffmpeg`` package.
"""
from __future__ import annotations

import os

import numpy as np

from . import palettes, spatial

__all__ = ["dot_mosaic", "splitter_frames", "splitter_movie", "default_k_levels"]

#: Frames held on each level's colours before/after a transition.
DEFAULT_HOLD = 3
#: Integer upscale factor of plain-pixel movie frames (so the GIF is not tiny).
PIXEL_SCALE = 6
#: Edge length (px) of one data pixel when a movie is drawn as dots.
DOT_PX = 8


# --------------------------------------------------------------------------- #
# colour helpers (hex / sRGB / OKLab)
# --------------------------------------------------------------------------- #
def _hex_rgb(hexc: str) -> np.ndarray:
    """``#rrggbb`` → float RGB in 0..1."""
    h = str(hexc).lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (``(..., 3)``, 0..1) → OKLab (``(..., 3)``)."""
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def _oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (``(..., 3)``) → sRGB (``(..., 3)``, 0..1, clipped to gamut)."""
    lab = np.asarray(lab, dtype=float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.stack([
        +4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    ], axis=-1)
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def _luminance(color) -> float:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# --------------------------------------------------------------------------- #
# dot mosaic
# --------------------------------------------------------------------------- #
def _nice_length(target: float) -> float:
    """Round ``target`` down to a 1/2/5 × 10ⁿ length."""
    if not np.isfinite(target) or target <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(target))
    for m in (5.0, 2.0, 1.0):
        if m * mag <= target:
            return float(m * mag)
    return float(mag)


def _format_um(um: float) -> str:
    if um >= 1000 and abs(um / 1000 - round(um / 1000)) < 1e-9:
        return f"{int(round(um / 1000))} mm"
    return f"{um:g} µm"


def _is_label_image(image: np.ndarray, colors) -> bool:
    return colors is not None or np.issubdtype(image.dtype, np.integer)


def _dot_colors(image: np.ndarray, colors, cmap, vmin, vmax):
    """Per-pixel RGBA for the finite pixels of ``image`` → ``(rows, cols, rgba)``."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize, to_rgba_array

    img = np.asarray(image)
    if _is_label_image(img, colors):
        lab = np.full(img.shape, -1, dtype=np.int64)
        fin = np.isfinite(img.astype(float))
        lab[fin] = np.rint(img[fin].astype(float)).astype(np.int64)
        keep = lab >= 0
        pal = list(colors) if colors is not None else list(palettes.CATEGORY)
        if not pal:
            raise ValueError("colors must hold at least one colour")
        rows, cols = np.nonzero(keep)
        idx = lab[rows, cols]
        if colors is not None and idx.size and idx.max() >= len(pal):
            raise ValueError(f"label {int(idx.max())} has no colour (got {len(pal)} colours)")
        table = to_rgba_array(pal) if colors is not None else to_rgba_array(
            [pal[i % len(pal)] for i in range(int(idx.max()) + 1 if idx.size else 1)])
        return rows, cols, table[idx]
    img = img.astype(float)
    fin = np.isfinite(img)
    rows, cols = np.nonzero(fin)
    vals = img[rows, cols]
    lo = float(np.min(vals)) if vmin is None and vals.size else vmin
    hi = float(np.max(vals)) if vmax is None and vals.size else vmax
    if lo is None:
        lo, hi = 0.0, 1.0
    if hi is None:
        hi = lo + 1.0
    if hi <= lo:
        hi = lo + 1.0
    cm = colormaps[cmap] if isinstance(cmap, str) else cmap
    return rows, cols, cm(Normalize(lo, hi, clip=True)(vals))


def dot_mosaic(image, path=None, *, colors=None, cmap="viridis", vmin=None, vmax=None,
               dot=0.85, background="black", pixel_size_um=None, scale_bar_um=None,
               title="", dpi=300, inset_outline=False):
    """Draw ``image`` as a mosaic of round dots on a dark background (the lipizones look).

    ``image`` is a 2-D array. A **float** image is an ion image coloured by ``cmap`` over
    ``[vmin, vmax]`` (default: the data range). An **integer** label image — or any image
    when ``colors`` is given — is painted with ``colors[label]`` (a list of hex colours, e.g.
    :func:`smile_msi.spatial.segment_colors`). NaN pixels (and negative labels) get no dot, so
    off-tissue stays background.

    Each dot has diameter ``dot`` × the pixel pitch, sized in data units so the dots scale
    with the figure. Axes are equal-aspect and hidden. ``pixel_size_um`` adds a scale bar
    (length ``scale_bar_um``, default a round ~20 % of the width); ``inset_outline`` adds a
    small white outline of the tissue footprint in the top-right corner. ``path`` (PNG / PDF /
    SVG by extension) is optional; the :class:`~matplotlib.figure.Figure` is returned either
    way."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import EllipseCollection
    from matplotlib.figure import Figure

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {img.shape}")
    if not 0.0 < float(dot) <= 1.5:
        raise ValueError("dot must be in (0, 1.5] (fraction of the pixel pitch)")
    h, w = img.shape
    rows, cols, rgba = _dot_colors(img, colors, cmap, vmin, vmax)
    fg = "white" if _luminance(background) < 0.5 else "black"

    pad = 0.0
    if pixel_size_um:
        pad = max(2.0, 0.07 * h)            # strip below the tissue for the scale bar
    top = 0.0
    if title or inset_outline:
        top = 0.12 if inset_outline else 0.07   # fraction of figure height

    base_w = 8.0
    ax_h = base_w * (h + pad) / max(w, 1)
    fig_h = ax_h / (1.0 - top)
    fig = Figure(figsize=(base_w, fig_h), dpi=dpi, facecolor=background)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0 - top])
    ax.set_facecolor(background)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5 + pad, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    xy = np.column_stack([cols, rows]).astype(float)
    n = len(xy)
    coll = EllipseCollection(
        np.full(n, float(dot)), np.full(n, float(dot)), np.zeros(n),
        units="xy", offsets=xy, offset_transform=ax.transData,
        facecolors=rgba, edgecolors="none", linewidths=0)
    coll.set_rasterized(True)
    ax.add_collection(coll)

    if pixel_size_um:
        length_um = float(scale_bar_um) if scale_bar_um else _nice_length(0.2 * w * pixel_size_um)
        length_px = length_um / float(pixel_size_um)
        x0 = 0.03 * w
        y0 = h - 0.5 + pad * 0.45
        ax.plot([x0, x0 + length_px], [y0, y0], color=fg, lw=2.5, solid_capstyle="butt")
        ax.text(x0 + length_px / 2, y0 + pad * 0.12, _format_um(length_um), color=fg,
                ha="center", va="top", fontsize=8)

    if title:
        fig.text(0.02, 1.0 - top * 0.5, title, color=fg, ha="left", va="center", fontsize=11)

    if inset_outline:
        foot = np.zeros((h + 2, w + 2))
        foot[rows + 1, cols + 1] = 1.0
        side = 0.10 * fig_h                                # inches
        iw = side * w / max(h, 1)
        if iw > 0.3 * base_w:                              # very wide tissue: cap the width
            iw, side = 0.3 * base_w, 0.3 * base_w * h / max(w, 1)
        iax = fig.add_axes([1.0 - 0.02 - iw / base_w, 1.0 - 0.01 - side / fig_h,
                            iw / base_w, side / fig_h])
        iax.set_facecolor(background)
        if n:
            iax.contour(np.arange(-1, w + 1), np.arange(-1, h + 1), foot, levels=[0.5],
                        colors=[fg], linewidths=0.8)
        iax.set_xlim(-1, w)
        iax.set_ylim(h, -1)
        iax.set_aspect("equal", adjustable="box")
        iax.axis("off")

    if path is not None:
        fig.savefig(os.fspath(path), dpi=dpi, facecolor=background)
    return fig


# --------------------------------------------------------------------------- #
# splitter frames / movie
# --------------------------------------------------------------------------- #
def default_k_levels(hier, k_max: int = 24, n_levels: int = 6) -> list:
    """A geometric-ish ladder of cut sizes from 2 to ``min(hier.max_clusters, k_max)``."""
    top = int(max(1, min(int(hier.max_clusters), int(k_max))))
    if top < 2:
        return [1]
    ks = np.unique(np.rint(np.geomspace(2, top, num=max(2, int(n_levels)))).astype(int))
    return [int(k) for k in ks]


def _level_pixel_rgb(hier, ds, k: int):
    """Per-pixel sRGB (``(n_pixels, 3)``, black where unsegmented) and the bool mask of
    segmented pixels, for the tree cut at ``k``."""
    sub = spatial.cut(hier, k)
    mask = getattr(hier, "pixel_mask", None)
    n_total = int(getattr(hier, "n_total", 0) or ds.n_pixels)
    if mask is None:
        labels = np.asarray(sub, dtype=int)
    else:
        labels = np.full(n_total, -1, dtype=int)
        labels[np.asarray(mask, dtype=bool)] = sub
    pal = np.array([_hex_rgb(c) for c in spatial.segment_colors(hier, k)])
    valid = labels >= 0
    rgb = np.zeros((len(labels), 3))
    rgb[valid] = pal[labels[valid]]
    return rgb, valid


def _pixel_frames(hier, ds, k_levels, steps_per_level: int, hold: int):
    """Per-pixel RGB for every movie frame, plus the segmented-pixel mask.

    Frame layout: ``hold`` frames on level 0, then for each next level ``steps_per_level``
    OKLab blend frames (strictly between the two levels) followed by ``hold`` frames on that
    level — ``L*hold + (L-1)*steps_per_level`` frames for ``L`` levels."""
    ks = [int(k) for k in k_levels]
    if not ks:
        raise ValueError("k_levels must hold at least one level")
    steps = max(0, int(steps_per_level))
    hold = max(1, int(hold))
    levels = [_level_pixel_rgb(hier, ds, k) for k in ks]
    valid = np.zeros_like(levels[0][1])
    for _, v in levels:
        valid |= v
    rgbs = [r for r, _ in levels]
    labs = [_rgb_to_oklab(r) for r in rgbs]
    frames = [rgbs[0]] * hold
    for i in range(1, len(ks)):
        for j in range(steps):
            t = (j + 1) / (steps + 1)
            mix = (1.0 - t) * labs[i - 1] + t * labs[i]
            out = _oklab_to_rgb(mix)
            out[~valid] = 0.0
            frames.append(out)
        frames.extend([rgbs[i]] * hold)
    return frames, valid


def _pixel_index(ds):
    """``(H, W)`` int map of each grid cell to its dataset pixel index (-1 = empty)."""
    n = int(ds.n_pixels)
    return np.rint(ds.to_image(np.arange(n, dtype=float), fill=-1.0)).astype(int)


def _scatter_rgb(idx: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(idx.shape + (3,))
    m = idx >= 0
    out[m] = rgb[idx[m]]
    return out


def splitter_frames(hier, ds, k_levels, *, steps_per_level: int = 8, hold: int = DEFAULT_HOLD):
    """RGB frames (``(H, W, 3)`` float in 0..1, NaN-free, off-tissue black) of the tree
    splitting level by level through ``k_levels`` (e.g. ``[2, 3, 4, 6, 8, 12]``).

    Each pixel fades, in OKLab, from its :func:`~smile_msi.spatial.segment_colors` colour at
    one level to its colour at the next. The movie holds ``hold`` frames exactly on each
    level's colours and inserts ``steps_per_level`` blend frames between consecutive levels,
    so there are ``L*hold + (L-1)*steps_per_level`` frames for ``L`` levels."""
    frames, _ = _pixel_frames(hier, ds, k_levels, steps_per_level, hold)
    idx = _pixel_index(ds)
    cache = {}
    out = []
    for f in frames:                       # hold frames repeat one array: scatter it once
        if id(f) not in cache:
            cache[id(f)] = _scatter_rgb(idx, f)
        out.append(cache[id(f)].copy())
    return out


class _DotRenderer:
    """Re-colours one fixed set of dots per frame (fast: one figure, one collection)."""

    def __init__(self, idx: np.ndarray, valid: np.ndarray, dot: float, px: int = DOT_PX):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import EllipseCollection
        from matplotlib.figure import Figure

        h, w = idx.shape
        rows, cols = np.nonzero((idx >= 0) & valid[np.maximum(idx, 0)])
        self.src = idx[rows, cols]
        dpi = 100
        fig = Figure(figsize=(w * px / dpi, h * px / dpi), dpi=dpi, facecolor="black")
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("black")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis("off")
        n = len(rows)
        self.coll = EllipseCollection(
            np.full(n, float(dot)), np.full(n, float(dot)), np.zeros(n), units="xy",
            offsets=np.column_stack([cols, rows]).astype(float),
            offset_transform=ax.transData, facecolors="white", edgecolors="none", linewidths=0)
        ax.add_collection(self.coll)

    def render(self, rgb: np.ndarray) -> np.ndarray:
        self.coll.set_facecolors(np.clip(rgb[self.src], 0.0, 1.0))
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def splitter_movie(hier, ds, path, k_levels=None, *, fps: int = 12, steps_per_level: int = 8,
                   dot=None) -> str:
    """Write the splitter animation to ``path`` (``.gif`` via Pillow; ``.mp4`` needs the
    optional ``imageio-ffmpeg`` and raises a clear error without it). Returns ``path``.

    ``k_levels`` defaults to a geometric ladder from 2 to ``min(hier.max_clusters, 24)``.
    With ``dot`` (a fraction of the pixel pitch, e.g. ``0.85``) every frame is drawn as a
    dot mosaic on black; otherwise as plain pixels upscaled ``6×`` by nearest neighbour.
    In a GIF, runs of identical frames (the holds on each level) are stored once with a
    longer display time, so a GIF has ``L + (L-1)*steps_per_level`` frames for ``L``
    levels; an MP4 keeps every frame."""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".gif", ".mp4"):
        raise ValueError(f"unsupported movie format {ext!r}: use .gif or .mp4")
    if ext == ".mp4":
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "MP4 output needs the optional 'imageio-ffmpeg' package "
                "(pip install imageio-ffmpeg); write a .gif instead.") from exc
    if k_levels is None:
        k_levels = default_k_levels(hier)
    fps = max(1, int(fps))

    frames, valid = _pixel_frames(hier, ds, k_levels, steps_per_level, DEFAULT_HOLD)
    idx = _pixel_index(ds)
    renderer = _DotRenderer(idx, valid, float(dot)) if dot else None

    cache = {}

    def image_of(f):
        key = id(f)
        if key not in cache:
            if renderer is not None:
                cache[key] = renderer.render(f)
            else:
                img = _scatter_rgb(idx, f)
                img = np.repeat(np.repeat(img, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)
                cache[key] = np.rint(img * 255).astype(np.uint8)
        return cache[key]

    if ext == ".mp4":
        first = image_of(frames[0])
        hh, ww = first.shape[:2]
        size = (ww + ww % 2, hh + hh % 2)              # yuv420 needs even dimensions
        writer = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264",
                                             pix_fmt_in="rgb24", macro_block_size=1)
        writer.send(None)
        try:
            for f in frames:
                im = image_of(f)
                if im.shape[:2] != (size[1], size[0]):
                    im = np.pad(im, ((0, size[1] - hh), (0, size[0] - ww), (0, 0)), mode="edge")
                writer.send(np.ascontiguousarray(im).tobytes())
        finally:
            writer.close()
        return path

    from PIL import Image

    imgs, durations = [], []
    frame_ms = 1000.0 / fps
    prev = None
    for f in frames:
        if prev is not None and f is prev:             # a hold frame: lengthen the last one
            durations[-1] += frame_ms
            continue
        prev = f
        imgs.append(Image.fromarray(image_of(f)).convert("P", palette=Image.Palette.ADAPTIVE,
                                                         colors=256))
        durations.append(frame_ms)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[int(round(d)) for d in durations],
                 loop=0, disposal=1)
    return path
