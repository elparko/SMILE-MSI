"""Ion-image display helpers — contrast handling and multi-channel overlays that
every MSI viewer offers.

* :func:`quantile_clip` — **hotspot removal**: clip a few saturating pixels so the
  dynamic range reflects the tissue, not one outlier (the standard MSI contrast
  control).
* :func:`rgb_overlay` — assign three ions to **R/G/B channels** to compare their
  distributions in one image.
* :func:`apply_colormap` — render a scalar ion image to an RGBA array with a named
  (matplotlib) colormap, for consistent PNG export.
* :func:`overlay_optical` — blend an ion image over a co-registered **optical /
  H&E image** (alpha overlay), as in MSiReader's image-overlay feature.
"""
from __future__ import annotations

import numpy as np

from .constants import DEFAULT_TOL_PPM


def quantile_clip(img: np.ndarray, low: float = 0.0, high: float = 99.0) -> np.ndarray:
    """Clip an image to the [low, high] percentile of its finite values (hotspot
    removal). ``high=99`` drops the top 1% of saturating pixels."""
    img = np.asarray(img, float)
    finite = img[np.isfinite(img)]
    if not finite.size:
        return img
    lo, hi = np.percentile(finite, (low, high))   # one pass for both cut points
    if hi <= lo:
        return img
    return np.clip(img, lo, hi)


def relative_max(img: np.ndarray, high: float = 99.0) -> float:
    """Peak intensity as a percentage of the hotspot-clip value.

    ``100%`` is anchored to the value at the ``high`` percentile — the same
    threshold :func:`quantile_clip` uses — so a genuine hotspot sitting above the
    clip reads **over 100%** (a pixel at 1.58x the clip value → ``158``). Returns
    ``0.0`` when the image is empty or the anchor is non-positive."""
    a = np.asarray(img, float)
    finite = a[np.isfinite(a)]
    if not finite.size:
        return 0.0
    anchor = float(np.percentile(finite, high))
    if anchor <= 0:
        return 0.0
    return 100.0 * float(finite.max()) / anchor


def relative_window(img: np.ndarray, lo: float, hi: float, clip: float = 99.0,
                    anchor: float | None = None) -> np.ndarray:
    """Normalize an image to [0,1] using a **relative-intensity** contrast window.

    ``100%`` is the value at the ``clip`` percentile (the hotspot-clip anchor that
    :func:`relative_max` also reports against), and ``lo``/``hi`` are percentages *of that
    anchor* — so ``hi=90`` saturates at 0.9x the anchor, **not** the 90th
    percentile of pixels. Off-tissue NaNs are preserved (``np.clip`` keeps NaN → transparent),
    and percentiles use the finite tissue values only so the empty background can't skew it.

    Pass an explicit ``anchor`` (an absolute intensity) to use it as the 100% value instead of
    the per-image ``clip`` percentile — this is how the Export Studio maps one shared intensity
    window across several sections so the same ion is honestly comparable between slides."""
    a = np.asarray(img, float)
    finite = a[np.isfinite(a)]
    if not finite.size:
        return a
    anchor = float(anchor) if (anchor is not None and anchor > 0) else float(np.percentile(finite, clip))
    if anchor <= 0:
        return np.clip(a, 0.0, 1.0)
    vlo, vhi = (lo / 100.0) * anchor, (hi / 100.0) * anchor
    if vhi <= vlo:
        return np.clip(a, 0.0, 1.0)
    return np.clip((a - vlo) / (vhi - vlo), 0.0, 1.0)


def _norm01(img, low, high):
    img = quantile_clip(img, low, high)
    finite = img[np.isfinite(img)]
    if not finite.size:
        return np.zeros_like(img)
    lo, hi = finite.min(), finite.max()
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)


def rgb_overlay(ds, mz_r, mz_g, mz_b, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                low: float = 0.0, high: float = 99.0) -> np.ndarray:
    """Three-ion RGB overlay (any channel may be ``None``). Returns an (H, W, 3)
    uint8 image; each channel is hotspot-clipped and scaled independently."""
    h, w = ds.height, ds.width
    rgb = np.zeros((h, w, 3), dtype=np.ubyte)
    for ch, mz in enumerate((mz_r, mz_g, mz_b)):
        if mz is None:
            continue
        img = ds.ion_image(float(mz), tol_ppm=tol_ppm, norm=norm)
        rgb[:, :, ch] = (_norm01(img, low, high) * 255).astype(np.ubyte)
    return rgb


def apply_colormap(img: np.ndarray, cmap: str = "viridis", low: float = 0.0,
                   high: float = 99.0, clip: float | None = None,
                   anchor: float | None = None) -> np.ndarray:
    """Scalar image -> (H, W, 4) uint8 via a matplotlib colormap (for export).

    When ``clip`` is given, ``low``/``high`` are treated as **relative intensity** (% of the
    value at the ``clip`` percentile, matching the on-screen viewer); otherwise
    they are pixel percentiles (legacy). An explicit ``anchor`` (absolute intensity) overrides
    the per-image clip percentile as the 100% value — used for shared cross-section contrast.
    The colormap name is matched leniently (exact, then lower-case, then a viridis fallback) so
    it mirrors the on-screen ``gui.common.colormap`` and never crashes export on a capitalized
    or unknown name."""
    from matplotlib import colormaps

    norm = relative_window(img, low, high, clip, anchor) if clip is not None else _norm01(img, low, high)
    name = cmap if cmap in colormaps else cmap.lower()
    if name not in colormaps:
        name = "viridis"
    rgba = colormaps[name](norm)
    return (rgba * 255).astype(np.ubyte)


def _draw_scale_bar(ax, h, w, pixel_size_um, scale_bar_um):
    """White scale bar in the bottom-right corner. ``h``/``w`` are the image height/width in
    pixels; ``ax`` is the matplotlib axes the image is drawn on. Length, rounding and label
    come from the same single source of truth as every other renderer
    (:mod:`smile_msi.annotations`): if the requested length is too long to draw at its true
    size it snaps to a fitting ``1-2-5`` round length and relabels, so the bar never overflows
    the frame and its caption never disagrees with the line it draws."""
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe
    from . import annotations as annot
    if not (pixel_size_um and scale_bar_um and w):
        return
    n_px = float(scale_bar_um) / float(pixel_size_um)
    if n_px > annot._SCALEBAR_MAX_FRAC * w:
        fitted = annot.nice_scalebar_um(float(w) * float(pixel_size_um))
        if not fitted:
            return
        scale_bar_um = fitted
        n_px = scale_bar_um / float(pixel_size_um)
    x0, y0 = w * 0.97 - n_px, h * 0.95
    # white bar + label with a thin dark halo so they stay legible on light colormaps too
    # (the footer-band renderers pick a contrast ink; this over-image bar keeps white).
    ax.add_patch(Rectangle((x0, y0), n_px, max(1, h * 0.012), facecolor="white",
                           edgecolor="black", linewidth=0.5))
    ax.text(x0 + n_px / 2, y0 - h * 0.02, annot.format_scalebar_label(scale_bar_um),
            color="white", ha="center", va="bottom", fontsize=9, weight="bold",
            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])


def export_ion_figure(image: np.ndarray, path: str, title: str = "", cmap: str = "viridis",
                      low: float = 0.0, high: float = 99.0, pixel_size_um: float | None = None,
                      scale_bar_um: float | None = None, dpi: int = 200, colorbar: bool = True,
                      window: tuple | None = None):
    """Save a publication-quality ion-image figure: hotspot-clipped, with title,
    colorbar, and an optional physical scale bar (needs ``pixel_size_um``). When
    ``window=(lo, hi)`` is given, the intensity contrast window is annotated on the
    colorbar so the displayed range is recorded in the saved image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = quantile_clip(image, low, high)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(img, cmap=cmap, interpolation="nearest")
    if title:
        ax.set_title(title)
    ax.axis("off")
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="intensity")
        if window is not None:
            lo, hi = window
            cbar.ax.set_ylabel(f"intensity   ·   window {lo:.0f}–{hi:.0f}%")
            cbar.ax.text(0.5, -0.04, f"{lo:.0f}%", transform=cbar.ax.transAxes,
                         ha="center", va="top", fontsize=8)
            cbar.ax.text(0.5, 1.04, f"{hi:.0f}%", transform=cbar.ax.transAxes,
                         ha="center", va="bottom", fontsize=8)
    if pixel_size_um and scale_bar_um:
        h, w = img.shape
        _draw_scale_bar(ax, h, w, pixel_size_um, scale_bar_um)
    # accept a path string or a file-like (e.g. BytesIO); the latter needs an explicit format
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                format=None if isinstance(path, str) else "png")
    plt.close(fig)
    return path


def export_overlay_figure(rgb: np.ndarray, path: str, entries, title: str = "",
                          pixel_size_um: float | None = None, scale_bar_um: float | None = None,
                          dpi: int = 200):
    """Save a multi-feature color-overlay (HxWx3 uint8) with a legend mapping each
    feature's color to its m/z + intensity window — so the composite is self-describing.
    ``entries`` is a list of ``(label, hex_color)`` pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.asarray(rgb, np.ubyte), interpolation="nearest")
    if title:
        ax.set_title(title)
    ax.axis("off")
    if entries:
        handles = [Patch(facecolor=c, edgecolor="none", label=lab) for lab, c in entries]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=8, frameon=False)
    if pixel_size_um and scale_bar_um:
        h, w = rgb.shape[:2]
        _draw_scale_bar(ax, h, w, pixel_size_um, scale_bar_um)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                format=None if isinstance(path, str) else "png")
    plt.close(fig)
    return path


def overlay_optical(ion_rgba: np.ndarray, optical: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend an ion image (RGBA uint8) over a co-registered optical image
    (RGB/RGBA uint8, resized to match). Returns (H, W, 3) uint8."""
    from PIL import Image

    h, w = ion_rgba.shape[:2]
    opt = Image.fromarray(np.asarray(optical, np.ubyte)).convert("RGB").resize((w, h))
    opt = np.asarray(opt, float)
    ion = ion_rgba[:, :, :3].astype(float)
    a = (ion_rgba[:, :, 3:4].astype(float) / 255.0) * alpha if ion_rgba.shape[2] == 4 else alpha
    out = opt * (1 - a) + ion * a
    return np.clip(out, 0, 255).astype(np.ubyte)
