"""Single-cell / subcellular spatial metabolomics engine (pure, no GUI).

This module turns an MSI acquisition plus a co-registered microscopy / histology
image into a **per-cell metabolite matrix**:

1. :func:`segment_cells` — segment cells (or nuclei) on the optical image. The
   **default** backend is a marker-controlled, distance-transform watershed built
   from ``scipy.ndimage`` primitives only (no extra dependency, fully offline).
   ``cellpose`` / ``stardist`` are **optional** deep backends, lazy-imported, and
   raise a clear error naming the ``uv sync --extra ...`` command when absent —
   never a silent fallback.
2. :func:`map_pixels_to_cells` — map every MSI pixel onto the cells it overlaps,
   through a registration transform (optical-px -> MSI base-px affine, exactly the
   transform produced by :mod:`smile_msi.registration`). When an MSI pixel overlaps
   several cells, its signal is split by **intersection area** (the SpaceM
   area-weighting). The whole computation works in the **un-rotated base tissue
   frame** (``coordinates - (x0, y0)``) and never reads ``ds.orientation``, so the
   mapping is rotation-invariant by construction.
3. :func:`build_cell_matrix` — pool the per-pixel feature rows into an
   area-weighted ``(n_cells x n_peaks)`` matrix, with optional per-pixel and
   per-cell normalization and a ``min_support`` cell filter.
4. :func:`cell_segments` — a thin composition that embeds + clusters the per-cell
   matrix by reusing :func:`smile_msi.multivariate._reduce_2d` /
   :func:`smile_msi.multivariate.cluster_points` — one point per cell.

``peaks`` everywhere is an **array/list of m/z floats** (matching the rest of the
engine: ``MSIDataset.feature_peaks`` is an ndarray, ``features_for_rows`` takes
floats), not peak dicts.

Literature
----------
- Rappez, L., Stadler, M., Triana, S. et al. (2021). SpaceM reveals metabolic
  states of single cells. *Nature Methods*, 18, 799-805.
  doi:10.1038/s41592-021-01316-y  (area-overlap apportioning of pixel signal
  across the cells it covers; reimplemented clean-room from the publication, the
  GPL-3 SpaceM source is **not** copied).
- (IMC-guided super-resolution single-cell metabolomics) *Nature Methods* (2024).
  doi:10.1038/s41592-024-02392-6  (motivation for the cell-level analysis tier).
- (Fluorescence as ground truth for single-cell quantitation) (2022).
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9730270/  (microscopy masks as the
  quantitation reference + the per-cell normalization step).
- Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. (2021). Cellpose: a
  generalist algorithm for cellular segmentation. *Nature Methods*, 18, 100-106.
  doi:10.1038/s41592-020-01018-x  (optional deep backend; BSD-3, called not
  vendored).
- Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. (2018). Cell detection with
  star-convex polygons. *MICCAI 2018*, LNCS 11071, 265-273.
  doi:10.1007/978-3-030-00934-9_30  (optional star-convex nucleus backend; BSD-3).
- Beucher, S. & Lantuejoul, C. (1979). Use of watersheds in contour detection.
  *Int. Workshop on Image Processing*.  (the default distance-transform
  marker-controlled watershed, built from SciPy primitives we already depend on).
- McInnes, L., Healy, J. & Melville, J. (2018). UMAP. arXiv:1802.03426.
  doi:10.48550/arXiv.1802.03426  (cell embedding, reused via ``_reduce_2d``).
- Lloyd, S.P. (1982). Least squares quantization in PCM. *IEEE Trans. Inf.
  Theory*, 28(2), 129-137. doi:10.1109/TIT.1982.1056489  (k-means over cells).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_TOL_PPM

SEGMENT_BACKENDS = ("watershed", "cellpose", "stardist")
WEIGHTINGS = ("area", "centroid")

_CELLPOSE_HINT = (
    "Cellpose backend needs the optional extra:\n"
    "    uv sync --extra cellpose"
)
_STARDIST_HINT = (
    "StarDist backend needs the optional extra:\n"
    "    uv sync --extra stardist"
)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class CellSet:
    """Segmented cells on the OPTICAL pixel grid.

    Attributes
    ----------
    label_image:
        ``(Hopt, Wopt)`` int32 label image; 0 = background, ``k`` = cell ``k``.
    n_cells:
        Number of distinct positive labels.
    centroids_opt:
        ``(n_cells, 2)`` cell centroids ``(row, col)`` in optical pixels.
    areas_opt:
        ``(n_cells,)`` cell area in optical pixels.
    labels:
        ``(n_cells,)`` the positive label values, ascending (so row ``i`` of
        ``centroids_opt`` / ``areas_opt`` corresponds to ``labels[i]``).
    backend:
        ``'watershed' | 'cellpose' | 'stardist'``.
    params:
        Segmentation knobs actually used (for provenance).
    """

    label_image: np.ndarray
    n_cells: int
    centroids_opt: np.ndarray
    areas_opt: np.ndarray
    labels: np.ndarray
    backend: str
    params: dict = field(default_factory=dict)


@dataclass
class PixelCellMap:
    """Sparse MSI-pixel <-> cell assignment with area weights.

    Attributes
    ----------
    n_cells:
        Number of cells (== ``len(cell_pixels)``).
    cell_ids:
        ``(n_cells,)`` the originating :class:`CellSet` label ids (column / row
        ``i`` here corresponds to ``cell_ids[i]``).
    cell_pixels:
        ``cell -> int`` pixel-row indices (into ``ds.coordinates`` order) that
        overlap the cell.
    cell_weights:
        ``cell -> float`` overlap-area weights, **sum-normalized to 1 per cell**
        (empty cells get an empty array).
    cell_support:
        ``(n_cells,)`` total **raw** overlap area behind each cell, in MSI-pixel
        units (a fully-covered single-pixel cell has support ~1.0). This is the
        un-normalized total the ``cell_weights`` were divided by, and what
        ``build_cell_matrix(min_support=...)`` filters on.
    cells_per_pixel:
        ``(n_pixels,)`` diagnostic: how many cells touch each MSI pixel (the
        resolution-mismatch readout; MSI-px >> cell -> many cells share a pixel).
    transform:
        The 3x3 optical-px -> MSI-base-px affine used (provenance).
    weighting:
        ``'area'`` or ``'centroid'``.
    """

    n_cells: int
    cell_ids: np.ndarray
    cell_pixels: list
    cell_weights: list
    cells_per_pixel: np.ndarray
    transform: np.ndarray
    weighting: str = "area"
    cell_support: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centroids_opt: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


@dataclass
class CellMatrix:
    """Per-cell area-weighted, normalized metabolite matrix.

    ``matrix`` is exactly the ``(rows x features)`` shape that
    :func:`smile_msi.multivariate._reduce_2d` / ``cluster_points`` expect.
    """

    matrix: np.ndarray
    cell_ids: np.ndarray
    peaks: np.ndarray
    centroids_opt: np.ndarray
    support: np.ndarray
    pixel_norm: str = "tic"
    cell_norm: str = "none"


# --------------------------------------------------------------------------- #
# 1. Cell segmentation
# --------------------------------------------------------------------------- #
def _stain_channel(optical_rgb: np.ndarray, channel: str) -> np.ndarray:
    """Pick a single 2-D float intensity channel from an RGB(A) image."""
    img = np.asarray(optical_rgb)
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim != 3:
        raise ValueError(f"optical_rgb must be 2-D or 3-D; got shape {img.shape}.")
    ch = (channel or "auto").lower()
    rgb = img[..., :3].astype(np.float64)
    if ch in ("gray", "grey", "auto"):
        # Luminance (Rec. 601). 'auto' currently == grayscale; a nuclear-stain
        # heuristic can be slotted in later without changing callers.
        return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    idx = {"r": 0, "g": 1, "b": 2}.get(ch)
    if idx is None:
        raise ValueError(
            f"channel must be 'gray'|'r'|'g'|'b'|'auto'; got {channel!r}."
        )
    return rgb[..., idx]


def _otsu_threshold(gray: np.ndarray, nbins: int = 256) -> float:
    """Pure-numpy Otsu (1979) threshold. Used when scikit-image is absent so the
    default path needs no extra dependency."""
    g = np.asarray(gray, dtype=np.float64).ravel()
    lo, hi = float(g.min()), float(g.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(g, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    # cumulative means
    cum = np.cumsum(hist * centers)
    grand = cum[-1]
    mean0 = np.where(w0 > 0, cum / np.where(w0 > 0, w0, 1), 0.0)
    mean1 = np.where(w1 > 0, (grand - cum) / np.where(w1 > 0, w1, 1), 0.0)
    between = w0 * w1 * (mean0 - mean1) ** 2
    k = int(np.argmax(between))
    return float(centers[k])


def _watershed_segment(gray: np.ndarray, *, invert: bool, min_diam_px: float,
                       random_state: int) -> np.ndarray:
    """Distance-transform marker-controlled watershed (Beucher & Lantuejoul 1979),
    SciPy primitives only. Returns an int32 label image (0 = background)."""
    from scipy import ndimage as ndi

    g = np.asarray(gray, dtype=np.float64)
    if invert:
        g = g.max() - g
    # smooth a touch before thresholding so speckle doesn't fragment cells
    sigma = max(0.5, min_diam_px / 6.0)
    sm = ndi.gaussian_filter(g, sigma=sigma)

    try:
        from skimage.filters import threshold_otsu  # optional, only for quality
        thr = float(threshold_otsu(sm))
    except Exception:  # noqa: BLE001 - pure-numpy fallback keeps the core dep-free
        thr = _otsu_threshold(sm)
    fg = sm > thr
    if not fg.any():
        return np.zeros(g.shape, dtype=np.int32)

    # Distance transform of the foreground; peaks = cell centres.
    dist = ndi.distance_transform_edt(fg)

    # Local maxima of the distance map, separated by >= min_diam_px, are markers.
    footprint = max(1, int(round(min_diam_px)))
    if footprint % 2 == 0:
        footprint += 1
    mx = ndi.maximum_filter(dist, size=footprint)
    peaks = (dist == mx) & fg & (dist > 0)
    markers, n_markers = ndi.label(peaks)
    if n_markers == 0:
        # No interior maxima (e.g. thin structures) -> fall back to connected
        # components of the foreground so we still emit one label per blob.
        markers, n_markers = ndi.label(fg)
        if n_markers == 0:
            return np.zeros(g.shape, dtype=np.int32)

    labels = _watershed_on_distance(dist, markers, fg)
    return labels.astype(np.int32)


def _watershed_on_distance(dist: np.ndarray, markers: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    """Watershed the negative distance map from ``markers`` inside ``mask``.

    Prefers ``skimage.segmentation.watershed`` when scikit-image is present (higher
    quality), otherwise uses ``scipy.ndimage.watershed_ift`` on a uint8-quantised
    surface so the default path stays dependency-free."""
    try:
        from skimage.segmentation import watershed as sk_watershed
        return sk_watershed(-dist, markers, mask=mask)
    except Exception:  # noqa: BLE001 - SciPy fallback (no scikit-image)
        from scipy import ndimage as ndi

        # Dependency-free marker-controlled split: assign every foreground pixel to
        # the *nearest marker* (a marker-seeded Voronoi within the mask). This is the
        # watershed-on-a-flat-surface limit and robustly cuts touching convex blobs
        # at the midline between their distance-transform maxima (Beucher &
        # Lantuejoul 1979). distance_transform_edt(return_indices) over the marker
        # complement gives, for each pixel, the coordinate of the nearest marker.
        mk = np.asarray(markers, dtype=np.int32)
        non_marker = mk == 0
        # indices of the nearest zero->marker source pixel
        _, (iy, ix) = ndi.distance_transform_edt(non_marker, return_indices=True)
        nearest = mk[iy, ix]
        out = np.where(np.asarray(mask, dtype=bool), nearest, 0)
        return out.astype(np.int32)


def _label_props(label_image: np.ndarray):
    """Return ``(labels, centroids_rc, areas)`` for the positive labels."""
    lab = np.asarray(label_image)
    flat = lab.ravel()
    pos = flat > 0
    if not pos.any():
        return (np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros(0))
    rows, cols = np.divmod(np.flatnonzero(pos), lab.shape[1])
    vals = flat[pos]
    uniq = np.unique(vals)
    # map label value -> compact index
    remap = {int(v): i for i, v in enumerate(uniq)}
    idx = np.array([remap[int(v)] for v in vals], dtype=int)
    areas = np.bincount(idx, minlength=uniq.size).astype(float)
    sum_r = np.bincount(idx, weights=rows.astype(float), minlength=uniq.size)
    sum_c = np.bincount(idx, weights=cols.astype(float), minlength=uniq.size)
    centroids = np.column_stack([sum_r / areas, sum_c / areas])
    return uniq.astype(int), centroids, areas


def _relabel_min_area(label_image: np.ndarray, min_area: float) -> np.ndarray:
    """Drop labels with area < ``min_area`` and compact the remaining labels to
    ``1..K`` (background stays 0)."""
    lab = np.asarray(label_image, dtype=np.int32)
    uniq, _, areas = _label_props(lab)
    keep = uniq[areas >= max(1.0, float(min_area))]
    out = np.zeros_like(lab)
    for new_id, old in enumerate(keep, start=1):
        out[lab == old] = new_id
    return out


def segment_cells(
    optical_rgb: np.ndarray,
    *,
    backend: str = "watershed",
    channel: str = "auto",
    min_diameter_um: float = 6.0,
    optical_pixel_size_um: float | None = None,
    invert: bool = False,
    random_state: int = 0,
    cellpose_model: str = "cyto3",
) -> CellSet:
    """Segment cells / nuclei on a microscopy / histology image.

    The **default** ``backend='watershed'`` is a marker-controlled, distance-transform
    watershed (Beucher & Lantuejoul 1979) built from ``scipy.ndimage`` primitives — no
    extra dependency, fully offline. ``'cellpose'`` (Stringer et al. 2021) and
    ``'stardist'`` (Schmidt et al. 2018) are optional deep backends, lazy-imported,
    each raising a clear error naming its ``uv sync --extra ...`` command if missing.

    Parameters
    ----------
    optical_rgb:
        ``(H, W)`` or ``(H, W, 3|4)`` microscopy / histology image (uint8 or float).
    backend:
        ``'watershed'`` (default) | ``'cellpose'`` | ``'stardist'``.
    channel:
        ``'gray'|'r'|'g'|'b'|'auto'`` nuclear / stain channel for the watershed
        default (``'auto'`` == grayscale luminance).
    min_diameter_um:
        Minimum cell diameter; markers are separated by at least this and labels
        smaller than the implied disc area are dropped. Converted to optical pixels
        via ``optical_pixel_size_um``; **when that is ``None`` it is interpreted as
        raw optical pixels** (recorded in ``params['diameter_in_pixels']``).
    optical_pixel_size_um:
        Optical pixel size (um/px). ``None`` -> ``min_diameter_um`` is in pixels.
    invert:
        ``True`` for dark-on-bright stain (e.g. H&E nuclei) so the foreground is the
        dark structures.
    random_state:
        Seeds the stochastic deep backends (watershed is deterministic).
    cellpose_model:
        Cellpose model name; ignored by watershed / stardist.

    Returns
    -------
    CellSet

    Raises
    ------
    ValueError
        For an unknown backend / channel.
    RuntimeError
        If an optional backend is requested but its package is not installed.

    References
    ----------
    Beucher & Lantuejoul (1979); Stringer et al. (2021) doi:10.1038/s41592-020-01018-x;
    Schmidt et al. (2018) doi:10.1007/978-3-030-00934-9_30.
    """
    backend = (backend or "watershed").lower()
    if backend not in SEGMENT_BACKENDS:
        raise ValueError(
            f"backend must be one of {SEGMENT_BACKENDS}; got {backend!r}."
        )

    diam_in_px = optical_pixel_size_um is None
    if diam_in_px:
        min_diam_px = float(min_diameter_um)
    else:
        min_diam_px = float(min_diameter_um) / float(optical_pixel_size_um)
    min_diam_px = max(1.0, min_diam_px)
    # disc area for the min-area filter
    min_area = np.pi * (min_diam_px / 2.0) ** 2

    params = {
        "backend": backend,
        "channel": channel,
        "min_diameter_um": float(min_diameter_um),
        "min_diameter_px": float(min_diam_px),
        "optical_pixel_size_um": optical_pixel_size_um,
        "diameter_in_pixels": bool(diam_in_px),
        "invert": bool(invert),
        "random_state": int(random_state),
    }

    if backend == "watershed":
        gray = _stain_channel(optical_rgb, channel)
        label_image = _watershed_segment(
            gray, invert=invert, min_diam_px=min_diam_px, random_state=random_state)
        label_image = _relabel_min_area(label_image, min_area)
    elif backend == "cellpose":
        label_image = _segment_cellpose(
            optical_rgb, channel=channel, min_diam_px=min_diam_px,
            invert=invert, random_state=random_state, model_name=cellpose_model)
        label_image = _relabel_min_area(label_image, min_area)
        params["cellpose_model"] = cellpose_model
    else:  # stardist
        label_image = _segment_stardist(
            optical_rgb, channel=channel, invert=invert, random_state=random_state)
        label_image = _relabel_min_area(label_image, min_area)

    uniq, centroids, areas = _label_props(label_image)
    return CellSet(
        label_image=label_image.astype(np.int32),
        n_cells=int(uniq.size),
        centroids_opt=centroids,
        areas_opt=areas,
        labels=uniq,
        backend=backend,
        params=params,
    )


def _segment_cellpose(optical_rgb, *, channel, min_diam_px, invert,
                      random_state, model_name):  # pragma: no cover - optional dep
    try:
        from cellpose import models
    except ImportError as e:
        raise RuntimeError(_CELLPOSE_HINT) from e
    gray = _stain_channel(optical_rgb, channel)
    if invert:
        gray = gray.max() - gray
    model = models.Cellpose(model_type=model_name, gpu=False)
    masks, *_ = model.eval(gray, diameter=float(min_diam_px), channels=[0, 0])
    return np.asarray(masks, dtype=np.int32)


def _segment_stardist(optical_rgb, *, channel, invert,
                      random_state):  # pragma: no cover - optional dep
    try:
        from stardist.models import StarDist2D
    except ImportError as e:
        raise RuntimeError(_STARDIST_HINT) from e
    gray = _stain_channel(optical_rgb, channel).astype(np.float32)
    if invert:
        gray = gray.max() - gray
    g = gray - gray.min()
    rng = g.max()
    if rng > 0:
        g = g / rng
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    labels, _ = model.predict_instances(g)
    return np.asarray(labels, dtype=np.int32)


# --------------------------------------------------------------------------- #
# 2. Pixel -> cell mapping (area-weighted)
# --------------------------------------------------------------------------- #
def _base_xy(ds) -> np.ndarray:
    """Per-pixel base-frame ``(x, y)`` (orientation-0), i.e. ``coordinates -
    (x0, y0)``. Never reads ``ds.orientation`` -> rotation-invariant."""
    coords = np.asarray(ds.coordinates, dtype=int)
    x0 = int(coords[:, 0].min())
    y0 = int(coords[:, 1].min())
    xs = coords[:, 0] - x0
    ys = coords[:, 1] - y0
    return np.column_stack([xs.astype(float), ys.astype(float)])


def _pixel_footprint_in_optical(transform: np.ndarray) -> float:
    """Optical-px length of one base-px edge, derived from the transform's scale.

    ``transform`` maps optical-px -> base-px; its linear part scales optical lengths
    to base lengths, so ``1 base-px`` is ``1 / scale`` optical-px."""
    M = np.asarray(transform, dtype=float)
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    # average of the two column norms = mean optical->base scale
    s = 0.5 * (np.hypot(a, c) + np.hypot(b, d))
    if s <= 0:
        return 1.0
    return 1.0 / s


def map_pixels_to_cells(
    ds,
    cells: CellSet,
    transform: np.ndarray,
    *,
    optical_pixel_size_um: float | None = None,
    weighting: str = "area",
) -> PixelCellMap:
    """Map MSI pixels onto cells through a registration transform, area-weighted.

    For every MSI pixel, its square footprint (one acquisition pixel wide) is pushed
    through ``inv(transform)`` into the optical frame and rasterised against
    ``cells.label_image``; the **intersection area** with each cell becomes that
    (pixel, cell) overlap weight (SpaceM area apportioning; Rappez et al. 2021). The
    weights are then normalized to sum to 1 per cell, so each cell's profile is an
    **area-weighted mean** of the pixel spectra it covers.

    All geometry is in the **un-rotated base frame** (``coordinates - (x0, y0)``);
    ``ds.orientation`` is never read, so the mapping is identical across display
    rotation.

    Parameters
    ----------
    ds:
        :class:`~smile_msi.msi.MSIDataset`.
    cells:
        :class:`CellSet` on the optical grid.
    transform:
        ``3x3`` affine mapping optical-px ``(col, row)`` -> MSI base-px ``(x, y)``
        (orientation-0), exactly :attr:`RegistrationResult.matrix`.
    optical_pixel_size_um:
        Optical pixel size (um/px). With both this and ``ds.pixel_size_um`` known,
        the MSI-pixel footprint spans ``ds.pixel_size_um / optical_pixel_size_um``
        optical-px; when **either is ``None`` the footprint falls back to the
        transform-derived scale** (one base-px edge in optical-px).
    weighting:
        ``'area'`` (default, SpaceM area overlap) or ``'centroid'`` (each cell is
        assigned to the single MSI pixel its centroid falls in, weight 1).

    Returns
    -------
    PixelCellMap

    Raises
    ------
    ValueError
        For an unknown ``weighting`` or a non-invertible / non-3x3 ``transform``.
    """
    weighting = (weighting or "area").lower()
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}; got {weighting!r}.")
    M = np.asarray(transform, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"transform must be a 3x3 affine; got shape {M.shape}.")
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError as e:
        raise ValueError("transform is singular and cannot be inverted.") from e

    lab = np.asarray(cells.label_image, dtype=np.int32)
    Hopt, Wopt = lab.shape
    n_cells = int(cells.n_cells)
    # label value -> compact cell index (0..n_cells-1)
    cell_ids = np.asarray(cells.labels, dtype=int)
    id_to_idx = {int(v): i for i, v in enumerate(cell_ids)}

    base_xy = _base_xy(ds)                 # (n_pixels, 2) (x, y)
    n_pixels = base_xy.shape[0]

    if weighting == "centroid":
        return _map_centroid(
            ds, cells, M, base_xy, id_to_idx, n_cells, cell_ids)

    # ---- area weighting -------------------------------------------------- #
    # MSI-pixel footprint side in optical px.
    if optical_pixel_size_um is not None and getattr(ds, "pixel_size_um", None):
        foot_opt = float(ds.pixel_size_um) / float(optical_pixel_size_um)
    else:
        foot_opt = _pixel_footprint_in_optical(M)
    foot_opt = max(1.0, float(foot_opt))

    # accumulate per (pixel, cell) overlap counts in a dict keyed by (pix, cell_idx)
    # using sampling of the footprint at optical resolution.
    overlap = {}                          # (pix_row, cell_idx) -> sampled area
    cells_per_pixel = np.zeros(n_pixels, dtype=int)

    # number of samples along each footprint edge (>=1); cap for very large footprints.
    n_side = int(min(64, max(1, round(foot_opt))))
    # sample offsets within a base-pixel, centred on the pixel: [-0.5, 0.5)
    offs = (np.arange(n_side) + 0.5) / n_side - 0.5     # in base-px units
    ox, oy = np.meshgrid(offs, offs)
    ox = ox.ravel()
    oy = oy.ravel()
    cell_area = float(n_side * n_side)     # samples per pixel footprint

    for p in range(n_pixels):
        bx, by = base_xy[p, 0], base_xy[p, 1]
        sx = bx + ox                       # base-frame sample x
        sy = by + oy
        pts = np.column_stack([sx, sy, np.ones(sx.size)])      # (S, 3)
        opt = pts @ Minv.T                 # base -> optical
        w = opt[:, 2:3]
        w = np.where(w == 0, 1.0, w)
        oc = opt[:, 0] / w[:, 0]           # optical col
        orr = opt[:, 1] / w[:, 0]          # optical row
        cc = np.floor(oc + 0.5).astype(int)
        rr = np.floor(orr + 0.5).astype(int)
        inside = (cc >= 0) & (cc < Wopt) & (rr >= 0) & (rr < Hopt)
        if not inside.any():
            continue
        vals = lab[rr[inside], cc[inside]]
        hit = vals[vals > 0]
        if hit.size == 0:
            continue
        ids, counts = np.unique(hit, return_counts=True)
        seen = 0
        for v, ct in zip(ids, counts):
            ci = id_to_idx.get(int(v))
            if ci is None:
                continue
            overlap[(p, ci)] = overlap.get((p, ci), 0.0) + ct / cell_area
            seen += 1
        cells_per_pixel[p] = seen

    # build CSR-like per-cell lists
    by_cell_pix = [[] for _ in range(n_cells)]
    by_cell_w = [[] for _ in range(n_cells)]
    for (p, ci), w in overlap.items():
        by_cell_pix[ci].append(p)
        by_cell_w[ci].append(w)

    cell_pixels = []
    cell_weights = []
    cell_support = np.zeros(n_cells, dtype=float)
    for ci in range(n_cells):
        if by_cell_pix[ci]:
            pix = np.asarray(by_cell_pix[ci], dtype=int)
            wt = np.asarray(by_cell_w[ci], dtype=float)
            order = np.argsort(pix)
            pix = pix[order]
            wt = wt[order]
            s = wt.sum()                          # raw overlap area, MSI-pixel units
            cell_support[ci] = s
            wt = wt / s if s > 0 else wt
            cell_pixels.append(pix)
            cell_weights.append(wt)
        else:
            cell_pixels.append(np.zeros(0, dtype=int))
            cell_weights.append(np.zeros(0, dtype=float))

    return PixelCellMap(
        n_cells=n_cells,
        cell_ids=cell_ids,
        cell_pixels=cell_pixels,
        cell_weights=cell_weights,
        cells_per_pixel=cells_per_pixel,
        transform=M,
        weighting="area",
        cell_support=cell_support,
        centroids_opt=np.asarray(cells.centroids_opt, dtype=float),
    )


def _map_centroid(ds, cells, M, base_xy, id_to_idx, n_cells, cell_ids):
    """Conservative mapping: each cell -> the single MSI pixel its centroid lands in."""
    # cell centroids (row, col) -> optical (col, row) -> base (x, y)
    cen = np.asarray(cells.centroids_opt, dtype=float)
    if cen.shape[0] == 0:
        return PixelCellMap(
            n_cells=0, cell_ids=cell_ids, cell_pixels=[], cell_weights=[],
            cells_per_pixel=np.zeros(base_xy.shape[0], dtype=int),
            transform=M, weighting="centroid",
            cell_support=np.zeros(0), centroids_opt=cen)
    opt_colrow = np.column_stack([cen[:, 1], cen[:, 0], np.ones(cen.shape[0])])
    base = opt_colrow @ M.T               # optical -> base
    w = base[:, 2:3]
    w = np.where(w == 0, 1.0, w)
    bx = base[:, 0] / w[:, 0]
    by = base[:, 1] / w[:, 0]
    cell_pixels = [np.zeros(0, dtype=int) for _ in range(n_cells)]
    cell_weights = [np.zeros(0, dtype=float) for _ in range(n_cells)]
    cells_per_pixel = np.zeros(base_xy.shape[0], dtype=int)
    # nearest MSI pixel to each cell centroid — one vectorized KD-tree query
    # instead of an O(n_cells x n_pixels) Python distance loop.
    from scipy.spatial import cKDTree
    nearest = cKDTree(base_xy[:, :2]).query(np.column_stack([bx, by]))[1]
    for ci in range(n_cells):
        p = int(nearest[ci])
        cell_pixels[ci] = np.array([p], dtype=int)
        cell_weights[ci] = np.array([1.0], dtype=float)
        cells_per_pixel[p] += 1
    return PixelCellMap(
        n_cells=n_cells, cell_ids=cell_ids, cell_pixels=cell_pixels,
        cell_weights=cell_weights, cells_per_pixel=cells_per_pixel,
        transform=M, weighting="centroid",
        cell_support=np.ones(n_cells, dtype=float), centroids_opt=cen)


# --------------------------------------------------------------------------- #
# 3. Per-cell metabolite matrix
# --------------------------------------------------------------------------- #
def _cell_norm_factor(row: np.ndarray, cell_norm: str) -> float:
    cn = (cell_norm or "none").lower()
    if cn in ("none", ""):
        return 1.0
    if cn == "tic":
        s = float(np.sum(row))
        return s if s > 0 else 1.0
    if cn == "median":
        pos = row[row > 0]
        m = float(np.median(pos)) if pos.size else 0.0
        return m if m > 0 else 1.0
    raise ValueError(f"cell_norm must be 'none'|'tic'|'median'; got {cell_norm!r}.")


def build_cell_matrix(
    ds,
    mapping: PixelCellMap,
    peaks=None,
    *,
    tol_ppm: float = DEFAULT_TOL_PPM,
    reduce: str = "sum",
    pixel_norm: str = "tic",
    cell_norm: str = "none",
    min_support: float = 0.25,
) -> CellMatrix:
    """Build the area-weighted ``(n_kept_cells x n_peaks)`` per-cell matrix.

    Pulls per-pixel feature rows via ``ds.features_for_rows(peaks, rows,
    tol_ppm=, reduce=, norm=pixel_norm)`` for the **union of pixels** referenced in
    ``mapping`` (one extraction; rows re-indexed locally), then forms each cell row
    as ``sum_i weight_i * pixelrow_i`` (an area-weighted mean since the weights sum
    to 1), applies ``cell_norm``, and drops cells whose total pixel-area support is
    below ``min_support`` (or that have no overlapping pixels).

    Parameters
    ----------
    ds:
        :class:`~smile_msi.msi.MSIDataset`.
    mapping:
        :class:`PixelCellMap` from :func:`map_pixels_to_cells`.
    peaks:
        Array / list of m/z floats. Defaults to ``ds.feature_peaks`` (an ndarray);
        if that is ``None`` the caller **must** pass an explicit m/z list.
    tol_ppm, reduce:
        Forwarded to ``features_for_rows`` (peak integration window / reducer).
    pixel_norm:
        Per-pixel normalization **before** pooling (``'none'|'tic'|'rms'|'median'``).
    cell_norm:
        Per-cell normalization **after** pooling (``'none'|'tic'|'median'``).
    min_support:
        Cells with total overlap-area support < this are dropped. For ``'area'``
        weighting support is the **raw (pre-normalization) overlap area in MSI
        pixels**, so a barely-touched cell (support << 1 pixel) is dropped.

    Returns
    -------
    CellMatrix

    Raises
    ------
    ValueError
        If ``peaks`` is ``None`` and ``ds.feature_peaks`` is ``None``.
    """
    if peaks is None:
        peaks = getattr(ds, "feature_peaks", None)
    if peaks is None:
        raise ValueError(
            "build_cell_matrix: no peaks given and ds.feature_peaks is None — "
            "call ds.build_features(peaks) first or pass an explicit m/z list.")
    peaks = np.asarray(peaks, dtype=float).ravel()

    support_arr = np.asarray(getattr(mapping, "cell_support", None)
                             if getattr(mapping, "cell_support", None) is not None
                             else np.zeros(0), dtype=float)
    has_support = support_arr.size == mapping.n_cells

    # union of referenced pixels -> one extraction
    union = set()
    for ci in range(mapping.n_cells):
        pix = mapping.cell_pixels[ci]
        if pix.size:
            union.update(int(p) for p in pix)
    rows = np.array(sorted(union), dtype=int)
    if rows.size == 0:
        return CellMatrix(
            matrix=np.zeros((0, peaks.size)), cell_ids=np.zeros(0, dtype=int),
            peaks=peaks, centroids_opt=np.zeros((0, 2)), support=np.zeros(0),
            pixel_norm=pixel_norm, cell_norm=cell_norm)

    feat = ds.features_for_rows(peaks, rows, tol_ppm=tol_ppm, reduce=reduce,
                                norm=pixel_norm)        # (len(rows) x peaks)
    local = {int(r): k for k, r in enumerate(rows)}

    cell_rows = []
    kept_ids = []
    kept_support = []
    cen = np.asarray(mapping.centroids_opt, dtype=float) \
        if getattr(mapping, "centroids_opt", None) is not None else None
    kept_cen = []

    for ci in range(mapping.n_cells):
        pix = mapping.cell_pixels[ci]
        wt = mapping.cell_weights[ci]
        if pix.size == 0:
            continue
        # raw support = un-normalized overlap area in MSI-pixel units (tracked by
        # map_pixels_to_cells); a barely-touched cell has support << 1 pixel.
        raw = float(support_arr[ci]) if has_support else float(pix.size)
        if raw < float(min_support):
            continue
        loc = np.array([local[int(p)] for p in pix], dtype=int)
        block = feat[loc]                                 # (k x peaks)
        row = (wt[:, None] * block).sum(axis=0)           # area-weighted mean
        nf = _cell_norm_factor(row, cell_norm)
        row = row / nf
        cell_rows.append(row)
        kept_ids.append(int(mapping.cell_ids[ci]))
        kept_support.append(raw)
        if cen is not None and cen.ndim == 2 and cen.shape[0] == mapping.n_cells:
            kept_cen.append(cen[ci])

    if not cell_rows:
        return CellMatrix(
            matrix=np.zeros((0, peaks.size)), cell_ids=np.zeros(0, dtype=int),
            peaks=peaks, centroids_opt=np.zeros((0, 2)), support=np.zeros(0),
            pixel_norm=pixel_norm, cell_norm=cell_norm)

    matrix = np.vstack(cell_rows)
    centroids = (np.vstack(kept_cen) if kept_cen
                 else np.zeros((len(cell_rows), 2)))
    return CellMatrix(
        matrix=matrix,
        cell_ids=np.asarray(kept_ids, dtype=int),
        peaks=peaks,
        centroids_opt=centroids,
        support=np.asarray(kept_support, dtype=float),
        pixel_norm=pixel_norm,
        cell_norm=cell_norm,
    )


# --------------------------------------------------------------------------- #
# 4. Cell-level embedding + clustering (reuse the existing engine)
# --------------------------------------------------------------------------- #
def cell_segments(
    cellmat: CellMatrix,
    *,
    method: str = "umap",
    k: int = 8,
    random_state: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    cluster_method: str = "kmeans",
):
    """Embed + cluster the per-cell matrix → ``(coords, labels, method_used)``.

    A thin composition: ``log1p`` + ``StandardScaler`` on ``cellmat.matrix``
    (mirroring ``multivariate.embedding()``'s preprocessing), then
    :func:`smile_msi.multivariate._reduce_2d` to a 2-D embedding and
    :func:`smile_msi.multivariate.cluster_points` for the labels — **one point per
    cell**. Deterministic for a fixed ``random_state``.

    Returns
    -------
    (coords, labels, method_used):
        ``coords`` ``(n_cells, 2)``; ``labels`` ``(n_cells,)`` int; ``method_used``
        the embedding method actually run (may differ from ``method`` if UMAP is
        unavailable — reported honestly, never silently swapped).

    References
    ----------
    UMAP — McInnes et al. (2018) arXiv:1802.03426; k-means — Lloyd (1982)
    doi:10.1109/TIT.1982.1056489.
    """
    from sklearn.preprocessing import StandardScaler

    from . import multivariate

    X = np.asarray(cellmat.matrix, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), "none"
    Xp = np.log1p(np.clip(X, 0, None))
    if Xp.shape[0] >= 2:
        Xp = StandardScaler().fit_transform(Xp)
    coords, used = multivariate._reduce_2d(
        Xp, method, random_state, metric=metric,
        n_neighbors=n_neighbors, min_dist=min_dist)
    labels = multivariate.cluster_points(
        coords, method=cluster_method, k=k, random_state=random_state)
    return coords, labels, used
