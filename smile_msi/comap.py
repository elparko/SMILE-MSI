"""Spatial multi-omics co-mapping — import a *second* assay measured on the same
tissue (spatial transcriptomics / proteomics) and run joint analysis against the
MSI ion data.

SMILE MSI is otherwise single-modality: a slide is one
:class:`~smile_msi.msi.MSIDataset` (an ion cube on a pixel grid). This module adds
a modality-agnostic second grid (:class:`ModalityGrid`) plus the joint primitives
the GUI/cohort layer wires up:

* :func:`load_modality` — import a second assay from a generic CSV/TSV (numpy +
  pandas, core), a Space Ranger directory, or an AnnData ``.h5ad`` file (lazy
  ``anndata`` import; the optional ``comap`` extra). Returns a
  :class:`ModalityGrid` in *its own* (source) coordinate frame.
* :func:`resample_to_msi` — map the modality grid onto the MSI **pixel grid** via a
  registration transform (``transform=None`` = same grid / identity), nearest-pixel
  binning in the pre-rotation pixel frame so it composes with ``ds.orientation``.
* :func:`cross_modality_correlation` — for every ``(m/z peak, modality feature)``
  pair, the per-pixel spatial correlation of their co-registered images (Pearson or
  Spearman) over a region / detected-foreground mask.
* :func:`joint_embedding` — a single 2-D embedding from z-scored, block-weighted
  MSI ⊕ modality features (reuses :func:`smile_msi.multivariate.embedding`).
* :func:`cross_region_comparison` — region fingerprints concatenating the MSI mean
  profile with the resampled modality mean profile, scored region×region by the same
  normalized-profile kernel that backs :func:`smile_msi.spatial.region_correlation`
  (returns the existing :class:`~smile_msi.spatial.RegionCorrelation` type).

Heavy / optional backends (``anndata``, ``h5py``) are imported **lazily** inside
:func:`load_modality`, mirroring ``explain.py``'s ``shap`` import; a missing backend
raises a clear ``ImportError`` naming ``uv sync --extra comap``. The core path
(generic CSV import + all four analysis functions) needs only numpy / scipy /
pandas, which are already core dependencies.

The cross-modality link is **correlation-first** (one vectorized resample +
Pearson/Spearman, reusing the normalized-profile math behind
``spatial._profile_similarity``) plus a **concatenated-feature joint embedding** for
joint clustering. We deliberately do *not* reimplement a heavyweight integration
model (MOFA / Seurat anchors); those are network/R-bound and overkill for two
co-registered grids on a CPU-only machine.

Literature
----------
- Vicari, M., Mirzazadeh, R., Nilsson, A. et al. (2024). Spatial multimodal
  analysis of transcriptomes and metabolomes in tissues. *Nature Biotechnology*,
  42, 1046-1050. doi:10.1038/s41587-023-01937-y  (the "SMA" MALDI-MSI-onto-Visium
  workflow: serial/same-section MSI + ST aligned to a common grid, then
  metabolite-gene spatial correlation — our reference for the same-grid resampling
  + Pearson/Spearman spatial-correlation step; reimplemented clean-room).
- Spatial-multi-omics MALDI-MSI + ST single-cell integration (2025). *Scientific
  Reports*. doi:10.1038/s41598-025-26735-1  (same-section assumption; binning the
  coarser assay onto the finer grid).
- Mass spectrometry imaging for spatially resolved multi-omics (2024). *npj
  Imaging*. doi:10.1038/s44303-024-00025-3  (modality-agnostic data model: a second
  assay = a values-per-coordinate matrix — motivates the generic importer).
- Multimodal MSI and LCM-LC-MS/MS integration (2025). *PROTEOMICS*.
  doi:10.1002/pmic.202400378  (keeps the second modality a generic feature x
  location matrix so spatial proteomics fits the same model as transcriptomics).
- Pearson, K. (1901). On lines and planes of closest fit to systems of points in
  space. *Philosophical Magazine*, 2(11), 559-572.
  doi:10.1080/14786440109462720  (the linear-correlation statistic).
- Spearman, C. (1904). The proof and measurement of association between two things.
  *American Journal of Psychology*, 15(1), 72-101. doi:10.2307/1412159  (rank
  correlation, for the monotone-but-nonlinear metabolite-gene case).
- McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform Manifold
  Approximation and Projection for dimension reduction. *arXiv*:1802.03426.
  doi:10.48550/arXiv.1802.03426  (joint embedding, via ``multivariate.embedding``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_TOL_PPM

from . import spatial
from .spatial import RegionCorrelation, _profile_similarity

_COMAP_EXTRA_HINT = (
    "Install the optional multi-omics co-mapping extra:\n"
    "    uv sync --extra comap"
)


# --------------------------------------------------------------------------- #
# Modality-agnostic second-assay grid
# --------------------------------------------------------------------------- #
@dataclass
class ModalityGrid:
    """A second assay measured at spatial locations on the SAME tissue frame as an
    :class:`~smile_msi.msi.MSIDataset`.

    Modality-agnostic: transcriptomics (gene counts / spot), proteomics / IMC
    (protein / channel), or a generic ``feature x location`` matrix. Holds **no**
    m/z. Coordinates live in the grid's *own* (source) frame; a registration
    ``transform`` (or ``resample_to_msi``'s ``transform=`` argument) maps them onto
    the MSI frame.

    Attributes
    ----------
    kind:
        ``"transcriptomics" | "proteomics" | "generic"``.
    feature_names:
        Gene / protein / channel names — one per column of :attr:`values`.
    coords_um:
        ``(n_loc, 2)`` physical ``(x, y)`` location coordinates in the source frame
        (microns where known; arbitrary same-frame units otherwise).
    values:
        ``(n_loc, n_feat)`` per-location intensity / count. A dense ``np.ndarray`` or
        a scipy sparse CSR/CSC matrix (transcriptomics is sparse); densified one
        column at a time by :meth:`feature_vector`.
    spot_diameter_um:
        Capture-spot diameter, when known (Visium ~55 um).
    source:
        File / directory the grid was imported from.
    transform:
        Plan-05 registration transform (modality frame -> MSI frame). ``None`` =
        identity (same grid). Accepted forms mirror :func:`resample_to_msi`.
    meta:
        Free-form provenance dict.
    """

    kind: str
    feature_names: list
    coords_um: np.ndarray
    values: object
    spot_diameter_um: float | None = None
    source: str = ""
    transform: object = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.feature_names = list(self.feature_names)
        self.coords_um = np.asarray(self.coords_um, dtype=float)
        if self.coords_um.ndim != 2 or self.coords_um.shape[1] != 2:
            raise ValueError(
                f"coords_um must be (n_loc, 2); got shape {self.coords_um.shape}."
            )
        n_loc = self.coords_um.shape[0]
        n_feat = _n_cols(self.values)
        if _n_rows(self.values) != n_loc:
            raise ValueError(
                f"values has {_n_rows(self.values)} rows but coords_um has {n_loc}."
            )
        if len(self.feature_names) != n_feat:
            raise ValueError(
                f"feature_names ({len(self.feature_names)}) must match the "
                f"{n_feat} value columns."
            )

    @property
    def n_loc(self) -> int:
        return int(self.coords_um.shape[0])

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def feature_vector(self, name) -> np.ndarray:
        """Dense ``(n_loc,)`` column for one feature (raw, source frame).

        ``name`` is either a feature name (looked up in :attr:`feature_names`) or an
        integer column index. Densifies only that column when :attr:`values` is
        sparse.
        """
        if isinstance(name, (int, np.integer)):
            j = int(name)
        else:
            try:
                j = self.feature_names.index(name)
            except ValueError as e:
                raise KeyError(f"unknown modality feature {name!r}") from e
        return _dense_col(self.values, j)

    def dense(self) -> np.ndarray:
        """The full ``(n_loc, n_feat)`` values as a dense ``np.ndarray``."""
        return _as_dense(self.values)


# --------------------------------------------------------------------------- #
# sparse-tolerant helpers (no hard scipy.sparse import requirement)
# --------------------------------------------------------------------------- #
def _is_sparse(v) -> bool:
    return hasattr(v, "tocsc") and hasattr(v, "toarray")


def _n_rows(v) -> int:
    return int(v.shape[0])


def _n_cols(v) -> int:
    return int(v.shape[1])


def _as_dense(v) -> np.ndarray:
    if _is_sparse(v):
        return np.asarray(v.toarray(), dtype=float)
    return np.asarray(v, dtype=float)


def _dense_col(v, j: int) -> np.ndarray:
    if _is_sparse(v):
        return np.asarray(v[:, j].toarray(), dtype=float).ravel()
    return np.asarray(v, dtype=float)[:, j]


# --------------------------------------------------------------------------- #
# Importers
# --------------------------------------------------------------------------- #
def load_modality(path: str, *, kind: str = "generic",
                  feature_col: str | None = None) -> ModalityGrid:
    """Import a second modality from disk. Dispatches on extension / structure:

    * ``.h5ad`` -> AnnData (``X`` + ``obsm['spatial']``); **lazy** ``import anndata``
      (the optional ``comap`` extra).
    * a directory -> Space Ranger style: a matrix file + a ``tissue_positions`` CSV
      (numpy / pandas, core).
    * ``.csv`` / ``.tsv`` -> generic: one row per location, feature columns plus
      ``x``/``y`` (or ``x_um``/``y_um``) coordinate columns (numpy / pandas, core).

    Parameters
    ----------
    path:
        File or directory to import.
    kind:
        Stored on the resulting grid (``"transcriptomics" | "proteomics" |
        "generic"``); does not change the parser.
    feature_col:
        For the generic CSV path, unused for the default wide layout; reserved for a
        long/tidy layout (a single value column named ``feature_col``).

    Returns
    -------
    ModalityGrid
        In its own (source) coordinate frame.

    Raises
    ------
    ImportError
        For an ``.h5ad`` file when ``anndata`` is not installed (names the ``comap``
        extra).
    ValueError
        For an unrecognised / malformed file.
    """
    import os

    p = str(path)
    lower = p.lower()
    if os.path.isdir(p):
        return _load_space_ranger(p, kind=kind)
    if lower.endswith(".h5ad"):
        return _load_h5ad(p, kind=kind)
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower.endswith((".tsv", ".txt")) else ","
        return _load_generic_csv(p, kind=kind, sep=sep, feature_col=feature_col)
    raise ValueError(
        f"Unrecognised modality file {p!r}: expected a .h5ad, a Space Ranger "
        f"directory, or a .csv/.tsv with x/y coordinate columns."
    )


_X_ALIASES = ("x_um", "x", "px_x", "pxl_col_in_fullres", "col", "imagecol", "x_coord")
_Y_ALIASES = ("y_um", "y", "px_y", "pxl_row_in_fullres", "row", "imagerow", "y_coord")


def _pick_coord_cols(columns) -> tuple[str, str]:
    lut = {str(c).lower(): c for c in columns}
    xc = next((lut[a] for a in _X_ALIASES if a in lut), None)
    yc = next((lut[a] for a in _Y_ALIASES if a in lut), None)
    if xc is None or yc is None:
        raise ValueError(
            "could not find coordinate columns; expected one of "
            f"{_X_ALIASES} for x and {_Y_ALIASES} for y. Got columns: {list(columns)}."
        )
    return xc, yc


def _load_generic_csv(path: str, *, kind: str, sep: str,
                      feature_col: str | None) -> ModalityGrid:
    import pandas as pd

    df = pd.read_csv(path, sep=sep)
    if df.shape[1] < 3:
        raise ValueError(
            f"{path!r} needs x/y coordinate columns + at least one feature column."
        )
    xc, yc = _pick_coord_cols(df.columns)
    coords = df[[xc, yc]].to_numpy(dtype=float)

    if feature_col is not None:
        # Long / tidy layout: a single value column + a location id; pivot to wide.
        # (Kept simple: treat the value column as the one feature.)
        if feature_col not in df.columns:
            raise ValueError(f"feature_col {feature_col!r} not in columns {list(df.columns)}.")
        values = df[[feature_col]].to_numpy(dtype=float)
        return ModalityGrid(kind=kind, feature_names=[str(feature_col)],
                            coords_um=coords, values=values, source=str(path))

    # Wide layout: every non-coordinate, non-id column is a feature.
    id_like = {str(c).lower() for c in (xc, yc)}
    id_like |= {"barcode", "spot", "spot_id", "location", "loc", "id", "in_tissue"}
    feat_cols = [c for c in df.columns if str(c).lower() not in id_like]
    feat_cols = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)]
    if not feat_cols:
        raise ValueError(f"{path!r} has no numeric feature columns besides coordinates.")
    values = df[feat_cols].to_numpy(dtype=float)
    return ModalityGrid(kind=kind, feature_names=[str(c) for c in feat_cols],
                        coords_um=coords, values=values, source=str(path))


def _load_space_ranger(path: str, *, kind: str) -> ModalityGrid:
    """Space Ranger style directory: a tissue-positions CSV + a feature matrix.

    Supports the common nerve-tissue export: a ``tissue_positions(_list).csv`` for
    coordinates and a wide counts CSV (``*matrix*.csv`` / ``*counts*.csv``, spots x
    genes) — numpy / pandas only. The compressed ``.mtx`` 10x bundle is *not* parsed
    here (use the ``.h5ad`` path for that).
    """
    import glob
    import os

    import pandas as pd

    pos = (glob.glob(os.path.join(path, "*tissue_positions*.csv"))
           or glob.glob(os.path.join(path, "*positions*.csv")))
    mats = (glob.glob(os.path.join(path, "*matrix*.csv"))
            or glob.glob(os.path.join(path, "*counts*.csv"))
            or glob.glob(os.path.join(path, "*expression*.csv")))
    if not pos or not mats:
        raise ValueError(
            f"{path!r} does not look like a Space Ranger export: need a "
            f"*tissue_positions*.csv and a *matrix*/*counts*.csv. "
            f"For a 10x .h5ad bundle, point load_modality at the .h5ad file."
        )
    posdf = pd.read_csv(pos[0])
    matdf = pd.read_csv(mats[0], index_col=0)            # rows = spots, cols = genes

    xc, yc = _pick_coord_cols(posdf.columns)
    # Align the position table to the matrix's spot order by the shared id column.
    id_col = next((c for c in posdf.columns
                   if str(c).lower() in ("barcode", "spot", "spot_id", "id")), None)
    if id_col is not None:
        posdf = posdf.set_index(id_col)
        common = [b for b in matdf.index if b in posdf.index]
        if not common:
            raise ValueError(f"{path!r}: no shared spot ids between positions and matrix.")
        matdf = matdf.loc[common]
        posdf = posdf.loc[common]
    coords = posdf[[xc, yc]].to_numpy(dtype=float)
    feat_cols = [c for c in matdf.columns if np.issubdtype(matdf[c].dtype, np.number)]
    values = matdf[feat_cols].to_numpy(dtype=float)
    return ModalityGrid(kind=kind, feature_names=[str(c) for c in feat_cols],
                        coords_um=coords, values=values, source=str(path))


def _load_h5ad(path: str, *, kind: str) -> ModalityGrid:
    try:
        import anndata as ad
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "Importing .h5ad spatial data needs the 'anndata' backend. "
            + _COMAP_EXTRA_HINT
        ) from e

    adata = ad.read_h5ad(path)
    if "spatial" not in adata.obsm:
        raise ValueError(
            f"{path!r}: AnnData has no obsm['spatial'] coordinates to co-map."
        )
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    values = adata.X
    if hasattr(values, "tocsc"):                          # keep scipy sparse sparse
        values = values.tocsc()
    else:
        values = np.asarray(values, dtype=float)
    feature_names = [str(v) for v in list(adata.var_names)]
    diam = None
    try:
        sp = adata.uns.get("spatial", {})
        for lib in sp.values():
            diam = float(lib.get("scalefactors", {}).get("spot_diameter_fullres")) or diam
            break
    except Exception:  # noqa: BLE001 - scalefactors are best-effort
        diam = None
    return ModalityGrid(kind=kind, feature_names=feature_names, coords_um=coords,
                        values=values, spot_diameter_um=diam, source=str(path))


# --------------------------------------------------------------------------- #
# Transform application (consumes plan-05 registration output)
# --------------------------------------------------------------------------- #
def _transform_matrix(transform) -> np.ndarray | None:
    """Coerce a ``transform`` argument to a 3x3 homogeneous matrix, or ``None`` for
    identity / callable transforms handled separately.

    Accepts:
      * ``None`` -> identity (same grid).
      * a :class:`~smile_msi.registration.RegistrationResult` (uses ``.matrix``;
        raises for a non-linear piecewise/bspline result).
      * a 3x3 / 2x3 array-like homogeneous matrix.
      * a dict ``{"matrix": [...]}`` (the persisted plan-05 transform shape).
    """
    if transform is None:
        return None
    # RegistrationResult-like
    if hasattr(transform, "matrix") and not isinstance(transform, dict):
        M = getattr(transform, "matrix")
        if M is None:
            raise ValueError(
                "resample_to_msi got a non-linear registration result (no 3x3 "
                "matrix); use warp_points on the modality coordinates first and pass "
                "the warped grid, or use a linear transform."
            )
        return _as_homogeneous(M)
    if isinstance(transform, dict):
        if transform.get("matrix") is not None:
            return _as_homogeneous(transform["matrix"])
        raise ValueError("transform dict must carry a 'matrix' key (3x3 homogeneous).")
    return _as_homogeneous(transform)


def _as_homogeneous(M) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.shape == (3, 3):
        return M
    if M.shape == (2, 3):
        out = np.eye(3)
        out[:2, :] = M
        return out
    raise ValueError(f"transform matrix must be 3x3 or 2x3; got shape {M.shape}.")


def _apply_transform(transform, pts: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` modality coordinates into the MSI source frame.

    Linear transforms apply the homogeneous matrix; a *callable* transform is
    called directly (so a plan-05 piecewise/bspline warp can be passed as a function
    ``pts -> pts``). ``None`` is identity.
    """
    pts = np.asarray(pts, dtype=float)
    if transform is None:
        return pts
    if callable(transform) and not hasattr(transform, "matrix"):
        return np.asarray(transform(pts), dtype=float)
    M = _transform_matrix(transform)
    if M is None:
        return pts
    n = pts.shape[0]
    hom = np.hstack([pts, np.ones((n, 1))])
    out = hom @ M.T
    w = out[:, 2:3]
    w = np.where(w == 0, 1.0, w)
    return out[:, :2] / w


# --------------------------------------------------------------------------- #
# Resampling onto the MSI pixel grid
# --------------------------------------------------------------------------- #
def _msi_pixel_centroids(ds) -> tuple[np.ndarray, float]:
    """Per-pixel ``(x, y)`` centroids in the PRE-rotation (orientation-0) acquisition
    frame — the same frame :meth:`MSIDataset._pixel_rows_cols` / ``feature_matrix``
    index in — scaled to physical units by ``ds.pixel_size_um`` when available.

    Binning is done in this frame (not the rotated display frame), so a correlation
    is invariant to ``ds.orientation`` (asserted in tests). Returns
    ``(centroids (n_pix, 2), pixel_pitch)``.
    """
    coords = np.asarray(ds.coordinates, dtype=float)
    x0 = float(coords[:, 0].min())
    y0 = float(coords[:, 1].min())
    xy = np.column_stack([coords[:, 0] - x0, coords[:, 1] - y0])
    pitch = getattr(ds, "pixel_size_um", None)
    if pitch is None or not np.isfinite(pitch) or pitch <= 0:
        pitch = 1.0
    return xy * float(pitch), float(pitch)


def resample_to_msi(grid: ModalityGrid, ds, transform=None, *,
                    agg: str = "mean") -> tuple[np.ndarray, list]:
    """Map the modality grid onto the MSI pixel grid.

    Each modality location is moved into the MSI source frame by ``transform``
    (modality -> MSI; ``None`` = same-grid identity), then assigned to the **nearest**
    MSI pixel (using ``ds.pixel_size_um`` for the pixel pitch when available, else
    unit spacing). Locations farther than half a pixel pitch from any pixel are
    dropped (they fall off the grid). ``agg`` (``mean`` | ``sum`` | ``max``)
    aggregates the locations that land in the same pixel.

    Binning happens in the pre-rotation pixel frame
    (:meth:`MSIDataset._pixel_rows_cols` / ``feature_matrix`` order), so the result
    is aligned to the dataset's **pixel order** and is independent of
    ``ds.orientation`` — display rotation is applied only by ``ds.to_image``, never
    inside the binning.

    Parameters
    ----------
    grid:
        The second-modality grid.
    ds:
        The MSI dataset whose pixel grid to resample onto.
    transform:
        Registration transform (modality frame -> MSI frame). ``None`` = identity.
        Accepts a :class:`~smile_msi.registration.RegistrationResult`, a 3x3/2x3
        matrix, a ``{"matrix": ...}`` dict, or a callable ``pts -> pts``.
    agg:
        ``"mean"`` (default), ``"sum"`` or ``"max"``.

    Returns
    -------
    (M, feature_names)
        ``M`` is ``(n_pixels, n_feat)`` aligned to the dataset's pixel order; pixels
        with no overlapping modality location are ``NaN``.

    Notes
    -----
    Standard point-in-grid binning after the registration map (Vicari et al. 2024,
    doi:10.1038/s41587-023-01937-y — same-grid convention).
    """
    if agg not in ("mean", "sum", "max"):
        raise ValueError(f"agg must be mean/sum/max; got {agg!r}.")
    centroids, pitch = _msi_pixel_centroids(ds)
    n_pix = centroids.shape[0]
    n_feat = grid.n_features

    src = _apply_transform(transform, grid.coords_um)    # modality -> MSI frame
    # shift modality coords into the same offset frame as the centroids
    coords = np.asarray(ds.coordinates, dtype=float)
    x0, y0 = float(coords[:, 0].min()), float(coords[:, 1].min())
    src_um = np.column_stack([(src[:, 0] - x0) * pitch, (src[:, 1] - y0) * pitch])

    out = np.full((n_pix, n_feat), np.nan, dtype=float)
    if n_pix == 0 or grid.n_loc == 0:
        return out, list(grid.feature_names)

    # Nearest MSI pixel per modality location via a KD-tree (scipy, core).
    from scipy.spatial import cKDTree

    tree = cKDTree(centroids)
    dist, idx = tree.query(src_um, k=1)
    # keep only locations within half a pixel pitch of a centroid (on-grid)
    keep = dist <= (pitch / 2.0 + 1e-9)
    idx = idx[keep]
    dense = _as_dense(grid.values)[keep]
    if idx.size == 0:
        return out, list(grid.feature_names)

    sums = np.zeros((n_pix, n_feat), dtype=float)
    counts = np.zeros(n_pix, dtype=float)
    np.add.at(counts, idx, 1.0)
    if agg == "max":
        maxv = np.full((n_pix, n_feat), -np.inf, dtype=float)
        for loc, pix in enumerate(idx):
            np.maximum(maxv[pix], dense[loc], out=maxv[pix])
        hit = counts > 0
        out[hit] = maxv[hit]
        return out, list(grid.feature_names)

    np.add.at(sums, idx, dense)
    hit = counts > 0
    if agg == "sum":
        out[hit] = sums[hit]
    else:                                                # mean
        out[hit] = sums[hit] / counts[hit][:, None]
    return out, list(grid.feature_names)


# --------------------------------------------------------------------------- #
# Cross-modality spatial correlation
# --------------------------------------------------------------------------- #
@dataclass
class CrossCorr:
    """Result of :func:`cross_modality_correlation`.

    ``matrix`` is ``(n_peaks, n_feat)`` — the spatial correlation of each ion image
    with each modality-feature image over the shared pixel set.
    """
    matrix: np.ndarray          # (n_peaks, n_feat)
    peaks: np.ndarray           # (n_peaks,) m/z axis
    feature_names: list         # (n_feat,) modality feature names
    n_pixels: int               # pixels in the shared, non-NaN, in-mask set
    method: str                 # 'pearson' | 'spearman'

    def top_pairs(self, k: int = 10) -> list:
        """The ``k`` strongest ``(mz, feature, score)`` pairs by ``|score|``."""
        M = self.matrix
        flat = np.argsort(-np.abs(np.nan_to_num(M, nan=0.0)), axis=None)[:k]
        out = []
        for f in flat:
            i, j = np.unravel_index(f, M.shape)
            out.append((float(self.peaks[i]), self.feature_names[j], float(M[i, j])))
        return out


def _rankdata_cols(X: np.ndarray) -> np.ndarray:
    """Average-rank transform each column of ``X`` (for Spearman = Pearson on ranks)."""
    from scipy.stats import rankdata

    return rankdata(np.asarray(X, dtype=float), axis=0)


def _corr_blocks(A: np.ndarray, B: np.ndarray, method: str) -> np.ndarray:
    """Column-wise correlation between every column of ``A`` and every column of
    ``B`` (shared rows): ``R[i, j] = corr(A[:, i], B[:, j])``.

    Vectorized: standardize both blocks column-wise (subtract mean, divide by L2 of
    the centred column), then ``R = Za.T @ Zb``. Spearman first rank-transforms each
    column (Spearman 1904, doi:10.2307/1412159); Pearson uses the raw values
    (Pearson 1901, doi:10.1080/14786440109462720).
    """
    if method == "spearman":
        A = _rankdata_cols(A)
        B = _rankdata_cols(B)
    elif method != "pearson":
        raise ValueError(f"method must be pearson/spearman; got {method!r}.")
    Za = _standardize(A)
    Zb = _standardize(B)
    return Za.T @ Zb


def _standardize(X: np.ndarray) -> np.ndarray:
    """Centre each column and divide by its L2 norm; zero-variance columns -> 0 (so a
    flat ion image correlates to NaN-free 0, not a divide-by-zero)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(Xc, axis=0, keepdims=True)
    nrm = np.where(nrm > 1e-12, nrm, np.nan)             # flat columns -> NaN corr
    return Xc / nrm


def cross_modality_correlation(ds, peaks, grid: ModalityGrid, *, transform=None,
                               tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                               method: str = "pearson", mask=None,
                               min_pixels: int = 30) -> CrossCorr:
    """Per-``(m/z peak, modality feature)`` spatial correlation of co-registered
    per-pixel images.

    Builds the MSI block with :func:`smile_msi.spatial.feature_matrix` and the
    modality block with :func:`resample_to_msi`, restricts to the shared non-NaN
    pixel set inside ``mask`` (or detected foreground when ``mask`` is ``None``),
    requires at least ``min_pixels`` shared pixels, then correlates column-wise.
    ``method`` is ``"pearson"`` (default) or ``"spearman"`` (rank-transform each
    column first — robust to a monotone-but-nonlinear metabolite-gene link).

    Returns
    -------
    CrossCorr
        ``matrix`` is ``(n_peaks, n_feat)``.

    Refs: Vicari et al. (2024) doi:10.1038/s41587-023-01937-y; Pearson (1901)
    doi:10.1080/14786440109462720; Spearman (1904) doi:10.2307/1412159.
    """
    Xmsi = spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)   # (n_pix, n_peaks)
    Xmod, feat_names = resample_to_msi(grid, ds, transform=transform)      # (n_pix, n_feat)

    if mask is None:
        mask = spatial._foreground_mask(ds)
    m = np.ones(Xmsi.shape[0], dtype=bool) if mask is None else np.asarray(mask, bool)
    # shared = in-mask AND every modality feature present (non-NaN) at that pixel
    valid = m & np.all(np.isfinite(Xmod), axis=1)
    if int(valid.sum()) < int(min_pixels):
        raise ValueError(
            f"only {int(valid.sum())} shared pixels overlap the modality grid "
            f"(need >= {min_pixels}); check the transform / grid alignment."
        )
    A = Xmsi[valid]
    B = Xmod[valid]
    R = _corr_blocks(A, B, method)
    pk = np.asarray([float(p["mz"]) if isinstance(p, dict) else float(p) for p in peaks],
                    dtype=float)
    return CrossCorr(matrix=R, peaks=pk, feature_names=list(feat_names),
                     n_pixels=int(valid.sum()), method=method)


# --------------------------------------------------------------------------- #
# Joint embedding (MSI + modality, concatenated)
# --------------------------------------------------------------------------- #
def joint_embedding(ds, peaks, grid: ModalityGrid, *, transform=None,
                    method: str = "umap", weight: float = 0.5,
                    tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                    log1p: bool = True, mask=None, random_state: int = 0):
    """Joint 2-D embedding from z-scored MSI features (+) z-scored modality features.

    Each block is column z-scored, then L2-weighted by ``weight`` in ``[0, 1]``
    (``weight`` on the MSI block, ``1 - weight`` on the modality block) and
    concatenated on the shared (non-NaN, in-mask) pixel set. The concatenated matrix
    is fed to :func:`smile_msi.multivariate.embedding` (UMAP if installed, else
    t-SNE) wrapped as a tiny in-memory dataset, so the returned
    :class:`~smile_msi.multivariate.Embedding` carries ``coords``/``index`` mapping
    back to the *original* MSI pixels (the GUI / region-creation path is unchanged).

    ``random_state`` is threaded from the active analysis profile. Refs: McInnes et
    al. (2018) arXiv:1802.03426; Vicari et al. (2024) doi:10.1038/s41587-023-01937-y.
    """
    from . import multivariate

    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight must be in [0, 1]; got {weight}.")
    Xmsi = spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    Xmod, feat_names = resample_to_msi(grid, ds, transform=transform)
    if log1p:
        Xmsi = np.log1p(np.clip(Xmsi, 0, None))

    if mask is None:
        mask = spatial._foreground_mask(ds)
    m = np.ones(Xmsi.shape[0], dtype=bool) if mask is None else np.asarray(mask, bool)
    valid_full = m & np.all(np.isfinite(Xmod), axis=1)
    index = np.flatnonzero(valid_full)
    if index.size < 3:
        raise ValueError(
            f"only {index.size} shared pixels for the joint embedding (need >= 3)."
        )

    Zmsi = _block_zscore(Xmsi[index])
    Zmod = _block_zscore(Xmod[index])
    # per-block L2 weighting (each block normalised to unit Frobenius then scaled)
    Zmsi = _block_scale(Zmsi, float(weight))
    Zmod = _block_scale(Zmod, float(1.0 - weight))
    Xjoint = np.hstack([Zmsi, Zmod])

    # Build a lightweight dataset double whose pixel order == the SHARED pixels, so
    # multivariate.embedding's index/geometry maps back to the real tissue pixels.
    rows, cols = ds._pixel_rows_cols()
    sub = _JointDataset(ds, index, Xjoint)
    emb = multivariate.embedding(
        sub, np.arange(Xjoint.shape[1], dtype=float), method=method,
        tol_ppm=tol_ppm, norm="none", log1p=False, random_state=random_state,
        standardize=False,            # Xjoint is already per-block z-scored + weight-scaled
    )
    # remap the embedding index from sub-pixel order to absolute ds pixel ids
    emb.index = index[np.asarray(emb.index, dtype=int)]
    return emb


def _block_zscore(X: np.ndarray) -> np.ndarray:
    # Per-column z-score; canonical impl is spatial._zscore_cols (identical math).
    return spatial._zscore_cols(np.asarray(X, dtype=float))


def _block_scale(X: np.ndarray, w: float) -> np.ndarray:
    """Normalise a block to unit Frobenius norm then scale by ``w`` — so the two
    blocks contribute in the ratio ``weight : 1 - weight`` regardless of their
    feature counts."""
    fro = np.linalg.norm(X)
    if fro > 1e-12:
        X = X / fro
    return X * float(w)


class _JointDataset:
    """Minimal MSIDataset-shaped double exposing exactly what
    :func:`multivariate.embedding` touches: ``_pixel_rows_cols``, ``to_image``,
    ``height``/``width`` (for the RGB tissue map) and a pre-built feature matrix via
    ``ensure_features`` / ``feature_matrix``. Pixels are the SHARED set, in order.
    """

    def __init__(self, ds, index: np.ndarray, X: np.ndarray):
        self._ds = ds
        self._index = np.asarray(index, dtype=int)
        self._X = np.asarray(X, dtype=float)
        coords = np.asarray(ds.coordinates, dtype=float)[self._index]
        self.coordinates = coords
        self.orientation = 0
        self.pixel_size_um = getattr(ds, "pixel_size_um", None)

    # geometry mirrors the orientation-0 path of MSIDataset
    @property
    def _base_hw(self):
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        return (int(y.max() - y.min()) + 1, int(x.max() - x.min()) + 1)

    @property
    def x_range(self):
        return int(self.coordinates[:, 0].min()), int(self.coordinates[:, 0].max())

    @property
    def y_range(self):
        return int(self.coordinates[:, 1].min()), int(self.coordinates[:, 1].max())

    @property
    def height(self):
        return self._base_hw[0]

    @property
    def width(self):
        return self._base_hw[1]

    @property
    def n_pixels(self):
        return int(self._index.shape[0])

    def _pixel_rows_cols(self):
        x0, _ = self.x_range
        y0, _ = self.y_range
        r = (self.coordinates[:, 1] - y0).astype(int)
        c = (self.coordinates[:, 0] - x0).astype(int)
        return r, c

    def to_image(self, values, fill=np.nan):
        values = np.asarray(values, dtype=float)
        img = np.full((self.height, self.width), fill, dtype=float)
        r, c = self._pixel_rows_cols()
        img[r, c] = values
        return img

    def ensure_features(self, peaks, tol_ppm=DEFAULT_TOL_PPM, reduce="sum"):
        return None

    def feature_matrix(self, norm="tic"):
        return self._X


# --------------------------------------------------------------------------- #
# Cross-modality region comparison
# --------------------------------------------------------------------------- #
def cross_region_comparison(ds, peaks, grid: ModalityGrid, region_masks: dict, *,
                            transform=None, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                            method: str = "pearson") -> RegionCorrelation:
    """Region-level cross-modality fingerprints.

    Each region collapses to a concatenated fingerprint ``[MSI mean profile ||
    modality mean profile (resampled)]``, and regions are scored region×region by the
    same normalized-profile kernel (:func:`smile_msi.spatial._profile_similarity`)
    that backs :func:`smile_msi.spatial.region_correlation`. Lets the user ask whether
    MSI-defined nerve compartments separate the same way in the second assay.

    The modality block is z-scored across regions (and the MSI block likewise) before
    concatenation so the two assays' very different intensity scales contribute
    comparably to the profile correlation. Pixels in a region with no overlapping
    modality location are simply excluded from that region's modality mean.

    ``region_masks`` maps region name -> per-pixel boolean mask (dataset pixel order).
    ``method`` is ``"pearson"`` (profile correlation) or ``"cosine"`` — matching
    :func:`region_correlation`.

    Returns
    -------
    smile_msi.spatial.RegionCorrelation
        So the result renders in the existing region-correlation view.

    Refs: Vicari et al. (2024) doi:10.1038/s41587-023-01937-y; npj Imaging (2024)
    doi:10.1038/s44303-024-00025-3.
    """
    names = [nm for nm, mk in region_masks.items()
             if mk is not None and np.asarray(mk, bool).any()]
    if len(names) < 2:
        raise ValueError("cross-region comparison needs at least two non-empty regions")

    Xmsi = spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    Xmod, feat_names = resample_to_msi(grid, ds, transform=transform)

    msi_fps = []
    mod_fps = []
    for nm in names:
        mk = np.asarray(region_masks[nm], bool)
        msi_fps.append(Xmsi[mk].mean(axis=0))
        sub = Xmod[mk]
        ok = np.all(np.isfinite(sub), axis=1)
        if ok.any():
            mod_fps.append(sub[ok].mean(axis=0))
        else:                                            # no modality coverage in region
            mod_fps.append(np.full(Xmod.shape[1], np.nan))
    msi_fps = np.vstack(msi_fps)
    mod_fps = np.vstack(mod_fps)

    # z-score each block across regions so neither assay's scale dominates; NaN
    # (uncovered) modality means -> 0 contribution.
    msi_z = _block_zscore(msi_fps)
    mod_z = _block_zscore(np.nan_to_num(mod_fps, nan=np.nanmean(mod_fps) if
                                        np.isfinite(mod_fps).any() else 0.0))
    fps = np.hstack([msi_z, mod_z])
    M = _profile_similarity(fps, method=method)

    pk = np.asarray([float(p["mz"]) if isinstance(p, dict) else float(p) for p in peaks],
                    dtype=float)
    # axis carries the MSI m/z plus the modality feature names (string-typed object array)
    axis = np.array([*[float(v) for v in pk], *list(feat_names)], dtype=object)
    return RegionCorrelation(names=names, matrix=M, fingerprints=fps, peaks=axis,
                             method=method)
