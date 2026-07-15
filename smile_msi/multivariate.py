"""Multivariate / component analysis over an :class:`~smile_msi.msi.MSIDataset`
— component analysis and spatial methods.

* :func:`pca_images` — principal-component **score images** + loadings (which ions
  drive each component).
* :func:`nmf_images` — **non-negative** spatial components (additive parts; the
  most interpretable component imaging for MSI) + each component's top ions.
* :func:`embedding` — 2-D **t-SNE / UMAP** embedding of the pixels, plus an RGB
  "molecular similarity" map colouring each pixel by its position in that space.
* :func:`pooled_embedding` — one t-SNE/UMAP over pixels **pooled from many samples**
  on a shared feature axis, so a cohort's slides share a coordinate frame and the cloud
  can be coloured by sample or group (cross-sample comparison the single-slide
  :func:`embedding` can't do).
* :func:`pooled_region_embedding` — the same, but **one point per region** (or per
  spatially-connected region *instance*) instead of per pixel: the kidney atlas'
  FTU-instance-level embedding (Farrow et al. 2025, Fig 4B/D), colourable by structure.
* :func:`spatial_segment` — spatially-aware segmentation: features are smoothed
  over each pixel's neighbourhood before clustering, giving coherent regions
  (a pragmatic analogue of Cardinal's spatially-aware shrunken centroids).
* :func:`spatial_dgmm` / :func:`segmentation_test` — per-ion spatially-aware Gaussian
  mixture segmentation and the differential-abundance test built on it (Cardinal's
  spatialDGMM + segmentationTest).
* :func:`plsda` / :func:`cross_validate` — supervised PLS-DA / OPLS-DA with VIP scores
  and leave-one-sample-out cross-validation.
"""
from __future__ import annotations

import hashlib
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_TOL_PPM

from . import spatial


@dataclass
class Components:
    method: str
    images: list                 # list of (height, width) score/abundance images
    loadings: np.ndarray         # (n_components, n_peaks)
    peaks: np.ndarray            # m/z per loading column
    explained_variance: np.ndarray  # per-component (PCA); empty for NMF
    scores: np.ndarray = None    # (n_pixels, n_components) per-pixel scores — drives the
                                 # linked feature-space scatter; ``None`` if not retained

    def top_peaks(self, k: int, n: int = 5):
        """m/z with the largest loadings for component ``k``."""
        order = np.argsort(-np.abs(self.loadings[k]))[:n]
        return [(float(self.peaks[j]), float(self.loadings[k, j])) for j in order]


@dataclass
class Embedding:
    """Result of :func:`embedding` — a 2-D pixel embedding kept *with* the pixels it
    came from, so a GUI can link each scatter point back to its tissue location."""
    method: str                  # 'UMAP' / 'TSNE' (the method requested)
    coords: np.ndarray           # (n_embedded, 2) — 2-D position of each embedded pixel
    rgba: np.ndarray             # (height, width, 4) molecular-similarity tissue map
    index: np.ndarray            # (n_embedded,) pixel index of each row of ``coords``
    features: np.ndarray = None  # (n_embedded, n_peaks) per-pixel values (pre-standardize),
                                 # retained only when keep_features=True — lets a viewer colour
                                 # the cloud by any ion's intensity. ``None`` otherwise.
    peaks: np.ndarray = None     # (n_peaks,) m/z per ``features`` column (when retained)

    def point_rgb(self):
        """Per-point (n_embedded, 3) float RGB matching the tissue ``rgba`` colours, so
        a point in the scatter and its pixel in tissue share a colour."""
        return _coords_rgb(self.coords)


def _matrix(ds, peaks, tol_ppm, norm, log1p):
    X = spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    if log1p:
        X = np.log1p(np.clip(X, 0, None))
    return X


def _matrix_rows(ds, peaks, rows, tol_ppm, norm, log1p):
    """Like :func:`_matrix` but extracts only ``rows`` (a capped/region pixel subsample)
    without building or caching the whole-slide matrix — bounds the per-slide extraction
    RAM/CPU to the subsample instead of all n_pixels. See
    :meth:`smile_msi.msi.MSIDataset.features_for_rows`. Falls back to the full path then
    subsets for datasets lacking the method (e.g. test doubles)."""
    fn = getattr(ds, "features_for_rows", None)
    if fn is None:
        return _matrix(ds, peaks, tol_ppm, norm, log1p)[np.asarray(rows, dtype=int)]
    X = fn(peaks, rows, tol_ppm=tol_ppm, norm=norm)
    if log1p:
        X = np.log1p(np.clip(X, 0, None))
    return X


def pca_images(ds, peaks, n_components: int = 5, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
               log1p: bool = True, random_state: int = 0, mask=None) -> Components:
    """PCA component images. Refs: Pearson (1901), doi:10.1080/14786440109462720;
    Hotelling (1933), doi:10.1037/h0071325. ``mask`` (bool[n_pix]) fits the components
    on that region's pixels only (e.g. nerve-only), so the captured variance is
    intra-region; out-of-region pixels are blank (NaN) in the score images."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = _matrix(ds, peaks, tol_ppm, norm, log1p)
    m = None if mask is None else np.asarray(mask, bool)
    Xfit = X if m is None else X[m]
    scaler = StandardScaler().fit(Xfit)
    k = int(min(n_components, Xfit.shape[1], max(1, Xfit.shape[0] - 1)))
    pca = PCA(n_components=k, random_state=random_state).fit(scaler.transform(Xfit))
    scores = pca.transform(scaler.transform(X))                       # project every pixel
    images = [ds.to_image(scores[:, c] if m is None else np.where(m, scores[:, c], np.nan))
              for c in range(k)]
    return Components("PCA", images, pca.components_, np.asarray(peaks, float),
                      pca.explained_variance_ratio_, scores=scores)


def nmf_images(ds, peaks, n_components: int = 5, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
               random_state: int = 0, mask=None) -> Components:
    """NMF component images. Refs: Lee & Seung (1999), doi:10.1038/44565; NNDSVD init —
    Boutsidis & Gallopoulos (2008), doi:10.1016/j.patcog.2007.09.010. ``mask`` fits the
    parts on that region's pixels only; out-of-region pixels are blank (NaN)."""
    from sklearn.decomposition import NMF

    X = np.clip(spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm), 0, None)
    m = None if mask is None else np.asarray(mask, bool)
    Xfit = X if m is None else X[m]
    k = int(min(n_components, Xfit.shape[1], max(1, Xfit.shape[0] - 1)))
    nmf = NMF(n_components=k, init="nndsvda", random_state=random_state, max_iter=400)
    if m is None:
        W = nmf.fit_transform(X)
    else:
        nmf.fit(Xfit)
        W = nmf.transform(X)                                          # project every pixel
    images = [ds.to_image(W[:, c] if m is None else np.where(m, W[:, c], np.nan))
              for c in range(k)]
    return Components("NMF", images, nmf.components_, np.asarray(peaks, float), np.array([]),
                      scores=W)


def embedding(ds, peaks, method: str = "umap", tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
              log1p: bool = True, random_state: int = 0, sample: int | None = None,
              mask=None, *, metric: str = "euclidean", n_neighbors: int = 15,
              min_dist: float = 0.1, keep_features: bool = False,
              standardize: bool = True, deterministic: bool = True) -> Embedding:
    """2-D embedding of pixels (UMAP if installed, else t-SNE) + an RGB map.

    Returns an :class:`Embedding` carrying the 2-D ``coords`` (n_embedded, 2), the
    ``rgba`` tissue map colouring each embedded pixel by its 2-D position (so
    molecularly similar pixels share a colour in tissue space), and the ``index`` of
    which pixel each row of ``coords`` came from — so a caller can map a point in the
    scatter back to its tissue location.

    ``sample`` caps how many pixels are embedded (t-SNE/UMAP scale poorly): when set
    and smaller than the pixel count, a random subset is embedded and only those
    pixels are coloured (the rest stay transparent). Default ``None`` embeds every
    pixel. ``mask`` (bool[n_pix]) restricts the embedding to a region's pixels — only
    those are standardized, embedded and coloured — so a few ROIs can be embedded on
    their own molecular axes.

    Refs: t-SNE — van der Maaten & Hinton (2008), JMLR 9:2579–2605; UMAP — McInnes,
    Healy & Melville (2018), arXiv:1802.03426.
    """
    from sklearn.preprocessing import StandardScaler

    Xraw = _matrix(ds, peaks, tol_ppm, norm, log1p)
    rows, cols = ds._pixel_rows_cols()
    # ``index`` always holds absolute pixel ids so a scatter point maps back to its tissue
    # pixel; the mask prunes the pool first (standardized on that pool), then ``sample``
    # subsamples within it.
    index = np.arange(Xraw.shape[0])
    if mask is not None:
        index = index[np.asarray(mask, bool)]
    # ``standardize=False`` lets a caller that has *already* scaled its matrix (e.g.
    # comap.joint_embedding's per-block z-score + weight) skip re-standardization here —
    # otherwise StandardScaler would force every column back to unit variance and erase
    # that weighting.
    X = (StandardScaler().fit_transform(Xraw[index]) if standardize
         else np.asarray(Xraw[index], dtype=float))
    if sample is not None and 0 < int(sample) < index.shape[0]:
        rng = np.random.default_rng(random_state)
        sel = np.sort(rng.choice(index.shape[0], size=int(sample), replace=False))
        index, X = index[sel], X[sel]
    rows, cols = rows[index], cols[index]
    coords, used = _reduce_2d(X, method, random_state, metric=metric,
                              n_neighbors=n_neighbors, min_dist=min_dist,
                              deterministic=deterministic)
    rgba = _embedding_rgba(ds, coords, rows, cols)
    feats = Xraw[index] if keep_features else None    # pre-standardize values for colour-by-ion
    return Embedding(used.upper(), coords, rgba, index,
                     features=feats, peaks=(np.asarray(peaks, float) if keep_features else None))


@dataclass
class PooledEmbedding:
    """Result of :func:`pooled_embedding` — one 2-D embedding of pixels pooled from
    **many** samples on a shared feature axis, so a point cloud from different slides
    lives in a single coordinate frame and can be coloured by sample or cohort group."""
    method: str                  # 'UMAP' / 'TSNE' (the method actually used)
    coords: np.ndarray           # (N, 2) — 2-D position of every pooled pixel
    sample_id: np.ndarray        # (N,) int index into ``sample_names`` for each point
    group: np.ndarray            # (N,) cohort group label per point ('' = ungrouped)
    sample_names: list           # display name per sample id
    targets: np.ndarray          # shared m/z axis the embedding was built on
    counts: dict                 # sample name -> number of pixels embedded
    features: np.ndarray = None  # (N, n_targets) per-pixel values used (pre-standardize) —
                                 # retained only when keep_features=True, so a viewer can
                                 # colour the cloud by any feature's intensity. ``None`` else.
    # Per-point tissue geometry — so a cluster found in the embedding can be painted back
    # onto each slide (the linked spatial↔embedding "molecular histology" figure). All
    # ``None`` unless ``keep_geometry=True`` (the GUI's per-pixel path always asks).
    px_row: np.ndarray = None    # (N,) display-grid row of each point's pixel (within its slide)
    px_col: np.ndarray = None    # (N,) display-grid col of each point's pixel
    shapes: dict = None          # sample name -> (height, width) of that slide's display grid
    # sample name -> (rows, cols) of EVERY candidate tissue pixel on that slide (the full mask,
    # independent of the per-sample random cap). The embedded points (px_row/px_col) are only the
    # capped subset; this lets the cluster back-map fill the whole tissue (nearest-neighbour) so a
    # big slide isn't painted as 3% speckle on white. ``None`` unless ``keep_geometry=True``.
    tissue_rc: dict = None
    # samples that loaded but were NOT embedded (e.g. no signal on the feature axis), as
    # ``[(name, reason)]`` — so a caller can report "8 of 10 embedded (2 dropped)" instead of
    # silently misreporting the cohort/group n. Empty when every loaded sample was embedded.
    dropped: list = None


def _zscore_cols(X):
    """Per-feature z-score (mean 0, unit variance); zero-variance columns collapse to 0.
    Applied per sample before pooling so a slide's overall intensity offset/scale can't
    dominate the embedding — a light per-sample batch alignment. Canonical implementation
    lives in :func:`smile_msi.spatial._zscore_cols`; kept here as a thin re-export so both
    modules stay in lock-step."""
    return spatial._zscore_cols(X)


def _row_tic(ds, rows, fallback):
    """Per-pixel **total ion current** for ``rows`` (absolute pixel ids) — the normalization
    vector the cohort-global scaling anchors on. Prefers the dataset's full-spectrum TIC
    (``_pix['tic']`` after :meth:`~smile_msi.msi.MSIDataset.prime`); for lightweight dataset
    doubles that don't expose it, falls back to the peak-sum of the extracted ``fallback``
    matrix (a valid TIC proxy on the analysed feature set)."""
    rows = np.asarray(rows, dtype=int)
    prime = getattr(ds, "prime", None)
    try:
        if callable(prime):
            prime()                              # idempotent + cached; no-op if already primed
        pix = getattr(ds, "_pix", None)
        if isinstance(pix, dict) and pix.get("tic") is not None:
            return np.asarray(pix["tic"], dtype=float)[rows]
    except Exception:  # noqa: BLE001 — any prime/_pix issue → peak-sum proxy below
        pass
    return np.asarray(fallback, dtype=float).sum(axis=1)


def pooled_embedding(samples, targets, loader, *, method: str = "umap",
                     tol_ppm: float = 20.0, norm: str = "tic", log1p: bool = True,
                     per_sample_cap: int | None = 3000, standardize_per_sample: bool = True,
                     cohort_norm: bool = False,
                     batch_correct: bool = False, batch_mean_only: bool = False,
                     batch_protect_group: bool = True,
                     random_state: int = 0, progress=None, metric: str = "euclidean",
                     n_neighbors: int = 15, min_dist: float = 0.1,
                     keep_features: bool = False,
                     keep_geometry: bool = False) -> PooledEmbedding:
    """Pool pixels from many samples on a shared feature axis and run **one** 2-D embedding.

    For each entry of ``samples``, ``loader(sample)`` must return ``(ds, pixel_index)`` —
    the :class:`~smile_msi.msi.MSIDataset` and the pixel indices to keep (``None`` = the
    whole slide; pass a region's pixel mask to embed only that region). Returning ``None``
    (or ``(None, …)``) skips the sample. Each sample's per-pixel feature matrix is built on
    the *shared* ``targets`` (so every slide spans the same molecular axis), optionally
    ``log1p``'d and per-sample z-scored (``standardize_per_sample`` — removes slide-to-slide
    intensity offsets so a batch effect can't masquerade as biology), capped to
    ``per_sample_cap`` random pixels so a big slide doesn't swamp a small one and t-SNE/UMAP
    stay tractable, then **stacked** and embedded together via :func:`_reduce_2d`. Each point
    carries its originating sample and group, so a caller can colour the cloud by either.

    ``cohort_norm`` switches the cross-sample comparability strategy. By default each slide is
    normalized to *its own* mean intensity, which (as Farrow et al. 2025 warn) makes intensities
    **incomparable across slides** — two slides with identical biology end up differing by their
    brightness ratio — so ``standardize_per_sample`` z-scores it back out (aggressively: it also
    flattens genuine cross-sample abundance differences). With ``cohort_norm=True`` every slide's
    per-pixel TIC scaling is instead anchored to **one shared reference — the mean TIC of all
    pooled pixels across the whole cohort** — so same-biology slides line up while real abundance
    differences survive (the paper's gentler recipe; it uses the *median* there, but that exact
    form is a withheld patented method, so we anchor on the mean). Comparability then comes from
    the shared anchor, which **per-sample z-scoring would erase**, so ``cohort_norm`` forces
    ``standardize_per_sample`` off — the two are mutually exclusive.

    ``loader`` is injected (not hard-wired to imzML) so the pooling logic stays unit
    testable without a real dataset. ``progress(done, total)`` matches the GUI worker
    protocol. Refs as :func:`embedding`."""
    from sklearn.preprocessing import StandardScaler

    targets = np.asarray(list(targets), dtype=float)
    if targets.size == 0:
        raise ValueError("pooled_embedding needs a non-empty shared feature axis (targets).")
    rng = np.random.default_rng(random_state)
    samples = list(samples)
    total = len(samples) + 1
    blocks, ids, groups, names, counts = [], [], [], [], {}
    batches = []                                   # per-row acquisition-batch label (for ComBat)
    raw_blocks = [] if keep_features else None     # pre-standardize values for colour-by
    # per-point tissue geometry (row/col on each slide's display grid) for back-mapping
    px_rows, px_cols, shapes = ([], [], {}) if keep_geometry else (None, None, None)
    tissue_rc = {} if keep_geometry else None       # name -> full-tissue (rows, cols), pre-cap
    # Cohort-global TIC normalization: comparability comes from one shared anchor, so per-sample
    # z-scoring is both redundant and would erase it — the two are mutually exclusive. We can't
    # normalize per slide in the loop (the anchor needs every slide's TIC first), so collect each
    # block's RAW counts + per-pixel TIC and normalize in one pass once the anchor is known.
    if cohort_norm:
        standardize_per_sample = False
    pend_raw, pend_tic, block_slot = [], [], []
    gsum, gcnt = 0.0, 0
    n_loaded = 0                                   # samples that loaded with ≥1 candidate pixel
    dropped: list = []                             # (name, reason) for samples loaded but not embedded
    for si, s in enumerate(samples):
        if progress is not None:
            progress(si, total)
        loaded = loader(s)
        if not loaded:
            continue
        ds, pix = loaded
        if ds is None:
            continue
        # Candidate pixels for this sample: a region's pixels, else the whole slide.
        if pix is not None:
            pix = np.asarray(pix, dtype=int)
            pix = pix[(pix >= 0) & (pix < ds.n_pixels)]
            if pix.size == 0:
                continue
            n_avail = pix.size
        else:
            n_avail = ds.n_pixels
        if n_avail == 0:
            continue
        n_loaded += 1
        # Pick the rows to KEEP *before* extraction (region ∩ random cap), so a million-pixel
        # slide never pays O(n_pixels × T) extraction + a float64 copy just to keep
        # per_sample_cap rows. Same rng.choice population/order as the old post-extraction cap,
        # so the embedding is bit-identical for normal data (differs only in rng timing for a
        # rare wholly-empty sample, which is skipped below).
        if per_sample_cap and n_avail > int(per_sample_cap):
            sel = np.sort(rng.choice(n_avail, size=int(per_sample_cap), replace=False))
            rows = pix[sel] if pix is not None else sel
        else:
            rows = pix if pix is not None else np.arange(n_avail)
        if cohort_norm:
            # defer normalization: keep RAW counts + per-pixel TIC, normalize after the loop
            # once the cohort-wide TIC anchor is known.
            raw = _matrix_rows(ds, targets, rows, tol_ppm, "none", log1p=False)
            if raw.shape[0] == 0 or not np.isfinite(raw).any() or float(np.nansum(raw)) == 0.0:
                dropped.append((getattr(s, "name", None) or f"sample {si}",
                                "no signal on the feature axis"))
                continue
            tic = _row_tic(ds, rows, raw)
            pos = tic[tic > 0]
            gsum += float(pos.sum()); gcnt += int(pos.size)
            n_keep = raw.shape[0]
            pend_raw.append(raw); pend_tic.append(tic)
            block_slot.append(len(blocks))
            blocks.append(None)                  # placeholder; filled once the anchor is known
            if raw_blocks is not None:
                raw_blocks.append(None)
        else:
            X = _matrix_rows(ds, targets, rows, tol_ppm, norm, log1p)
            if X.shape[0] == 0 or not np.isfinite(X).any() or float(np.nansum(X)) == 0.0:
                dropped.append((getattr(s, "name", None) or f"sample {si}",
                                "no signal on the feature axis"))
                continue
            if raw_blocks is not None:           # capture the values fed to the embedding,
                raw_blocks.append(X.copy())      # before any per-sample standardization
            if standardize_per_sample:
                X = _zscore_cols(X)
            n_keep = X.shape[0]
            blocks.append(X)
        name = getattr(s, "name", None) or f"sample {si}"
        if name in counts:                      # disambiguate duplicate display names
            name = f"{name} #{sum(n == getattr(s, 'name', name) for n in names) + 1}"
        gid = len(names)
        names.append(name)
        counts[name] = int(n_keep)
        ids.append(np.full(n_keep, gid, dtype=int))
        groups.append(np.full(n_keep, str(getattr(s, "group", "") or ""), dtype=object))
        batches.append(np.full(n_keep, str(getattr(s, "batch", "") or ""), dtype=object))
        if keep_geometry:
            # ``rows`` are absolute pixel ids; map each to its (row, col) on the display grid
            # so the embedding's clusters can be scattered back onto this slide. Degrades
            # gracefully for test doubles that lack the orientation-aware accessor.
            rc = getattr(ds, "_pixel_rows_cols", None)
            if rc is not None:
                _rr, _cc = rc()
                rr, cc = np.asarray(_rr, dtype=int), np.asarray(_cc, dtype=int)
                idx = np.asarray(rows, dtype=int)
                px_rows.append(rr[idx])
                px_cols.append(cc[idx])
                # full tissue mask (every candidate pixel, before the cap) for back-map fill
                full = pix if pix is not None else np.arange(n_avail, dtype=int)
                tissue_rc[name] = (rr[full], cc[full])
            else:
                px_rows.append(np.full(n_keep, -1, dtype=int))
                px_cols.append(np.full(n_keep, -1, dtype=int))
            shapes[name] = (int(getattr(ds, "height", 0)), int(getattr(ds, "width", 0)))
    if not blocks:
        # Distinguish the two failure modes so the message points at the real fix: nothing
        # loaded (missing/unreadable files on this machine) vs. slides loaded but every feature
        # block was all-zero — the target m/z never fell within tol of any slide's peaks, the
        # classic "feature set from a different dataset / wrong polarity / tol too tight" case.
        if not samples:
            raise ValueError("pooled_embedding got no samples to pool.")
        if n_loaded == 0:
            raise ValueError(
                f"None of the {len(samples)} samples could be loaded — the dataset files may be "
                "missing or unreadable (check the sample paths on this machine).")
        raise ValueError(
            f"{n_loaded} of {len(samples)} samples loaded, but none had any signal on the "
            f"{targets.size}-feature axis: no target m/z fell within {tol_ppm:g} ppm of these "
            "slides' peaks. The feature set is likely from a different dataset (or the slides "
            "differ in polarity/calibration) — use a consensus axis built from these slides, or "
            "widen the m/z tolerance.")
    if cohort_norm:
        # one shared anchor = MEAN TIC over every pooled pixel (mean, not median: the median
        # recipe is a withheld patented method). Normalize each block's raw counts by its own
        # per-pixel TIC but scaled to that shared reference, so slides become comparable while
        # genuine cross-slide abundance differences survive.
        anchor = (gsum / gcnt) if gcnt > 0 else 1.0
        for k, slot in enumerate(block_slot):
            raw, tic = pend_raw[k], pend_tic[k]
            safe = np.where(tic > 0, tic, anchor)          # empty-TIC pixels → no boost
            C = raw * (anchor / safe)[:, None]
            if log1p:
                C = np.log1p(np.clip(C, 0, None))
            blocks[slot] = C
            if raw_blocks is not None:
                raw_blocks[slot] = C.copy()
    Xall = np.vstack(blocks)
    if batch_correct:
        # Opt-in pixel-level ComBat across acquisition batches, applied to the pooled feature
        # matrix before embedding; the biological group contrast is protected by a drop-first
        # one-hot covariate design. Pair with standardize_per_sample=False so ComBat — not the
        # per-sample z-score — does the cross-sample harmonization. No-op with <2 batches.
        batch_row = np.concatenate(batches) if batches else None
        if batch_row is not None and len(np.unique(batch_row)) >= 2:
            from . import batchfx
            cov = None
            if batch_protect_group:
                grp_row = np.concatenate(groups)
                levels = sorted(set(grp_row.tolist()))
                if len(levels) >= 2:
                    cov = np.column_stack([(grp_row == g).astype(float) for g in levels[1:]])
            Xall = batchfx.combat(Xall, batch_row, covariates=cov, mean_only=batch_mean_only)
    if not standardize_per_sample:           # otherwise each block is already standardized
        Xall = StandardScaler().fit_transform(Xall)
    coords, used = _reduce_2d(Xall, method, random_state, metric=metric,
                              n_neighbors=n_neighbors, min_dist=min_dist)
    if progress is not None:
        progress(total, total)
    features = np.vstack(raw_blocks) if raw_blocks else None
    prow = np.concatenate(px_rows) if px_rows else None
    pcol = np.concatenate(px_cols) if px_cols else None
    if dropped:
        # Surface partial drops UNCONDITIONALLY — not only when every sample failed. A sample
        # that loaded but had no in-tolerance signal was previously dropped silently, so the
        # returned cohort/group n was wrong with no diagnostic.
        warnings.warn(
            f"{len(dropped)} of {len(samples)} samples were dropped from the embedding "
            f"(no signal on the feature axis): {', '.join(n for n, _ in dropped)}. The cloud "
            f"shows {len(names)} sample(s).", RuntimeWarning, stacklevel=2)
    return PooledEmbedding(used.upper(), coords, np.concatenate(ids), np.concatenate(groups),
                           names, targets, counts, features=features,
                           px_row=prow, px_col=pcol, shapes=shapes, tissue_rc=tissue_rc,
                           dropped=dropped)


def cluster_points(coords, *, method: str = "kmeans", k: int = 8,
                   random_state: int = 0) -> np.ndarray:
    """Unsupervised clusters over a 2-D embedding → ``(N,)`` int labels (one per point).

    This is the "molecular histology" colouring of the lipid-atlas figures: pixels that
    land together in the UMAP/t-SNE plane are one cluster, so each colour island is a
    coherent molecular phenotype that (painted back onto tissue) reads like a segmentation.

    Clustering is done **in the embedding plane** — not the raw feature space — on purpose:
    it makes every colour island align with a visible blob (the way those panels read) and
    keeps the spatial back-map coherent with the cloud. (Caveat, by design: UMAP distances
    are not a faithful metric, so the clusters are a *visual* grouping, not a statistical
    one — for inference use the per-region tests, not these labels.)

    ``method`` is ``'kmeans'`` (a fixed ``k``, always returns ``k`` clusters) or
    ``'hdbscan'`` (density-based, picks its own count and marks sparse points ``-1`` =
    noise) when the optional ``hdbscan`` package is installed; an unavailable HDBSCAN
    falls back to k-means. Refs: Lloyd (1982), doi:10.1109/TIT.1982.1056489; HDBSCAN —
    Campello, Moulavi & Sander (2013), doi:10.1007/978-3-642-37456-2_14.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    method = (method or "kmeans").lower()
    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore

            mcs = int(max(10, n // 200))         # scale the min cluster size with the cloud
            labels = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(coords)
            return np.asarray(labels, dtype=int)
        except ImportError:
            method = "kmeans"                    # graceful fallback (caller may warn)
    from sklearn.cluster import KMeans, MiniBatchKMeans

    k = int(max(1, min(int(k), n)))
    # MiniBatch keeps a big per-pixel cloud (10⁵–10⁶ points) tractable; full k-means is
    # crisper for the small region/sample-mean clouds.
    km = (MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=2048)
          if n > 20000 else
          KMeans(n_clusters=k, random_state=random_state, n_init=10))
    return np.asarray(km.fit_predict(coords), dtype=int)


@dataclass
class RegionEmbedding:
    """Result of :func:`pooled_region_embedding` — a 2-D embedding with **one point per
    region** (or per spatially-connected region *instance*) rather than per pixel, pooled
    across many samples on a shared feature axis. This is the kidney atlas' FTU-instance-
    level view (Farrow et al. 2025, Fig 4B/D): each point is a tissue structure's mean
    spectrum, so the cloud can be coloured by *which* structure (region) it is — bringing
    region-related chemical variation to the foreground — or by which sample it came from."""
    method: str                  # 'UMAP' / 'TSNE' (the method actually used)
    coords: np.ndarray           # (N, 2) — 2-D position of each region/instance point
    region: np.ndarray           # (N,) region label per point (the subunit name)
    sample_id: np.ndarray        # (N,) int index into ``sample_names`` per point
    group: np.ndarray            # (N,) cohort group label per point ('' = ungrouped)
    sample_names: list           # display name per sample id
    sizes: np.ndarray            # (N,) pixel count behind each mean
    targets: np.ndarray          # shared m/z axis the embedding was built on
    unit: str                    # 'region' (one point per region) | 'instance'
    counts: dict                 # sample name -> number of points contributed
    features: np.ndarray = None  # (N, n_targets) raw per-region mean vectors (norm applied,
                                 # pre per-sample-centering) — the region×m/z intensity matrix
                                 # behind the cloud, row-aligned with region/sample_id/sizes;
                                 # None if not retained
    log1p: bool = False          # whether ``features`` holds log1p(intensity) values, so an
                                 # exporter can ``expm1`` back to linear intensity


def _region_instances(ds, idx):
    """Split a region's pixel-index array into spatially **connected** components (8-
    connectivity) — each blob is one *instance* of the region (e.g. a single nerve
    fascicle), the unit the kidney atlas embeds as an FTU instance. Returns a list of
    pixel-index arrays. Falls back to the whole region as one instance when the grid
    geometry isn't available."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []
    try:
        from scipy.ndimage import label

        rows, cols = ds._pixel_rows_cols()
        img = np.zeros((ds.height, ds.width), dtype=bool)
        r, c = rows[idx], cols[idx]
        img[r, c] = True
        lab, n = label(img, structure=np.ones((3, 3), dtype=int))   # 8-connectivity
        comp = lab[r, c]                                             # 1..n per region pixel
        return [idx[comp == k] for k in range(1, n + 1)]
    except Exception:  # noqa: BLE001 — geometry unavailable → treat as one instance
        return [idx]


def pooled_region_embedding(samples, targets, loader, regions_of, *, unit: str = "region",
                            method: str = "umap", tol_ppm: float = 20.0, norm: str = "tic",
                            log1p: bool = True, standardize_per_sample: bool = True,
                            min_pixels: int = 3, random_state: int = 0, progress=None,
                            metric: str = "euclidean", n_neighbors: int = 15,
                            min_dist: float = 0.1) -> RegionEmbedding:
    """Embed **one point per region** (``unit='region'``) or **per connected region
    instance** (``unit='instance'``) pooled across many samples — the FTU-instance-level
    analogue of :func:`pooled_embedding` (which embeds pixels).

    ``loader(sample)`` returns ``(ds, _pix)`` as in :func:`pooled_embedding` (the pixel
    index is ignored here — region masks come from ``regions_of``); returning ``None``
    skips the sample. ``regions_of(sample)`` must return a list of ``(region_name,
    pixel_index_array)`` for that sample's slide. For each region a mean feature vector is
    built on the shared ``targets`` (optionally ``log1p``); with ``unit='instance'`` the
    region is first split into 8-connected blobs and each contributes its own mean. Means
    over fewer than ``min_pixels`` pixels are dropped. When ``standardize_per_sample`` each
    sample's point block is mean-centred (a light per-slide offset removal — the batch-
    alignment analogue), then the full stack is z-scored before :func:`_reduce_2d`.

    Each point carries its region label, sample, group and pixel count, so the cloud can
    be coloured by structure or by sample. ``loader``/``regions_of`` are injected (not
    hard-wired) so the pooling logic stays unit-testable without a real dataset. Refs as
    :func:`embedding`; FTU-instance-level embedding — Farrow et al. (2025), Sci. Adv. 11,
    eadu3730, Fig 4B/D."""
    from sklearn.preprocessing import StandardScaler

    targets = np.asarray(list(targets), dtype=float)
    if targets.size == 0:
        raise ValueError("pooled_region_embedding needs a non-empty shared feature axis (targets).")
    unit = (unit or "region").lower()
    samples = list(samples)
    total = len(samples) + 1
    blocks, regions, ids, groups, sizes, names, counts = [], [], [], [], [], [], {}
    raw_blocks = []                                # uncentred region means → export matrix
    n_loaded = 0                                   # samples whose dataset actually loaded
    for si, s in enumerate(samples):
        if progress is not None:
            progress(si, total)
        loaded = loader(s)
        if not loaded:
            continue
        ds = loaded[0] if isinstance(loaded, tuple) else loaded
        if ds is None:
            continue
        n_loaded += 1
        regs = regions_of(s) or []
        if not regs:
            continue
        # Extract features for ONLY the pixels these regions cover — not the whole slide — so a
        # region-mean run touches a fraction of a big slide (with a lazy loader it never even
        # loads the full cube). Build a pixel-index → local-row map to average per region.
        npx = ds.n_pixels
        clipped = [np.asarray(ridx, dtype=int)[(np.asarray(ridx, dtype=int) >= 0)
                                               & (np.asarray(ridx, dtype=int) < npx)]
                   for _rn, ridx in regs]
        union = np.unique(np.concatenate(clipped)) if any(c.size for c in clipped) else np.empty(0, int)
        if union.size == 0:
            continue
        Xu = _matrix_rows(ds, targets, union, tol_ppm, norm, log1p)   # region pixels only
        row_of = {int(p): k for k, p in enumerate(union)}
        vecs, vregions, vsizes = [], [], []
        for rname, ridx in regs:
            ridx = np.asarray(ridx, dtype=int)
            ridx = ridx[(ridx >= 0) & (ridx < npx)]
            if ridx.size == 0:
                continue
            members = [ridx] if unit == "region" else _region_instances(ds, ridx)
            for u in members:
                if u.size < int(min_pixels):
                    continue
                local = np.fromiter((row_of[int(p)] for p in u), dtype=int, count=u.size)
                v = Xu[local].mean(axis=0)
                if not np.isfinite(v).any() or float(np.nansum(v)) == 0.0:
                    continue
                vecs.append(v)
                vregions.append(str(rname))
                vsizes.append(int(u.size))
        if not vecs:
            continue
        raw = np.vstack(vecs)
        raw_blocks.append(raw)                                    # keep the uncentred means
        block = raw
        if standardize_per_sample and block.shape[0] >= 2:
            block = block - block.mean(axis=0, keepdims=True)     # per-slide offset removal
                                                                  # (returns a new array; raw intact)
        name = getattr(s, "name", None) or f"sample {si}"
        if name in counts:                          # disambiguate duplicate display names
            name = f"{name} #{sum(n == getattr(s, 'name', name) for n in names) + 1}"
        gid = len(names)
        names.append(name)
        counts[name] = len(vecs)
        blocks.append(block)
        regions.extend(vregions)
        sizes.extend(vsizes)
        ids.append(np.full(len(vecs), gid, dtype=int))
        groups.append(np.full(len(vecs), str(getattr(s, "group", "") or ""), dtype=object))
    if not blocks:
        if n_loaded == 0 and samples:
            raise ValueError(
                f"None of the {len(samples)} samples could be loaded — the dataset files may be "
                "missing or unreadable (check the sample paths on this machine).")
        raise ValueError("No samples produced any region means on the shared feature axis "
                         "— the slides need saved named regions (Segmentation tab), and the "
                         "target m/z must fall within tolerance of those regions' peaks.")
    Xall = np.vstack(blocks)
    if Xall.shape[0] < 2:
        raise ValueError(f"Need at least two region points to embed; got {Xall.shape[0]}.")
    Xall = StandardScaler().fit_transform(Xall)
    coords, used = _reduce_2d(Xall, method, random_state, metric=metric,
                              n_neighbors=n_neighbors, min_dist=min_dist)
    if progress is not None:
        progress(total, total)
    return RegionEmbedding(used.upper(), coords, np.asarray(regions, dtype=object),
                           np.concatenate(ids), np.concatenate(groups), names,
                           np.asarray(sizes, dtype=int), targets, unit, counts,
                           features=np.vstack(raw_blocks), log1p=bool(log1p))


_REDUCE_CACHE: "OrderedDict" = OrderedDict()   # in-process memo of 2-D embeddings (re-run = instant)
_REDUCE_CACHE_MAX = 8
_REDUCE_CACHE_LOCK = threading.Lock()          # embedding runs on worker threads — guard the LRU


def _reduce_2d(X, method, random_state, *, metric="euclidean", n_neighbors=15, min_dist=0.1,
               deterministic=True):
    """Thread-safe in-process cache around :func:`_reduce_2d_compute` keyed by the exact input
    matrix + every knob, so re-running an embedding with identical inputs (re-click Run on the
    Components tab with the same slide/peaks/settings) returns instantly instead of recomputing
    a multi-second UMAP/t-SNE. Coords are copied out so a caller can't mutate the cached array."""
    X = np.asarray(X, dtype=float)
    try:
        key = (hashlib.sha1(np.ascontiguousarray(X)).hexdigest(), X.shape, method,
               random_state, metric, int(n_neighbors), float(min_dist), bool(deterministic))
    except Exception:                          # pragma: no cover - unhashable X → just don't cache
        key = None
    if key is not None:
        with _REDUCE_CACHE_LOCK:
            hit = _REDUCE_CACHE.get(key)
            if hit is not None:
                _REDUCE_CACHE.move_to_end(key)
        if hit is not None:
            return hit[0].copy(), hit[1]
    coords, used = _reduce_2d_compute(X, method, random_state, metric=metric,
                                      n_neighbors=n_neighbors, min_dist=min_dist,
                                      deterministic=deterministic)
    if key is not None:
        with _REDUCE_CACHE_LOCK:
            _REDUCE_CACHE[key] = (coords, used)
            _REDUCE_CACHE.move_to_end(key)
            while len(_REDUCE_CACHE) > _REDUCE_CACHE_MAX:
                _REDUCE_CACHE.popitem(last=False)
    return coords.copy(), used


def _reduce_2d_compute(X, method, random_state, *, metric="euclidean", n_neighbors=15, min_dist=0.1,
                       deterministic=True):
    """Return ``(coords, method_used)``. Falls back to t-SNE if UMAP is requested
    but ``umap-learn`` isn't installed — and **warns**, so a silent method swap can't
    masquerade as the requested embedding.

    ``metric`` / ``n_neighbors`` / ``min_dist`` expose the UMAP knobs the embedding's
    *look* depends on (the kidney atlas, Farrow et al. 2025, uses ``metric='cosine',
    n_neighbors=250, min_dist=0`` over millions of pixels). ``metric`` is also forwarded to
    the t-SNE fallback (which has no ``n_neighbors``/``min_dist``). ``n_neighbors`` is
    clamped below the sample count so a small cloud can't error.

    ``deterministic`` (default True) keeps the reproducible, byte-identical-to-before UMAP
    call — a fixed ``random_state``, which umap-learn honours by pinning to a single thread.
    Pass ``deterministic=False`` for *interactive* embeddings to trade exact reproducibility
    for ~3-5x: the seed is dropped so umap runs multicore (``n_jobs=-1``), and a wide input
    (e.g. a pooled-cohort feature union) is PCA-pre-reduced to 50 dims first, where UMAP's
    dominant k-NN graph is far cheaper. The result is statistically equivalent, not numerically
    identical (cluster colours can shift run to run), so keep it on the export/repro path."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 5:
        # Too few points for a meaningful (or numerically stable — Barnes-Hut t-SNE can
        # even segfault) manifold. Lay the handful of points out with PCA instead, and
        # report 'pca' so the method label never masquerades as the one requested.
        from sklearn.decomposition import PCA

        k = min(2, n, X.shape[1])
        coords = (PCA(n_components=k, random_state=random_state).fit_transform(X)
                  if k >= 1 else np.zeros((n, 1)))
        if coords.shape[1] < 2:
            coords = np.column_stack([coords, np.zeros(n)])
        return coords, "pca"
    if method == "umap":
        import warnings
        try:
            import umap  # type: ignore
        except ImportError:
            warnings.warn("umap-learn not installed; falling back to t-SNE. "
                          "Install the [embed] extra for UMAP.", RuntimeWarning, stacklevel=2)
            method = "tsne"
        else:
            try:
                nn = int(max(2, min(int(n_neighbors), n - 1)))   # UMAP needs 2 ≤ nn < n
                if deterministic:
                    # Unchanged from before — seeded; umap-learn pins n_jobs=1 internally. Do
                    # NOT add an n_jobs kwarg here: an older umap that lacks it would raise and
                    # silently drop the *default* path to t-SNE.
                    return umap.UMAP(n_components=2, random_state=random_state, metric=metric,
                                     n_neighbors=nn, min_dist=float(min_dist)).fit_transform(X), "umap"
                Xu = X
                # Clamp components to the data (a small ROI can have n_samples < 50);
                # an unclamped PCA(50) would raise and the broad except below would
                # silently swap UMAP for t-SNE with a misleading "UMAP failed" warning.
                n_comp = min(50, X.shape[0] - 1, X.shape[1])
                if X.shape[1] > 50 and n_comp >= 2:              # PCA-pre-reduce wide input
                    from sklearn.decomposition import PCA
                    Xu = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
                return umap.UMAP(n_components=2, random_state=None, metric=metric, n_neighbors=nn,
                                 min_dist=float(min_dist), n_jobs=-1).fit_transform(Xu), "umap"
            except Exception as e:  # noqa: BLE001 - real UMAP failure: be honest, don't blame "not installed"
                warnings.warn(f"UMAP failed ({e}); falling back to t-SNE.",
                              RuntimeWarning, stacklevel=2)
                method = "tsne"
    from sklearn.manifold import TSNE

    perplexity = float(min(30, max(5, X.shape[0] // 4)))
    # t-SNE requires perplexity < n_samples; clamp **down** so a small point set (e.g. a
    # region-mean embedding of a few structures) can't error. Only bites for tiny N — for
    # the pixel paths (N≥16) (N-1)/3 ≥ 5 so the value is unchanged.
    perplexity = min(perplexity, max(1.0, (X.shape[0] - 1) / 3.0))
    # PCA init requires a euclidean space; for a non-euclidean metric (e.g. cosine) use the
    # 'random' init so sklearn doesn't error on the mismatch.
    init = "pca" if metric == "euclidean" else "random"
    # Prefer openTSNE (FIt-SNE, multicore — 5-30x over sklearn TSNE at scale) when installed;
    # fall back to sklearn TSNE on absence or any error. Same 2-D output and 'tsne' label.
    try:
        from openTSNE import TSNE as _OpenTSNE
    except ImportError:
        _OpenTSNE = None
    if _OpenTSNE is not None:
        try:
            emb = _OpenTSNE(n_components=2, perplexity=perplexity, metric=metric,
                            random_state=random_state, initialization=init,
                            n_jobs=(1 if deterministic else -1)).fit(X)
            return np.asarray(emb, dtype=float), "tsne"
        except Exception:  # noqa: BLE001 — any openTSNE issue: fall back to sklearn, never fail the embed
            pass
    return TSNE(n_components=2, random_state=random_state, perplexity=perplexity,
                metric=metric, init=init).fit_transform(X), "tsne"


def _hsv_to_rgb(h, s, v):
    """Vectorised HSV→RGB (all inputs (N,) in 0..1) → (N, 3) RGB in 0..1. Kept dependency-
    free (no matplotlib) so the engine stays importable without the GUI stack."""
    h = np.asarray(h, dtype=float)
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def _coords_rgb(coords):
    """Map 2-D embedding coords → (N, 3) float RGB in 0..1 via a **2-D colour wheel**: a
    point's *direction* from the cloud centre sets its hue (the full spectrum, so every
    region of the embedding — greens included — is reachable) and its *distance* sets
    saturation (points near the centre fade toward grey). Shared by the tissue map and the
    linked scatter, so a point and its pixel always share a colour and a given hue means
    the same thing in both views.

    This replaces an earlier ``x→red, y→green, complement→blue`` mapping whose complementary
    blue channel could never reach a strong green and so hid green points in the scatter
    even while they showed clearly (as coherent regions) on the tissue map."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return np.zeros((0, 3))

    def center_scale(v):                          # robust centre + spread (outlier-proof)
        c = np.median(v)
        s = float(np.percentile(np.abs(v - c), 90)) or 1.0
        return (v - c) / s

    x = center_scale(coords[:, 0])
    y = center_scale(coords[:, 1])
    hue = (np.arctan2(y, x) / (2.0 * np.pi)) % 1.0
    rad = np.sqrt(x * x + y * y)
    hi = float(np.percentile(rad, 95)) or 1.0
    sat = 0.25 + 0.75 * np.clip(rad / hi, 0.0, 1.0)   # centre keeps a little colour, edges saturate
    return _hsv_to_rgb(hue, sat, np.ones_like(hue))


def _embedding_rgba(ds, coords, rows, cols):
    rgb = _coords_rgb(coords)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rgba = np.zeros((ds.height, ds.width, 4), dtype=np.ubyte)
    rgba[rows, cols, 0] = (r * 255).astype(np.ubyte)
    rgba[rows, cols, 1] = (g * 255).astype(np.ubyte)
    rgba[rows, cols, 2] = (b * 255).astype(np.ubyte)
    rgba[rows, cols, 3] = 255
    return rgba


def _shift2d(a, dy, dx):
    """Shift a 2-D (or H×W×C) array by (dy, dx) with zero fill (no wrap-around)."""
    out = np.zeros_like(a)
    ys, yd = (slice(max(0, -dy), a.shape[0] - max(0, dy)),
              slice(max(0, dy), a.shape[0] - max(0, -dy)))
    xs, xd = (slice(max(0, -dx), a.shape[1] - max(0, dx)),
              slice(max(0, dx), a.shape[1] - max(0, -dx)))
    out[yd, xd] = a[ys, xs]
    return out


def _spatial_smooth_scores(ds, scores, sigma, weights):
    """Spatially smooth PCA scores over the tissue grid (Cardinal-style spatial
    weights applied to the projected features):

    * ``gaussian`` — isotropic distance weights (spatially-aware, SA).
    * ``adaptive`` — bilateral: distance × spectral-similarity weights, so smoothing
      is suppressed across morphological edges (spatially-aware structurally-adaptive,
      SASA — the edge-preserving variant).

    Only acquired pixels contribute (off-tissue cells are excluded from every window).
    """
    rows, cols = ds._pixel_rows_cols()
    H, W, C = ds.height, ds.width, scores.shape[1]
    cube = np.zeros((H, W, C))
    cube[rows, cols, :] = scores
    valid = np.zeros((H, W))
    valid[rows, cols] = 1.0
    if sigma <= 0:
        return scores
    if weights == "gaussian":
        from scipy.ndimage import gaussian_filter
        wn = gaussian_filter(valid, sigma)                 # renormalize for missing pixels
        out = np.zeros_like(cube)
        for c in range(C):
            sm = gaussian_filter(cube[:, :, c], sigma)
            out[:, :, c] = np.where(wn > 0, sm / np.where(wn > 0, wn, 1.0), 0.0)
        return out[rows, cols, :]
    # bilateral / adaptive: distance × range (spectral) weighting
    r = max(1, int(round(2 * sigma)))
    sr = float(np.median(np.std(scores, axis=0)) * np.sqrt(C)) + 1e-9
    acc = np.zeros((H, W, C))
    wsum = np.zeros((H, W))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sw = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            nb = _shift2d(cube, dy, dx)
            nv = _shift2d(valid, dy, dx)
            diff2 = ((cube - nb) ** 2).sum(axis=2)
            w = sw * np.exp(-diff2 / (2.0 * sr * sr)) * nv * valid
            acc += nb * w[:, :, None]
            wsum += w
    out = np.where(wsum[:, :, None] > 0, acc / np.where(wsum[:, :, None] > 0, wsum[:, :, None], 1.0), 0.0)
    return out[rows, cols, :]


def spatial_segment(ds, peaks, n_clusters=6, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                    spatial_sigma: float = 1.0, n_components: int = 10, log1p: bool = True,
                    random_state: int = 0, k_range=range(2, 9),
                    weights: str = "gaussian", mask=None) -> spatial.Segmentation:
    """Spatially-aware segmentation: PCA-project the features, smooth the scores over
    each pixel's neighbourhood (radius ``spatial_sigma``), then k-means — coherent
    regions instead of speckle. ``weights`` selects the spatial kernel, matching
    Cardinal's spatial methods:

    * ``"gaussian"`` — distance-only smoothing (spatially-aware, SA).
    * ``"adaptive"`` — bilateral distance × spectral-similarity smoothing, which
      preserves morphological edges (spatially-aware structurally-adaptive, SASA).

    Pass ``n_clusters=None`` to auto-pick k by silhouette.

    Refs: spatially-aware MSI segmentation — Alexandrov & Kobarg (2011),
    doi:10.1093/bioinformatics/btr246; spatial weights (gaussian/adaptive) —
    Bemis et al. (2016), Mol. Cell. Proteomics 15(5):1761, doi:10.1074/mcp.O115.053918.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xfull = _matrix(ds, peaks, tol_ppm, norm, log1p)
    m = None if mask is None else np.asarray(mask, bool)
    scaler = StandardScaler().fit(Xfull if m is None else Xfull[m])
    X = scaler.transform(Xfull)
    fit_rows = X if m is None else X[m]
    n_comp = int(min(n_components, X.shape[1], max(1, fit_rows.shape[0] - 1)))
    pca = PCA(n_components=n_comp, random_state=random_state).fit(fit_rows)
    scores = pca.transform(X)
    scores = _spatial_smooth_scores(ds, scores, spatial_sigma, weights)   # smooth over the full grid
    sub = scores if m is None else scores[m]                              # cluster the region's pixels
    if n_clusters is None:
        k, labels, sil = spatial.choose_k(sub, k_range, random_state)
    else:
        k = int(min(n_clusters, sub.shape[0]))
        labels = spatial._cluster(sub, k, random_state)
        sil = spatial._silhouette(sub, labels)
    return spatial._seg_result(ds, peaks, labels, m, k,
                               float(pca.explained_variance_ratio_.sum()), sil)


# --------------------------------------------------------------------------- #
# spatialDGMM — per-ion spatially-aware Gaussian mixture segmentation
# --------------------------------------------------------------------------- #
def spatial_dgmm(ds, mz, k: int = 3, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                 sigma: float = 1.5, beta: float = 1.0, n_iter: int = 30,
                 anneal: bool = True, random_state: int = 0):
    """Spatially-aware Gaussian-mixture segmentation of a **single ion image**.

    A pragmatic spatialDGMM: an EM mixture on the ion's per-pixel intensities where each
    pixel's class *prior* is the Gaussian-smoothed neighbourhood class distribution (the
    spatial / Dirichlet term, strength ``beta``), optionally annealed from soft to hard.
    Components are returned ordered low→high intensity, so label ``k-1`` is the "hot"
    component — segmenting one ion into background / dim / bright zones that respect
    spatial structure (the basis for :func:`segmentation_test`).

    Returns ``(labels, means)`` — per-pixel component label and the sorted component means.

    Refs: spatially-aware mixture segmentation — Bemis et al. (2019), Bioinformatics
    35(14):i208, doi:10.1093/bioinformatics/btz219; spatially-variant finite mixtures —
    Sanjay-Gopal & Hebert (1998), doi:10.1109/83.704309.
    """
    x = np.asarray(ds.ion_vector(mz, tol_ppm=tol_ppm, norm=norm), float)
    n = len(x)
    k = int(max(2, min(k, n)))
    if x.std() <= 0:                                   # flat ion -> single class
        return np.zeros(n, dtype=int), np.array([float(x.mean())])
    mu = np.quantile(x, np.linspace(0.1, 0.9, k))
    var = np.full(k, max(float(np.var(x)), 1e-6))
    R = np.zeros((n, k))
    R[np.arange(n), np.argmin(np.abs(x[:, None] - mu[None, :]), axis=1)] = 1.0
    for it in range(n_iter):
        like = np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]) \
            / np.sqrt(2 * np.pi * var[None, :])
        prior = _spatial_smooth_scores(ds, R, sigma, "gaussian") if sigma > 0 else R
        prior = np.clip(prior, 1e-9, None)
        post = like * prior ** beta
        if anneal:                                     # cool temperature 2 -> 1
            T = 1.0 + 1.0 * (1.0 - it / max(1, n_iter - 1))
            post = post ** (1.0 / T)
        Z = post.sum(1, keepdims=True)
        Z[Z <= 0] = 1.0
        R = post / Z
        Nk = R.sum(0) + 1e-9
        mu = (R * x[:, None]).sum(0) / Nk
        var = (R * (x[:, None] - mu[None, :]) ** 2).sum(0) / Nk + 1e-6
    order = np.argsort(mu)
    labels = np.argmax(R[:, order], axis=1)
    return labels.astype(int), mu[order]


def segmentation_test(ds, peaks, groups, samples, k: int = 3, tol_ppm: float = DEFAULT_TOL_PPM,
                      norm: str = "tic", summary: str = "mean_top", **dgmm_kw):
    """Differential abundance per ion via spatialDGMM + an across-sample test
    (Cardinal's ``segmentationTest`` = spatialDGMM then ``meansTest``).

    For each peak: segment its image with :func:`spatial_dgmm`, summarize **each tissue
    sample** by its hot-component signal — the mean intensity of the top component
    (``summary='mean_top'``) or the fraction of the sample's pixels in it
    (``summary='frac_top'``) — then Mann-Whitney across the two ``groups`` at the
    **sample** level (so spatially-correlated pixels don't masquerade as replicates).

    ``groups`` and ``samples`` are per-pixel arrays (group label and replicate id; use
    −1 in ``samples`` to drop pixels). Returns a DataFrame ``mz, statistic (AUC),
    p_value, q_value, n_a, n_b`` sorted by ascending p. ``df.attrs['groups']`` records
    the two group labels (AUC > 0.5 ⇒ higher in the second).
    """
    import pandas as pd

    groups = np.asarray(groups)
    samples = np.asarray(samples, dtype=int)
    peaks = np.asarray(peaks, dtype=float)
    glabels = [g for g in np.unique(groups) if str(g) != "-1"]
    if len(glabels) != 2:
        raise ValueError("segmentation_test needs exactly two groups")
    ga, gb = glabels
    # map each sample to its group (majority group of the sample's pixels)
    samp_group = {}
    for sid in np.unique(samples):
        if sid < 0:
            continue
        gs = groups[samples == sid]
        gs = gs[gs != -1] if gs.dtype != object else gs
        if len(gs):
            vals, cnts = np.unique(gs, return_counts=True)
            samp_group[int(sid)] = vals[int(np.argmax(cnts))]
    a_ids = [s for s, g in samp_group.items() if g == ga]
    b_ids = [s for s, g in samp_group.items() if g == gb]

    rows = []
    for mz in peaks:
        x = np.asarray(ds.ion_vector(mz, tol_ppm=tol_ppm, norm=norm), float)
        labels, means = spatial_dgmm(ds, mz, k=k, tol_ppm=tol_ppm, norm=norm, **dgmm_kw)
        top = labels.max()

        def summ(ids):
            out = []
            for sid in ids:
                sel = samples == sid
                if not sel.any():
                    continue
                if summary == "frac_top":
                    out.append(float((labels[sel] == top).mean()))
                else:
                    hot = sel & (labels == top)
                    out.append(float(x[hot].mean()) if hot.any() else 0.0)
            return np.asarray(out, float)

        sa, sb = summ(a_ids), summ(b_ids)
        if len(sa) and len(sb):
            auc, p = spatial._auc_mwu(sa.reshape(-1, 1), sb.reshape(-1, 1))
            rows.append((float(mz), float(auc[0]), float(p[0]), len(sa), len(sb)))
        else:
            rows.append((float(mz), np.nan, 1.0, len(sa), len(sb)))
    df = pd.DataFrame(rows, columns=["mz", "statistic", "p_value", "n_a", "n_b"])
    df["q_value"] = spatial._bh_fdr(df["p_value"].to_numpy())
    df = df.sort_values("p_value").reset_index(drop=True)
    df.attrs["groups"] = (ga, gb)
    df.attrs["unit"] = "sample"
    return df


# --------------------------------------------------------------------------- #
# Supervised classification — PLS-DA / OPLS-DA + cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class Classifier:
    method: str                  # 'PLS-DA' / 'OPLS-DA'
    classes: list                # class labels in column order
    peaks: np.ndarray            # m/z per feature
    vip: np.ndarray              # (n_peaks,) Variable Importance in Projection
    scores: np.ndarray           # (n_pixels, n_components) X-scores (for plotting)
    model: object = None         # fitted PLSRegression
    scaler: object = None
    log1p: bool = True
    _ortho: list = None          # OPLS orthogonal (w, p) filters to re-apply on predict

    def _prep(self, X):
        Xs = np.log1p(np.clip(X, 0, None)) if self.log1p else np.asarray(X, float)
        Xs = self.scaler.transform(Xs)
        for w_o, p_o in (self._ortho or []):
            Xs = Xs - (Xs @ w_o) @ p_o.T
        return Xs

    def predict(self, X):
        """Predict class labels for a (pixels × peaks) feature matrix."""
        yh = self.model.predict(self._prep(X))
        return np.array([self.classes[i] for i in np.argmax(yh, axis=1)])

    def top_peaks(self, n: int = 10):
        """The ``n`` most discriminating ions by VIP score."""
        order = np.argsort(-self.vip)[:n]
        return [(float(self.peaks[j]), float(self.vip[j])) for j in order]


def _vip(pls) -> np.ndarray:
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p_, A = w.shape
    ss = (t ** 2).sum(0) * (q ** 2).sum(0)             # explained SS per component
    total = float(ss.sum())
    if total <= 0:
        return np.zeros(p_)
    wn2 = (w ** 2).sum(0)
    wn2[wn2 == 0] = 1.0
    return np.sqrt(p_ * ((w ** 2 / wn2) * ss).sum(1) / total)


def _opls_filter(X, y, n_ortho):
    """Orthogonal signal correction (Trygg & Wold OPLS) for a single y; returns the
    filtered X and the list of (w_ortho, p_ortho) to re-apply to new data."""
    X = X.copy()
    y = np.asarray(y, float).reshape(-1, 1)
    yy = float((y.T @ y).item()) or 1.0
    orths = []
    for _ in range(int(n_ortho)):
        w = (X.T @ y) / yy
        w /= (np.linalg.norm(w) or 1.0)
        t = X @ w
        p = (X.T @ t) / (float((t.T @ t).item()) or 1.0)
        w_o = p - (float((w.T @ p).item()) / (float((w.T @ w).item()) or 1.0)) * w
        no = np.linalg.norm(w_o)
        if no < 1e-12:
            break
        w_o /= no
        t_o = X @ w_o
        p_o = (X.T @ t_o) / (float((t_o.T @ t_o).item()) or 1.0)
        X = X - t_o @ p_o.T
        orths.append((w_o, p_o))
    return X, orths


def plsda(ds, peaks, labels, n_components: int = 2, tol_ppm: float = DEFAULT_TOL_PPM,
          norm: str = "tic", log1p: bool = True, orthogonal: bool = False,
          n_ortho: int = 1) -> Classifier:
    """Supervised PLS-DA (or OPLS-DA) over per-pixel class ``labels``.

    Fits PLS regression on one-hot class membership and returns a :class:`Classifier`
    with X-scores (for the discriminant scatter), **VIP** scores (which ions drive the
    separation), and ``predict``. ``orthogonal=True`` first removes ``n_ortho``
    y-orthogonal components (OPLS-DA, binary only) so the predictive variation is
    concentrated in the first component.

    Refs: PLS — Wold et al. (2001), doi:10.1016/S0169-7439(01)00155-1; OPLS — Trygg &
    Wold (2002), doi:10.1002/cem.695; VIP — Chong & Jun (2005), doi:10.1016/j.chemolab.2004.12.011.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels)
    peaks = np.asarray(peaks, float)
    Xraw = _matrix(ds, peaks, tol_ppm, norm, log1p)
    if np.issubdtype(labels.dtype, np.number) and (labels < 0).any():
        keep = labels >= 0                              # drop unassigned (region-grouping) pixels
        Xraw, labels = Xraw[keep], labels[keep]
    classes = list(np.unique(labels))
    scaler = StandardScaler().fit(Xraw)
    X = scaler.transform(Xraw)
    Y = np.zeros((len(labels), len(classes)))
    for i, c in enumerate(classes):
        Y[labels == c, i] = 1.0

    ortho = None
    method = "PLS-DA"
    if orthogonal and len(classes) == 2:
        X, ortho = _opls_filter(X, Y[:, 1] - Y[:, 0], n_ortho)
        method = "OPLS-DA"
    k = int(max(1, min(n_components, X.shape[1], len(classes), X.shape[0] - 1)))
    pls = PLSRegression(n_components=max(k, 2) if len(classes) > 2 else k).fit(X, Y)
    return Classifier(method=method, classes=classes, peaks=peaks, vip=_vip(pls),
                      scores=pls.x_scores_, model=pls, scaler=scaler, log1p=log1p, _ortho=ortho)


def cross_validate(ds, peaks, labels, samples=None, n_components: int = 2,
                   tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic", log1p: bool = True,
                   orthogonal: bool = False, n_folds: int = 5, random_state: int = 0) -> dict:
    """Cross-validated PLS-DA accuracy.

    If ``samples`` (per-pixel replicate ids) is given, folds are **whole samples**
    (leave-one-sample-out / grouped) so a tissue's pixels never split across train and
    test — the spatial-autocorrelation leakage that otherwise inflates pixel-level CV,
    exactly the rigor Cardinal's ``crossValidate(folds=run(x))`` enforces. Without
    ``samples`` it falls back to stratified pixel folds and flags the result optimistic.

    Returns ``{accuracy, per_fold, n_folds, leakage_safe, classes, confusion}``.
    """
    from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupKFold

    labels = np.asarray(labels)
    peaks = np.asarray(peaks, float)
    Xfull = spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    if np.issubdtype(labels.dtype, np.number) and (labels < 0).any():
        keep = labels >= 0                              # drop unassigned (region-grouping) pixels
        labels, Xfull = labels[keep], Xfull[keep]
        if samples is not None:
            samples = np.asarray(samples)[keep]
    classes = list(np.unique(labels))
    cls_idx = {c: i for i, c in enumerate(classes)}

    use_groups = False
    if samples is not None:
        samples = np.asarray(samples)
        keep = samples != -1
        groups = samples[keep]
        use_groups = len(np.unique(groups)) >= 2        # need ≥2 samples for leave-one-sample-out

    if use_groups:
        idx_all = np.flatnonzero(keep)
        n_groups = len(np.unique(groups))
        splitter = (LeaveOneGroupOut() if n_groups <= n_folds
                    else GroupKFold(n_splits=n_folds))
        split = splitter.split(idx_all, labels[keep], groups)
        idx_map = idx_all
        leakage_safe = True
    else:
        # No usable sample grouping — samples absent, or only one sample present (a single slide,
        # or all but one region/sample deselected). Leave-one-sample-out is impossible, so fall
        # back to stratified pixel folds and flag the estimate as leakage-prone, rather than
        # crashing on sklearn's "fewer than 2 groups".
        _, counts = np.unique(labels, return_counts=True)
        n_eff = int(max(2, min(n_folds, int(counts.min()))))
        skf = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=random_state)
        split = skf.split(np.arange(len(labels)), labels)
        idx_map = np.arange(len(labels))
        leakage_safe = False

    conf = np.zeros((len(classes), len(classes)), int)
    per_fold, correct, total = [], 0, 0
    for tr, te in split:
        tr_i, te_i = idx_map[tr], idx_map[te]
        clf = _fit_on(ds, peaks, Xfull, labels, tr_i, n_components, log1p, orthogonal,
                      tol_ppm, norm)
        pred = clf.predict(Xfull[te_i])
        truth = labels[te_i]
        acc = float(np.mean(pred == truth)) if len(te_i) else float("nan")
        per_fold.append(acc)
        correct += int(np.sum(pred == truth))
        total += len(te_i)
        for t_, p_ in zip(truth, pred):
            conf[cls_idx[t_], cls_idx[p_]] += 1
    out = {"accuracy": correct / total if total else float("nan"),
           "per_fold": per_fold, "n_folds": len(per_fold), "leakage_safe": leakage_safe,
           "classes": classes, "confusion": conf}
    if not leakage_safe:
        out["warning"] = (
            ("only one sample present — leave-one-sample-out needs ≥2 samples; used stratified "
             "pixel folds (optimistic: pixels leak spatial autocorrelation). Add a second sample "
             "for an honest estimate.") if samples is not None else
            ("pixel-level folds leak spatial autocorrelation — pass samples= for "
             "leave-one-sample-out CV (the honest estimate)"))
    return out


def _fit_on(ds, peaks, Xfull, labels, idx, n_components, log1p, orthogonal, tol_ppm, norm):
    """Fit a PLS-DA Classifier on a pixel subset using a pre-extracted feature matrix."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    classes = list(np.unique(labels))
    Xraw = Xfull[idx]
    Xraw = np.log1p(np.clip(Xraw, 0, None)) if log1p else Xraw
    scaler = StandardScaler().fit(Xraw)
    X = scaler.transform(Xraw)
    y = labels[idx]
    Y = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        Y[y == c, i] = 1.0
    ortho = None
    if orthogonal and len(classes) == 2:
        X, ortho = _opls_filter(X, Y[:, 1] - Y[:, 0], 1)
    k = int(max(1, min(n_components, X.shape[1], len(classes), X.shape[0] - 1)))
    pls = PLSRegression(n_components=max(k, 2) if len(classes) > 2 else k).fit(X, Y)
    return Classifier("PLS-DA", classes, np.asarray(peaks, float), _vip(pls),
                      pls.x_scores_, pls, scaler, log1p, ortho)
