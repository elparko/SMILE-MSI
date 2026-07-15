"""Spatial analytics over an :class:`~smile_msi.msi.MSIDataset` — the analysis
panels that run on top of ion images.

* :func:`segment` / :func:`auto_segment` — unsupervised **spatial segmentation**
  (PCA + k-means on the per-pixel peak matrix) into molecularly distinct regions;
  ``auto_segment`` picks the cluster count by silhouette score.
* :func:`discriminating_features` — for a segmentation (or any labelling), the
  top **discriminating ions per region** (one-vs-rest exact ROC AUC + Mann-Whitney
  p + BH-FDR).
* :func:`colocalize` — rank every peak by how similar its ion image is to a target
  (**co-localization** search).
* :func:`roi_comparison` — **exact** rank-based ROC AUC + p + FDR for every peak
  between two chosen pixel regions.

All functions operate on the dataset's compact, cached ``pixels x peaks`` feature
matrix (built once, streaming), so they scale to large files. Peaks are passed as
plain m/z floats (e.g. :meth:`MSIDataset.pick_peaks` output) so they compose with
lipid annotation downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_TOL_PPM


# --------------------------------------------------------------------------- #
# Shared feature matrix
# --------------------------------------------------------------------------- #
def feature_matrix(ds, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                   reduce: str = "sum") -> np.ndarray:
    """(n_pixels x n_peaks) normalized intensity matrix (cached on the dataset)."""
    ds.ensure_features(peaks, tol_ppm=tol_ppm, reduce=reduce)
    return ds.feature_matrix(norm)


def _fold_tau(X: np.ndarray) -> float:
    """Data-scaled fold pseudocount: the 5th percentile of the nonzero intensities of
    ``X``. Adding ``tau`` before the ratio means two near-noise group means give
    ``log2 FC ≈ 0`` instead of a spurious ratio (KNOWN_ISSUES.md artifact #2). The
    caller passes the population it wants regularized — whole-``X`` for slide-wide
    contrasts, on-tissue-only for ROI-vs-tissue. Empty / all-zero ``X`` → ``1.0``.
    The single source of the fold pseudocount used by every per-slide fold metric
    (audit plan 18, Issue A)."""
    nz = X[X > 0]
    return float(np.percentile(nz, 5)) if nz.size else 1.0


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
@dataclass
class Segmentation:
    labels: np.ndarray          # per-pixel cluster id (0..k-1)
    label_image: np.ndarray     # (height, width), NaN off-tissue
    peaks: list                 # m/z used as features
    n_clusters: int
    explained_variance: float   # fraction captured by the PCA components used
    silhouette: float = float("nan")

    def mask(self, cluster: int) -> np.ndarray:
        return self.labels == cluster


def _zscore_cols(X):
    """Per-feature z-score (mean 0, unit variance); zero-variance columns collapse to 0.
    Applied per sample before pooling several slides so a slide's overall intensity
    offset/scale can't dominate the joint embedding — a light per-sample batch
    alignment (mirrors :func:`smile_msi.multivariate._zscore_cols`)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def _transform(X, metric, n_components, log1p, random_state):
    """Map a ``(pixels, peaks)`` intensity matrix to the embedding the clusterers run
    on, returning ``(scores, explained_variance)``.

    * ``"euclidean"`` (default) — per-feature z-score then PCA, so k-means / Ward
      group pixels by overall intensity profile (Pearson 1901).
    * ``"correlation"`` — mean-centre and L2-normalise each *pixel's* spectrum, so
      plain Euclidean distance on the result equals ``1 - r`` (Pearson correlation
      distance): pixels cluster by spectral *shape*, ignoring absolute intensity, the
      way SCiLS' correlation-distance segmentation does. PCA is skipped (it would
      distort that geometry), so ``explained_variance`` is ``NaN``.
    """
    if log1p:
        X = np.log1p(np.clip(X, 0, None))
    if metric == "correlation":
        Xc = X - X.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return (Xc / nrm).astype(np.float64, copy=False), float("nan")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X)
    n_comp = int(min(n_components, X.shape[1], max(1, X.shape[0] - 1)))
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(X), float(pca.explained_variance_ratio_.sum())


def _embed(ds, peaks, tol_ppm, norm, n_components, log1p, random_state, metric="euclidean",
           mask=None):
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    if X.shape[1] == 0:
        raise ValueError("segmentation needs at least one peak")
    if mask is not None:
        X = X[np.asarray(mask, bool)]           # fit/cluster on the region's pixels only
    return _transform(X, metric, n_components, log1p, random_state)


def _seg_result(ds, peaks, sub_labels, mask, k, ev, sil) -> "Segmentation":
    """Build a :class:`Segmentation`, scattering region-subset cluster ids back to
    full-pixel space. When ``mask`` is given, out-of-region pixels become ``-1``
    (unassigned) and blank (NaN) in the label image."""
    sub_labels = np.asarray(sub_labels, int)
    if mask is None:
        return Segmentation(labels=sub_labels, label_image=ds.to_image(sub_labels),
                            peaks=list(peaks), n_clusters=k, explained_variance=ev, silhouette=sil)
    mask = np.asarray(mask, bool)
    labels = np.full(ds.n_pixels, -1, dtype=int)
    labels[mask] = sub_labels
    vals = np.where(mask, labels.astype(float), np.nan)
    return Segmentation(labels=labels, label_image=ds.to_image(vals), peaks=list(peaks),
                        n_clusters=k, explained_variance=ev, silhouette=sil)


def _cluster(scores, k, random_state):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(scores)
    # relabel by descending cluster size for stable, meaningful colors
    order = np.argsort(-np.bincount(labels, minlength=k))
    remap = np.zeros(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels]


def segment(ds, peaks, n_clusters: int = 6, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
            n_components: int = 10, log1p: bool = True, random_state: int = 0,
            mask=None) -> Segmentation:
    """Partition the tissue into ``n_clusters`` molecularly distinct regions.

    PCA (Pearson 1901, doi:10.1080/14786440109462720) embedding then k-means
    (Lloyd 1982, doi:10.1109/TIT.1982.1056489). ``mask`` (bool[n_pix]) restricts the
    embedding + clustering to a region's pixels (e.g. nerve-only), so the clusters
    resolve intra-region structure; out-of-region pixels are left unassigned.
    """
    scores, ev = _embed(ds, peaks, tol_ppm, norm, n_components, log1p, random_state, mask=mask)
    k = int(min(n_clusters, scores.shape[0]))
    labels = _cluster(scores, k, random_state)
    sil = _silhouette(scores, labels)
    return _seg_result(ds, peaks, labels, mask, k, ev, sil)


def choose_k(scores, k_range=range(2, 9), random_state: int = 0, sample: int = 4000):
    """Pick the cluster count with the best silhouette over an embedding.

    Each candidate k is scored on a random pixel subsample (``sample``) for speed.
    Returns ``(best_k, labels, silhouette)`` for the full data at the best k.
    """
    rng = np.random.default_rng(random_state)
    n = scores.shape[0]
    sub = rng.choice(n, size=min(sample, n), replace=False) if n > sample else np.arange(n)
    best_k, best_score = None, -np.inf
    for k in k_range:
        if k >= n:
            continue
        labels = _cluster(scores, k, random_state)
        s = _silhouette(scores[sub], labels[sub])
        if s > best_score:
            best_k, best_score = k, s
    best_k = best_k or 2
    labels = _cluster(scores, best_k, random_state)
    return best_k, labels, _silhouette(scores[sub], labels[sub])


def auto_segment(ds, peaks, k_range=range(2, 9), tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                 n_components: int = 10, log1p: bool = True, random_state: int = 0,
                 sample: int = 4000, mask=None) -> Segmentation:
    """Segment, choosing the cluster count automatically by silhouette score
    (Rousseeuw 1987, doi:10.1016/0377-0427(87)90125-7). ``mask`` (bool[n_pix])
    restricts the embedding + clustering to a region's pixels (nerve-only)."""
    scores, ev = _embed(ds, peaks, tol_ppm, norm, n_components, log1p, random_state, mask=mask)
    best_k, labels, sil = choose_k(scores, k_range, random_state, sample)
    return _seg_result(ds, peaks, labels, mask, best_k, ev, sil)


def subcluster(ds, peaks, pixel_mask, n_clusters: int = 2, tol_ppm: float = DEFAULT_TOL_PPM,
               norm: str = "tic", n_components: int = 10, log1p: bool = True,
               random_state: int = 0, metric: str = "euclidean"):
    """Re-cluster *only* the pixels in ``pixel_mask`` — the embedding is re-fit on that
    subset, so structure that was washed out at the global level (the SCiLS
    "drill into a region" workflow) is revealed. ``metric`` matches the parent
    segmentation (``"euclidean"`` / ``"correlation"``).

    Returns ``(sub_labels, silhouette)`` where ``sub_labels`` has length
    ``pixel_mask.sum()`` with values ``0..k-1`` ordered by descending size, or
    ``(None, nan)`` if the subset is too small / has no features to split.
    """
    mask = np.asarray(pixel_mask, dtype=bool)
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)[mask]
    if X.shape[0] < max(2, n_clusters) or X.shape[1] == 0:
        return None, float("nan")
    scores, _ = _transform(X, metric, n_components, log1p, random_state)
    k = int(min(n_clusters, X.shape[0]))
    labels = _cluster(scores, k, random_state)
    return labels, _silhouette(scores, labels)


# --------------------------------------------------------------------------- #
# Agglomerative granularity tree (one model, cuttable at any detail level)
# --------------------------------------------------------------------------- #
@dataclass
class Hierarchy:
    """A granularity tree over the tissue, cuttable at any level so a single
    coarse↔fine "Detail" control can re-segment live — replacing a blind up-front
    ``k`` and the manual over-segment-then-merge / drill-down loop.

    Built **two-stage** so it scales: pixels are over-segmented into ``n_micro``
    k-means *micro-clusters* in the PCA embedding, then the micro-cluster
    *centroids* are Ward-linked into a tree (``linkage``, ``n_micro - 1`` rows).
    :func:`cut` then slices the tree to any number of segments in microseconds, so
    dragging a slider re-labels every pixel without re-clustering. ``scores`` (the
    pixel embedding) is retained to compute :func:`silhouette` lazily and to power
    a feature-space scatter.
    """
    scores: np.ndarray           # (n_used, n_comp) embedding of the segmented pixels
    micro_labels: np.ndarray     # (n_used,) micro-cluster id per segmented pixel, 0..n_micro-1
    linkage: np.ndarray          # (n_micro-1, 4) scipy linkage over micro centroids
    peaks: list                  # m/z used as features
    explained_variance: float    # fraction captured by the PCA components used (NaN if none)
    n_micro: int                 # number of micro-clusters (== max cuttable detail)
    # --- optional: how the tree was built / what it covers ----------------- #
    method: str = "ward"         # linkage strategy: "ward" (agglomerative) | "bisecting"
    metric: str = "euclidean"    # distance the embedding encodes: "euclidean" | "correlation"
    pixel_mask: np.ndarray = None  # (n_total,) bool — which dataset pixels were segmented
    n_total: int = 0             # full dataset pixel count (== len(pixel_mask) when scoped)

    @property
    def max_clusters(self) -> int:
        return int(self.n_micro)

    def labels_for(self, n_clusters: int) -> np.ndarray:
        return cut(self, n_clusters)


def _microcluster(scores, n_micro, random_state):
    """Over-segment ``scores`` into ``n_micro`` micro-clusters (size-ordered ids).
    Uses MiniBatchKMeans so over-segmenting into many micro-clusters stays cheap."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=int(n_micro), n_init=3, random_state=random_state,
                         batch_size=max(256, 3 * int(n_micro)))
    labels = km.fit_predict(scores)
    order = np.argsort(-np.bincount(labels, minlength=n_micro))
    remap = np.zeros(n_micro, dtype=int)
    remap[order] = np.arange(n_micro)
    return remap[labels]


def _bisecting_linkage(cent, sizes, random_state):
    """Divisive **bisecting k-means** (Steinbach et al. 2000) over ``cent`` (one row per
    micro-cluster, weighted by pixel ``sizes``), returned as a scipy-format linkage
    matrix so it drops straight into :func:`cut` / ``fcluster``.

    Repeatedly 2-means-splits the highest-SSE cluster until every leaf is a singleton.
    The split *order* sets the merge heights — the first (coarsest) split is the tallest
    — so cutting the tree to ``k`` clusters reproduces bisecting k-means stopped at ``k``.
    Heights are the (strictly increasing) row index, which is all ``maxclust`` needs.
    """
    from sklearn.cluster import KMeans

    M = cent.shape[0]
    w = np.asarray(sizes, dtype=float)
    w[w <= 0] = 1.0

    def _split(members):
        """Partition ``members`` (indices into ``cent``) into two non-empty groups."""
        km = KMeans(n_clusters=2, n_init=5, random_state=random_state)
        sub = km.fit_predict(cent[members], sample_weight=w[members])
        left = members[sub == 0]
        right = members[sub == 1]
        if left.size == 0 or right.size == 0:            # degenerate 2-means → peel off
            c = np.average(cent[members], axis=0, weights=w[members])  # the farthest point
            far = members[np.argmax(((cent[members] - c) ** 2).sum(1))]
            left = np.array([far])
            right = members[members != far]
        return left, right

    def _wsse(members):
        if members.size <= 1:
            return 0.0
        c = np.average(cent[members], axis=0, weights=w[members])
        return float((w[members] * ((cent[members] - c) ** 2).sum(1)).sum())

    splits = []                                          # (members, left, right), coarse→fine
    active = [np.arange(M)]
    while True:
        # Only ever split a multi-member cluster: empty micro-clusters collapse to identical
        # zero centroids, so a singleton can tie a degenerate group on SSE — picking it would
        # hand 2-means a single sample. Restrict the choice to splittable clusters.
        splittable = [j for j in range(len(active)) if active[j].size > 1]
        if not splittable:
            break
        i = max(splittable, key=lambda j: _wsse(active[j]))
        members = active.pop(i)
        left, right = _split(members)
        splits.append((members, left, right))
        active.extend([left, right])

    node_id, Z, next_id = {}, [], M
    for members, left, right in reversed(splits):        # finest split first → bottom-up
        def _id(ch):
            return int(ch[0]) if ch.size == 1 else node_id[frozenset(ch.tolist())]
        Z.append([_id(left), _id(right), 0.0, int(members.size)])
        node_id[frozenset(members.tolist())] = next_id
        next_id += 1
    Z = np.asarray(Z, dtype=float)
    Z[:, 2] = np.arange(1, Z.shape[0] + 1, dtype=float)  # strictly increasing merge heights
    return Z


def hierarchy(ds, peaks, n_micro: int = 200, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
              n_components: int = 10, log1p: bool = True, random_state: int = 0,
              method: str = "ward", metric: str = "euclidean",
              pixel_mask=None) -> Hierarchy:
    """Build a cuttable granularity :class:`Hierarchy` over the tissue (or, when
    ``pixel_mask`` is given, only those pixels — the "segment within this region" path).

    ``n_micro`` caps the finest detail (and is clamped to the pixel count). The
    heavy work (embed + micro-cluster + linkage) happens once; :func:`cut` is cheap
    enough to call on every slider tick. ``method`` picks the divisive tree —
    ``"ward"`` (agglomerative, Ward 1963) or ``"bisecting"`` (divisive bisecting
    k-means, SCiLS' default) — and ``metric`` picks the distance the embedding encodes
    (see :func:`_transform`). Both run over micro-cluster centroids so the live cut
    stays cheap on large slides.

    Refs: MiniBatch k-means micro-clustering (Sculley 2010, doi:10.1145/1772690.1772862)
    + Ward linkage (Ward 1963, doi:10.1080/01621459.1963.10500845)
    / bisecting k-means (Steinbach et al. 2000).
    """
    n_total = int(ds.n_pixels)
    mask = None if pixel_mask is None else np.asarray(pixel_mask, dtype=bool)
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    if X.shape[1] == 0:
        raise ValueError("segmentation needs at least one peak")
    if mask is not None:
        X = X[mask]
    scores, ev = _transform(X, metric, n_components, log1p, random_state)
    return _tree_from_scores(scores, ev, peaks, n_micro, method, metric, random_state,
                             pixel_mask=mask, n_total=n_total)


def _tree_from_scores(scores, ev, peaks, n_micro, method, metric, random_state,
                      pixel_mask=None, n_total=0) -> Hierarchy:
    """Build the micro-cluster + linkage :class:`Hierarchy` from a ready pixel embedding
    ``scores`` (shape ``(n_pixels, n_comp)``). Shared by the single-slide
    :func:`hierarchy` and the multi-slide :func:`joint_hierarchy` so the tree-building
    half (over-segment into micro-clusters, then link their centroids) is identical no
    matter how the embedding was produced."""
    from scipy.cluster.hierarchy import linkage as _scipy_linkage

    n = scores.shape[0]
    M = int(min(max(2, n_micro), n))
    if n < 2 or M < 2:                                   # degenerate: nothing to split
        micro = np.zeros(n, dtype=int)
        return Hierarchy(scores, micro, np.empty((0, 4)), list(peaks), ev, max(1, n),
                         method=method, metric=metric, pixel_mask=pixel_mask, n_total=n_total)
    micro = _microcluster(scores, M, random_state)
    # Centroid of each micro-cluster via per-column weighted bincount instead of M
    # full-length boolean scans (`scores[micro==m].mean(0)` for every m) — ~16-22x faster
    # at 1e6 pixels and bit-identical: bincount accumulates in pixel-index order, exactly
    # as the boolean-selected mean did. Empty micro-clusters keep a zero centroid + zero
    # size (the correlation normalisation and Ward linkage below depend on that).
    sizes = np.bincount(micro, minlength=M).astype(float)   # pixels per micro-cluster
    cent = np.zeros((M, scores.shape[1]))                    # centroid of each micro-cluster
    for c in range(scores.shape[1]):
        cent[:, c] = np.bincount(micro, weights=scores[:, c], minlength=M)
    nz = sizes > 0
    cent[nz] /= sizes[nz, None]
    if metric == "correlation":                          # keep centroids on the unit sphere
        nrm = np.linalg.norm(cent, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        cent = cent / nrm
    if method == "bisecting":
        link = _bisecting_linkage(cent, sizes, random_state)
    else:
        link = _scipy_linkage(cent, method="ward")
    return Hierarchy(scores, micro, link, list(peaks), float(ev), M,
                     method=method, metric=metric, pixel_mask=pixel_mask, n_total=n_total)


def _relabel_by_size(macro_of_micro, micro_sizes):
    """Map arbitrary macro ids (one per micro-cluster) to ``0..k-1`` ordered by
    descending total pixel count — matching the colour convention of :func:`_cluster`."""
    uniq = np.unique(macro_of_micro)
    totals = [int(micro_sizes[macro_of_micro == u].sum()) for u in uniq]
    order = [u for _, u in sorted(zip(totals, uniq), key=lambda t: -t[0])]
    remap = {int(u): i for i, u in enumerate(order)}
    return np.array([remap[int(m)] for m in macro_of_micro], dtype=int)


def cut(hier: Hierarchy, n_clusters: int) -> np.ndarray:
    """Slice ``hier`` into ``n_clusters`` segments → per-pixel labels ``0..k-1``
    (size-ordered). Clamped to ``[1, hier.max_clusters]``. Microsecond-cheap, so a
    Detail slider can call this on every tick. ``fcluster`` may merge below the
    requested count when the tree has fewer well-separated branches; the result is
    compacted so labels stay contiguous."""
    from scipy.cluster.hierarchy import fcluster

    k = int(max(1, min(int(n_clusters), hier.max_clusters)))
    micro_sizes = np.bincount(hier.micro_labels, minlength=hier.n_micro)
    if k <= 1 or hier.linkage.shape[0] == 0:
        micro_to_macro = np.zeros(hier.n_micro, dtype=int)
    else:
        macro = fcluster(hier.linkage, t=k, criterion="maxclust")   # 1..≤k, len n_micro
        micro_to_macro = _relabel_by_size(macro, micro_sizes)
    return micro_to_macro[hier.micro_labels]


def silhouette(hier: Hierarchy, labels, sample: int = 4000, random_state: int = 0) -> float:
    """Silhouette of a ``labels`` partition over the hierarchy's pixel embedding,
    on a random subsample for speed (kept off the slider's hot path — compute it
    only once the slider settles)."""
    rng = np.random.default_rng(random_state)
    n = hier.scores.shape[0]
    sub = rng.choice(n, size=min(sample, n), replace=False) if n > sample else np.arange(n)
    return _silhouette(hier.scores[sub], np.asarray(labels)[sub])


def segmentation_at(ds, hier: Hierarchy, n_clusters: int,
                    with_silhouette: bool = False) -> Segmentation:
    """Wrap a :func:`cut` of ``hier`` into a :class:`Segmentation` so it drops into
    the same downstream (region builder, discriminating features, report) the
    k-means path feeds. Silhouette is skipped by default (don't pay for it on every
    slider tick); pass ``with_silhouette=True`` when the value is actually shown."""
    sub = cut(hier, n_clusters)
    k = int(sub.max()) + 1 if sub.size else 0
    sil = silhouette(hier, sub) if with_silhouette else float("nan")
    mask = getattr(hier, "pixel_mask", None)
    if mask is None:                                     # whole-slide: labels are per-pixel
        labels = sub
        img_src = sub.astype(float)
    else:                                                # region-scoped: scatter back, -1 outside
        labels = np.full(int(getattr(hier, "n_total", 0) or ds.n_pixels), -1, dtype=int)
        labels[mask] = sub
        img_src = np.where(labels < 0, np.nan, labels.astype(float))
    return Segmentation(labels=labels, label_image=ds.to_image(img_src), peaks=list(hier.peaks),
                        n_clusters=k, explained_variance=hier.explained_variance, silhouette=sil)


# --------------------------------------------------------------------------- #
# Joint (consensus) segmentation — one tree over several slides at once
# --------------------------------------------------------------------------- #
@dataclass
class JointHierarchy:
    """A granularity tree built **jointly** over the pixels of several samples on a
    shared feature axis, so a single :func:`cut` assigns the *same* cluster ids across
    every slide — "cluster 3" means the same molecular profile in each sample, the
    cross-slide comparison a per-sample :class:`Hierarchy` can't give (independent runs
    have unrelated, unmatchable labels).

    The pixels of every sample are stacked in order (sample 0's pixels first, then
    sample 1, …), embedded together, and linked into one tree (``hier``). Cut it with
    :func:`cut` exactly like a single-slide hierarchy, then split the per-pixel labels
    back out per sample with :func:`joint_segmentation_at` (``sample_sizes`` records the
    row count each sample contributed, ``masks``/``n_totals`` scatter region-scoped
    labels back onto each slide's full grid).
    """
    hier: Hierarchy              # the underlying tree over the pooled pixels
    sample_names: list           # display name per sample (pooled order)
    sample_sizes: list           # pixels each sample contributed to the pool (pooled order)
    masks: list                  # per-sample pixel mask used (None = whole slide)
    n_totals: list               # per-sample ds.n_pixels (to scatter masked labels back)
    targets: np.ndarray          # shared m/z axis the tree was built on
    standardized: bool           # were features per-sample z-scored before pooling?
    features: np.ndarray = None  # pooled (Σ pixels × targets) RAW normalized intensities in
                                 # pooled order — retained so per-cluster signatures need no cube
                                 # re-extraction (lets the GUI release each cube after the run)

    @property
    def max_clusters(self) -> int:
        return self.hier.max_clusters

    @property
    def n_samples(self) -> int:
        return len(self.sample_names)


def _joint_embed(blocks, metric, n_components, log1p, random_state, standardize_per_sample):
    """Embed a list of per-sample ``(n_pixels_i, n_features)`` intensity matrices into one
    shared space, returning ``(scores, explained_variance)`` over the **pooled** pixels.

    Each block is optionally ``log1p``'d and per-sample z-scored
    (``standardize_per_sample`` — removes slide-to-slide intensity offset/scale so a batch
    effect can't masquerade as biology), then stacked. For ``"euclidean"`` the pooled
    matrix is PCA-projected (skipping the redundant global standardize when blocks are
    already z-scored); for ``"correlation"`` each pooled pixel's spectrum is mean-centred
    and L2-normalised so Euclidean distance equals ``1 - r`` — mirroring :func:`_transform`
    so a joint run matches the single-slide geometry."""
    proc = []
    for X in blocks:
        if log1p:
            X = np.log1p(np.clip(X, 0, None))
        if standardize_per_sample:
            X = _zscore_cols(X)
        proc.append(X)
    Xall = np.vstack(proc)
    if metric == "correlation":
        Xc = Xall - Xall.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return (Xc / nrm).astype(np.float64, copy=False), float("nan")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # per-sample z-score already centred/scaled each block; a global StandardScaler would
    # be redundant. Without it, standardize globally like the single-slide path.
    Xs = Xall if standardize_per_sample else StandardScaler().fit_transform(Xall)
    n_comp = int(min(n_components, Xs.shape[1], max(1, Xs.shape[0] - 1)))
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(Xs), float(pca.explained_variance_ratio_.sum())


def joint_hierarchy(datasets, peaks, *, masks=None, names=None, n_micro: int = 200,
                    tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic", n_components: int = 10,
                    log1p: bool = True, random_state: int = 0, method: str = "ward",
                    metric: str = "euclidean",
                    standardize_per_sample: bool = True,
                    release_cubes: bool = False) -> JointHierarchy:
    """Build one cuttable granularity tree over the pixels of **several** ``datasets`` at
    once, on the *shared* ``peaks`` axis (so every slide spans the same molecular
    features). The result cuts to a partition whose cluster ids mean the same thing in
    every sample — the multi-slide analogue of :func:`hierarchy`.

    ``masks`` (one bool pixel-mask per dataset, ``None`` = whole slide) restricts each
    slide to a region; ``names`` labels the samples. ``standardize_per_sample`` per-sample
    z-scores the features before pooling (a light batch alignment — strongly recommended
    across independent acquisitions). The heavy embed+link happens once; :func:`cut` is
    then microsecond-cheap, so a Detail slider re-segments every slide live.

    ``release_cubes`` frees each slide's dense in-RAM cube the moment its (much smaller)
    feature block is extracted — so a big cohort holds only **one** cube plus the pooled
    feature blocks at a time, instead of every cube at once (the joint-seg OOM). The slides'
    geometry survives, so :func:`joint_segmentation_at` can still render each label map.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("joint segmentation needs at least one sample")
    peaks = list(peaks)
    if len(peaks) == 0:
        raise ValueError("segmentation needs at least one peak")
    blocks, sizes, n_totals, used_masks = [], [], [], []
    for i, ds in enumerate(datasets):
        X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
        if X.shape[1] == 0:
            raise ValueError("segmentation needs at least one peak")
        m = None if not masks else masks[i]
        m = None if m is None else np.asarray(m, dtype=bool)
        if m is not None:
            X = X[m]
        n_total = int(ds.n_pixels)                       # read before release (still valid after)
        blocks.append(np.array(X, dtype=np.float32))     # own the data so release can free the cube
        sizes.append(int(X.shape[0]))
        n_totals.append(n_total)
        used_masks.append(m)
        if release_cubes and hasattr(ds, "release"):
            ds.release()                                  # free this cube; block + geometry remain
    # Capture the pooled RAW feature matrix before _joint_embed (which log1p/z-scores copies)
    # — per-cluster signatures read from this, so the GUI can release each cube after the run.
    pooled = (np.vstack(blocks).astype(np.float32) if blocks
              else np.zeros((0, len(peaks)), dtype=np.float32))
    scores, ev = _joint_embed(blocks, metric, n_components, log1p, random_state,
                              standardize_per_sample)
    hier = _tree_from_scores(scores, ev, peaks, n_micro, method, metric, random_state,
                             pixel_mask=None, n_total=int(sum(sizes)))
    names = list(names) if names else [f"sample {i}" for i in range(len(datasets))]
    return JointHierarchy(hier=hier, sample_names=names, sample_sizes=sizes,
                          masks=used_masks, n_totals=n_totals,
                          targets=np.asarray(peaks, dtype=float),
                          standardized=bool(standardize_per_sample), features=pooled)


def joint_segmentation_at(datasets, jh: JointHierarchy, n_clusters: int,
                          with_silhouette: bool = False) -> list:
    """Cut a :class:`JointHierarchy` to ``n_clusters`` and split the labels back out into
    one :class:`Segmentation` **per sample**, all sharing the same cluster ids/palette.
    ``datasets`` must be the same list (same order) passed to :func:`joint_hierarchy` —
    each ``label_image`` is rendered on its own slide's grid. The silhouette (computed
    once over the pooled embedding) is copied onto every returned segmentation."""
    datasets = list(datasets)
    sub = cut(jh.hier, n_clusters)                       # per pooled-pixel labels, sample-concat
    k = int(sub.max()) + 1 if sub.size else 0
    sil = silhouette(jh.hier, sub) if with_silhouette else float("nan")
    out, off = [], 0
    for i, ds in enumerate(datasets):
        n = jh.sample_sizes[i]
        sub_i = sub[off:off + n]
        off += n
        mask = jh.masks[i] if i < len(jh.masks) else None
        if mask is None:                                 # whole-slide: labels are per-pixel
            labels = sub_i
            img_src = sub_i.astype(float)
        else:                                            # region-scoped: scatter back, -1 outside
            labels = np.full(jh.n_totals[i], -1, dtype=int)
            labels[mask] = sub_i
            img_src = np.where(labels < 0, np.nan, labels.astype(float))
        out.append(Segmentation(labels=labels, label_image=ds.to_image(img_src),
                                peaks=list(jh.hier.peaks), n_clusters=k,
                                explained_variance=jh.hier.explained_variance, silhouette=sil))
    return out


# --------------------------------------------------------------------------- #
# Spatial feature finding (which ions are actually spatially structured)
# --------------------------------------------------------------------------- #
def _grid_edges(ds):
    """Undirected 4-neighbour adjacency between acquired pixels as (ei, ej) arrays."""
    rows, cols = ds._pixel_rows_cols()
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    ei, ej = [], []
    for (r, c), i in lookup.items():
        for nbr in ((r, c + 1), (r + 1, c)):
            j = lookup.get(nbr)
            if j is not None:
                ei.append(i)
                ej.append(j)
    return np.asarray(ei, int), np.asarray(ej, int)


def detect_samples(ds, rel_threshold: float = 0.10, min_frac: float = 0.004,
                   smooth: float = 1.0):
    """Auto-find tissue 'samples' on a slide: smooth + threshold the total-ion-current
    image, fill holes, and label 4/8-connected components. Returns a list of boolean
    per-pixel masks (one per detected sample), largest first; components below
    ``min_frac`` of the pixels are dropped as specks. Returns ``[]`` when nothing
    clears the threshold (e.g. a single tissue piece that fills the field)."""
    from scipy import ndimage
    img = np.asarray(ds.tic_image(), dtype=float)
    finite = np.isfinite(img)
    vals = img[finite & (img > 0)]
    if vals.size == 0:
        return []
    filled = np.where(finite, img, 0.0)
    img_s = ndimage.gaussian_filter(filled, sigma=float(smooth)) if smooth and smooth > 0 else filled
    thr = float(rel_threshold) * float(np.median(vals))
    binary = ndimage.binary_fill_holes(finite & (img_s > thr))
    labeled, n = ndimage.label(binary)
    if n == 0:
        return []
    rows, cols = ds._pixel_rows_cols()
    comp = labeled[rows, cols]                          # component id per acquired pixel (0 = bg)
    min_px = max(20, int(min_frac * ds.n_pixels))
    masks = [(comp == cid) for cid in range(1, n + 1)]
    masks = [m for m in masks if int(m.sum()) >= min_px]
    masks.sort(key=lambda m: -int(m.sum()))
    return masks


def ring_mask(ds, mask, width_px: float, mode: str = "inner"):
    """Turn a filled per-pixel ``mask`` into a border ring of ``width_px`` pixels, using
    a Euclidean distance transform on the tissue grid. Modes:

    * ``"inner"`` — the rim *inside* the shape (drop the interior, keep the border).
    * ``"outer"`` — a collar of tissue *outside* the shape.
    * ``"band"``  — both sides of the boundary (inner ∪ outer).

    Returns a per-pixel boolean mask (only acquired pixels can be True, so an outer
    ring never includes off-tissue grid cells)."""
    from scipy import ndimage
    mask = np.asarray(mask, dtype=bool)
    w = float(width_px)
    if w <= 0 or not mask.any():
        return mask
    rows, cols = ds._pixel_rows_cols()
    grid = np.zeros((ds.height, ds.width), dtype=bool)
    grid[rows, cols] = mask
    ring = np.zeros_like(grid)
    if mode in ("inner", "band"):
        d_in = ndimage.distance_transform_edt(grid)        # dist from inside → boundary
        ring |= grid & (d_in <= w)
    if mode in ("outer", "band"):
        d_out = ndimage.distance_transform_edt(~grid)      # dist from outside → boundary
        ring |= (~grid) & (d_out <= w)
    out = np.zeros(ds.n_pixels, dtype=bool)
    out[:] = ring[rows, cols]
    return out


_MORANS_CHUNK_BYTES = 64 << 20     # transient budget per Moran's chunk (~64 MB)


def _stage(progress, lo, hi):
    """Re-scale a sub-call's ``(done, total)`` into the ``[lo, hi]`` band of a 0..100 whole.

    Feature finding used to hand ``progress`` to the extraction step alone, so the bar reached
    100% and then sat there — motionless but working — through the frequency and Moran's
    stages. Staging keeps it honest about which fraction of the *whole* job is done."""
    if progress is None:
        return None

    def cb(done, total):
        frac = (done / total) if total else 0.0
        progress(int(lo + (hi - lo) * frac), 100)
    return cb


def spatial_autocorrelation(ds, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                            progress=None):
    """Moran's I spatial autocorrelation per peak (how spatially structured each ion
    image is, from -1 to ~1; ~0 = random/noise). Returns a DataFrame ``mz, morans_i``
    sorted by descending I — the backbone of spatial feature finding.

    Reference: Moran (1950), Biometrika 37(1–2):17–23, doi:10.1093/biomet/37.1-2.17.
    """
    import pandas as pd

    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    ei, ej = _grid_edges(ds)
    n, E = X.shape[0], len(ei)
    P = X.shape[1]
    Z = X - X.mean(axis=0, keepdims=True)
    denom = (Z * Z).sum(axis=0)
    if E == 0:
        morans = np.zeros(P)
    else:
        # Chunk over ions. Computing ``(Z[ei] * Z[ej]).sum(axis=0)`` in one shot allocates two
        # E x P fancy-index copies plus their product: for 2000 candidates over a 60k-pixel
        # slide that is ~8 GB of transients, which swaps and reads to the user as a hang.
        # Per chunk the transient is E x CHUNK instead, and each chunk is a cancellation point.
        num = np.empty(P, dtype=np.float64)
        chunk = max(1, int(_MORANS_CHUNK_BYTES // max(1, E * 8 * 3)))
        for s in range(0, P, chunk):
            e = min(P, s + chunk)
            num[s:e] = (Z[ei, s:e] * Z[ej, s:e]).sum(axis=0)
            if progress is not None:
                progress(e, P)
        # flat columns (denom == 0, e.g. baseline ripples) have undefined Moran's → 0
        morans = np.divide(n * num, E * denom, out=np.zeros(P), where=denom > 0)
    df = pd.DataFrame({"mz": np.asarray(peaks, float), "morans_i": morans})
    return df.sort_values("morans_i", ascending=False).reset_index(drop=True)


def select_spatial_features(ds, peaks, min_morans: float = 0.1, tol_ppm: float = DEFAULT_TOL_PPM,
                            norm: str = "tic"):
    """Keep only peaks whose ion image is spatially structured (Moran's I >=
    ``min_morans``). Returns the surviving m/z list (descending I)."""
    df = spatial_autocorrelation(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    return df[df["morans_i"] >= min_morans]["mz"].tolist()


# --------------------------------------------------------------------------- #
# Spatially-aware feature finder
# --------------------------------------------------------------------------- #
def feature_frequency(ds, peaks, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                      present_frac: float = 0.05):
    """Per-peak **footprint**: the fraction of pixels in which each peak is *present*.
    A peak counts as present in a pixel when its raw (un-normalized) intensity exceeds
    ``present_frac`` of that peak's own maximum over all pixels — a scale-free,
    per-ion floor. So a single bright spike reads as present in ~one pixel (a
    vanishing footprint) while a *real* ion reads as present across its whole spatial
    extent, whether it is focal or ubiquitous. Returns an ndarray of fractions in
    ``[0, 1]`` aligned to ``peaks``.

    The earlier ``median + k·MAD`` floor was *not* a footprint measure: for an ion
    present in many pixels the median itself becomes signal, so almost nothing cleared
    the floor and a perfectly reproducible ion reported a frequency near zero — which
    made the frequency gate cull real, well-distributed features (the over-culling
    that dragged the spatial finder below a plain mean threshold). A floor relative to
    each ion's own peak height fixes that.

    This is the frequency / "present in >=X% of spectra" gate (Cardinal's
    ``peakFilter`` freq.min; Bemis et al. 2015, doi:10.1093/bioinformatics/btv146)
    that drops single-pixel noise before the spatial-coherence step.
    """
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm="none", reduce=reduce)
    if X.shape[0] == 0:
        return np.zeros(X.shape[1])
    cmax = X.max(axis=0)
    floor = float(np.clip(present_frac, 0.0, 1.0)) * cmax
    present = (X > floor[None, :]) & (cmax[None, :] > 0)   # absent ions (max 0) → 0
    return present.mean(axis=0)


# --------------------------------------------------------------------------- #
# Ion-image quality metrics (is this image real biology or an artifact?)
# --------------------------------------------------------------------------- #
def hotspot_fraction(ds, peaks, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                     top_frac: float = 0.01):
    """Per-peak **hotspot concentration**: the fraction of an ion's *total* signal that
    sits in its brightest ``top_frac`` of on-tissue pixels.

    A spatially spread ion piles only about ``top_frac`` of its signal into its top
    ``top_frac`` of pixels (e.g. ~1% of the signal in the top 1% of pixels); a matrix
    crystallization / delocalization artifact concentrates almost all of its signal into a
    handful of pixels, so the fraction approaches 1. So **high = likely artifact** — the
    opposite polarity to Moran's I / :func:`spatial_chaos`, where high = good. Returns an
    ndarray in ``[0, 1]`` aligned to ``peaks`` (an absent ion → 0).

    Computed on **raw** (un-normalized) intensities: hotspotting is a property of the
    physical signal, not of any per-pixel normalization. ``top_frac`` is a fraction of the
    acquired-pixel count (rounded up, at least one pixel) so the metric is comparable across
    slides of different sizes.
    """
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm="none", reduce=reduce)
    n_pix, n_peaks = X.shape
    if n_pix == 0:
        return np.zeros(n_peaks)
    k = int(np.clip(np.ceil(float(top_frac) * n_pix), 1, n_pix))
    total = X.sum(axis=0)
    topk = np.partition(X, n_pix - k, axis=0)[n_pix - k:]      # the k largest per column (unordered)
    topsum = topk.sum(axis=0)
    return np.divide(topsum, total, out=np.zeros(n_peaks), where=total > 0)


# Below this many non-zero pixels an ion image has no meaningful morphology, so the chaos
# measure is undefined (matches the reference implementation's guard).
CHAOS_MIN_PIXELS = 4


def spatial_chaos(ds, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                  nlevels: int = 30, hotspot_pct: float = 99.0):
    """Per-peak **measure of spatial chaos** ρ via the level-sets method (Alexandrov &
    Bartels 2013, Bioinformatics 29(18):2335, doi:10.1093/bioinformatics/btr585; the
    structure term in METASPACE's MSM, Palmer et al. 2017, Nat Methods 14:57).

    Each ion image is hotspot-clipped (at the ``hotspot_pct`` percentile) and normalized to
    ``[0, 1]``; for each of ``nlevels`` thresholds it is binarized and its number of
    connected components (4-connectivity) counted. A spatially organized ion forms one (or a
    few) blobs at every level → few objects → ρ near 1; random speckle shatters into many
    disconnected objects → ρ near 0.

    **High = structured (real), low = chaotic (candidate for auto-reject)** — the same
    polarity as Moran's I, but morphology-based rather than autocorrelation-based, so the two
    disagree usefully: a *uniform* ion scores ~0 on Moran's yet high here (it is organized,
    not chaotic). Note the historical name is a misnomer — the score rises with *structure*.

    Returns an ndarray aligned to ``peaks``; a peak with fewer than :data:`CHAOS_MIN_PIXELS`
    non-zero pixels is scored ``nan`` (no meaningful morphology).
    """
    from scipy import ndimage

    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    n_pix, n_peaks = X.shape
    out = np.full(n_peaks, np.nan)
    if n_pix == 0:
        return out
    rows, cols = ds._pixel_rows_cols()
    conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)   # 4-connectivity
    levels = np.linspace(0.0, 1.0, int(nlevels) + 2)[1:-1]          # nlevels thresholds in (0, 1)
    nlev = len(levels)
    if nlev == 0:
        return out
    for j in range(n_peaks):
        col = X[:, j]
        nz = col > 0
        sum_nz = int(nz.sum())
        if sum_nz < CHAOS_MIN_PIXELS:
            continue                                               # stays nan: no morphology
        hot = float(np.percentile(col[nz], hotspot_pct))
        if hot <= 0:
            continue
        v = np.clip(col / hot, 0.0, 1.0)
        grid = np.zeros((ds.height, ds.width), dtype=float)        # off-tissue cells stay 0 → excluded
        grid[rows, cols] = v
        total_objs = 0
        for lev in levels:
            _, n = ndimage.label(grid > lev, structure=conn)
            total_objs += int(n)
        out[j] = 1.0 - total_objs / (sum_nz * nlev)
    return out


def _interp_cross(x0, y0, x1, y1, y):
    """m/z where a straight segment between ``(x0,y0)`` and ``(x1,y1)`` crosses ``y``."""
    if y1 == y0:
        return x0
    return x0 + (y - y0) * (x1 - x0) / (y1 - y0)


def peak_fwhm_ppm(axis, spec, mz: float, max_ppm: float = 200.0, snap: int = 3) -> float:
    """Full-width-at-half-maximum of the spectral peak nearest ``mz``, in ppm — the
    data-driven integration width for that feature instead of a fixed tolerance.

    Snaps to the local apex within ``±snap`` bins, then linearly interpolates the
    half-height crossing on each flank for sub-bin precision. Falls back to
    ``max_ppm`` when the peak can't be bracketed (runs into a neighbour or the
    spectrum edge)."""
    axis = np.asarray(axis, float)
    spec = np.asarray(spec, float)
    n = len(axis)
    if n == 0 or mz <= 0:
        return float(max_ppm)
    i = int(np.argmin(np.abs(axis - mz)))
    lo_s, hi_s = max(0, i - snap), min(n, i + snap + 1)
    i = lo_s + int(np.argmax(spec[lo_s:hi_s]))
    top = float(spec[i])
    if top <= 0:
        return float(max_ppm)
    half = top / 2.0
    left = i
    while left > 0 and spec[left] > half:
        left -= 1
    right = i
    while right < n - 1 and spec[right] > half:
        right += 1
    # left/right now sit at (or just past) the half-height crossing; interpolate
    x_lo = _interp_cross(axis[left], spec[left], axis[left + 1], spec[left + 1], half) \
        if spec[left] <= half < spec[left + 1] else axis[left]
    x_hi = _interp_cross(axis[right - 1], spec[right - 1], axis[right], spec[right], half) \
        if spec[right] <= half < spec[right - 1] else axis[right]
    width = float(x_hi - x_lo)
    if width <= 0:
        return float(max_ppm)
    return float(min(width / mz * 1e6, max_ppm))


@dataclass
class SpatialFeatures:
    """Result of :func:`find_spatial_features` — the surviving features plus the
    drop count at each pipeline stage (the detection funnel)."""
    peaks: list                  # enriched peak dicts, descending Moran's I
    n_candidates: int            # candidate peaks detected (mean / skyline / both)
    n_after_frequency: int       # survived the reproducibility gate
    n_after_morans: int          # survived the spatial-denoise gate
    suggested_ppm: float         # median auto-width of the survivors (NaN if none)
    params: dict
    n_after_collapse: int = -1   # survived isotope collapse (== len(peaks)); -1 if collapse off


def _merge_candidates(primary, extra, tol_ppm: float, max_candidates: int):
    """Union two candidate peak lists, keeping every ``primary`` entry and adding only
    the ``extra`` peaks not already present within ``tol_ppm`` (``primary`` wins on
    overlap). Re-sorted by descending intensity and capped at ``max_candidates``."""
    out = list(primary)
    mzs = [float(p["mz"]) for p in primary]
    arr = np.asarray(mzs, float)
    for p in extra:
        mz = float(p["mz"])
        if arr.size and float(np.min(np.abs(arr - mz))) <= mz * tol_ppm / 1e6:
            continue
        out.append(p)
        arr = np.append(arr, mz)
    out.sort(key=lambda d: d["intensity"], reverse=True)
    return out[:max_candidates]


def _pick_candidates(ds, projection, snr, min_rel_intensity, max_candidates, mask,
                     prominence, tol_ppm):
    """Candidate peaks for the spatial finder. ``projection`` selects the source
    spectrum:

    * ``'mean'`` (default) — the mean spectrum. This is the high-coverage, low-noise
      feature set; picking candidates on the *mean*
      (not the noisy skyline) is what keeps the finder's coverage up.
    * ``'max'`` — the skyline (per-m/z maximum); surfaces ions bright in only a few
      pixels, but the skyline is noisy so it admits far more junk candidates.
    * ``'both'`` — union of the two, de-duplicated within ``tol_ppm`` (mean for the
      bulk + skyline to rescue focal ions); the spatial-denoise gate then strips the
      extra skyline noise.
    """
    proj = (projection or "mean").lower()
    if proj in ("max", "skyline"):
        proj = "max"
    if proj in ("mean", "max"):
        return ds.pick_peaks(snr=snr, min_rel_intensity=min_rel_intensity,
                             max_peaks=max_candidates, projection=proj, mask=mask,
                             prominence=prominence)
    if proj in ("both", "mean+max", "union"):
        mean = ds.pick_peaks(snr=snr, min_rel_intensity=min_rel_intensity,
                             max_peaks=max_candidates, projection="mean", mask=mask,
                             prominence=prominence)
        mx = ds.pick_peaks(snr=snr, min_rel_intensity=min_rel_intensity,
                           max_peaks=max_candidates, projection="max", mask=mask,
                           prominence=prominence)
        return _merge_candidates(mean, mx, tol_ppm=tol_ppm, max_candidates=max_candidates)
    raise ValueError(f"unknown projection {projection!r}; use 'mean', 'max', or 'both'")


def find_spatial_features(ds, snr: float = 3.0, min_rel_intensity: float = 0.0,
                          max_candidates: int = 2000, min_frequency: float = 0.0,
                          min_morans: float = 0.0, tol_ppm: float = DEFAULT_TOL_PPM,
                          norm: str = "tic", reduce: str = "sum", mask=None,
                          auto_width: bool = True, prominence: float = 1.0,
                          projection: str = "mean", rescue_frequency: float | None = None,
                          collapse_isotopes: bool = False, charge: int = 1,
                          iso_tol_ppm: float | None = None,
                          progress=None) -> SpatialFeatures:
    """Spatially-aware feature finder — chains the engine's primitives into one
    pass that surfaces reproducible, spatially-structured ions, built from published
    primitives (spatial autocorrelation, frequency gating, isotope collapse):

    1. **Candidate detection** on the **mean** spectrum by default
       (:meth:`MSIDataset.pick_peaks` ``projection='mean'``). Earlier versions picked
       on the *skyline* (per-m/z maximum); that surfaces locally-bright ions but the
       skyline is noisy, so it admitted far more junk than it rescued and dragged
       coverage of the real feature set *down*. The mean spectrum is the high-coverage,
       low-noise starting point for the candidate pool; ``projection='max'`` or
       ``'both'`` (mean ∪ skyline, de-duplicated) opt back into skyline recall when
       focal ions matter.
    2. **Frequency / reproducibility** gate — keep peaks *present* in at least
       ``min_frequency`` of pixels, dropping single-pixel spikes
       (:func:`feature_frequency`).
    3. **Spatial denoise** — drop candidates whose ion image is spatially random
       (Moran's I < ``min_morans``), the hallmark of chemical/electronic noise. Real
       features — focal *or* ubiquitous — sit far above the noise floor on Moran's I
       (a uniformly-present *real* ion still scores high from tissue-vs-background
       structure), so a low threshold strips noise without culling signal. The earlier
       over-culling came not from this gate but from a broken frequency metric (now
       fixed in :func:`feature_frequency`) and from picking candidates on the noisy
       skyline. Set ``rescue_frequency`` (a fraction in ``(0, 1]``) to *also* keep any
       reproducible ion present in at least that fraction of pixels regardless of
       Moran's — an opt-in escape hatch for genuinely uniform ions inside a homogeneous
       ROI; left ``None`` (default) it would readmit uniformly-present noise. Uses
       :func:`spatial_autocorrelation`.
    4. **Auto-interval width** — each survivor gets its own FWHM-derived window
       (:func:`peak_fwhm_ppm`) reported as ``width_ppm`` / ``fwhm_mz`` instead of a
       single fixed tolerance.
    5. **Isotope collapse** (opt-in, ``collapse_isotopes=True``) — fold M+1/M+2
       satellites into their monoisotopic peak (:func:`isotopes.deisotope`; 13C spacing
       1.00336 Da / ``charge``, Senko et al. 1995). This is the "region-complete"
       redundancy reduction (one compound → one feature), and is the single biggest lever
       bringing the count down — on these data ~30–42% of picked peaks are
       isotopologues of another peak. Off by default (preserves the raw feature set);
       ``iso_tol_ppm`` controls the satellite-match window (defaults to ``min(tol_ppm, 15)``).

    Defaults (``min_frequency=0``, ``min_morans=0``, ``collapse_isotopes=False``) reproduce
    the plain mean feature set; turn the gates up to denoise. ``mask`` restricts *detection*
    to a region's pixels (the feature list is still extracted over the whole slide). Returns a
    :class:`SpatialFeatures` whose ``peaks`` are ready to drop into the working set;
    each carries ``frequency``, ``morans_i`` and (when ``auto_width``) ``width_ppm`` /
    ``fwhm_mz`` alongside the usual ``mz`` / ``intensity`` / ``rel_intensity`` / ``snr``.
    """
    params = dict(snr=snr, min_rel_intensity=min_rel_intensity, projection=projection,
                  min_frequency=min_frequency, min_morans=min_morans, tol_ppm=tol_ppm,
                  norm=norm, reduce=reduce, auto_width=auto_width,
                  rescue_frequency=rescue_frequency, scoped=mask is not None,
                  collapse_isotopes=collapse_isotopes)
    cand = _pick_candidates(ds, projection, snr, min_rel_intensity, max_candidates,
                            mask, prominence, tol_ppm)
    if not cand:
        return SpatialFeatures([], 0, 0, 0, float("nan"), params)
    mzs = [p["mz"] for p in cand]
    # one raw feature matrix built here is reused by both gates below (same peaks/
    # tol/reduce → cache hit in feature_frequency and spatial_autocorrelation)
    ds.ensure_features(mzs, tol_ppm=tol_ppm, reduce=reduce,
                       progress=_stage(progress, 0, 50))
    if progress is not None:
        progress(50, 100)
    freq = feature_frequency(ds, mzs, tol_ppm=tol_ppm, reduce=reduce)
    if progress is not None:
        progress(62, 100)
    sa = spatial_autocorrelation(ds, mzs, tol_ppm=tol_ppm, norm=norm,
                                 progress=_stage(progress, 62, 95))
    if progress is not None:
        progress(95, 100)
    morans = {round(float(m), 6): float(i) for m, i in zip(sa["mz"], sa["morans_i"])}
    freq_map = {round(float(m), 6): float(f) for m, f in zip(mzs, freq)}
    rescue = None if rescue_frequency is None else float(rescue_frequency)
    # auto-width is read off the same projection the candidates came from
    if auto_width:
        axis, spec = (ds.mean_spectrum(mask=mask) if (projection or "mean").lower() == "mean"
                      else ds.max_spectrum(mask=mask))
    else:
        axis, spec = None, None
    out, widths = [], []
    n_after_freq = 0
    for p in cand:
        key = round(float(p["mz"]), 6)
        f = freq_map.get(key, 0.0)
        mi = morans.get(key, 0.0)
        if f < min_frequency:                         # reproducibility floor
            continue
        n_after_freq += 1
        # spatial denoise: keep if spatially coherent OR widely reproducible; only ions
        # that fail both (scattered noise) are dropped.
        if mi < min_morans and not (rescue is not None and f >= rescue):
            continue
        q = dict(p)
        q["frequency"] = f
        q["morans_i"] = mi
        if auto_width:
            w = peak_fwhm_ppm(axis, spec, p["mz"])
            q["width_ppm"] = w
            q["fwhm_mz"] = p["mz"] * w / 1e6
            widths.append(w)
        out.append(q)
    out.sort(key=lambda d: d["morans_i"], reverse=True)
    n_spatial = len(out)                                   # survived spatial denoise (pre-collapse)
    n_collapsed = -1                                       # sentinel: collapse not run
    if collapse_isotopes and out:
        # Collapse M+1/M+2 satellites to the monoisotopic peak (one compound → one feature),
        # the "region-complete" reduction (one compound → one feature). Match window defaults tight.
        from . import isotopes
        itol = iso_tol_ppm if (iso_tol_ppm and iso_tol_ppm > 0) else min(tol_ppm, 15.0)
        out, _ = isotopes.deisotope(out, charge=charge, tol_ppm=itol)
        n_collapsed = len(out)
    suggested = float(np.median(widths)) if widths else float("nan")
    if progress is not None:
        progress(100, 100)
    return SpatialFeatures(out, len(cand), n_after_freq, n_spatial, suggested, params, n_collapsed)


# --------------------------------------------------------------------------- #
# Coherent feature extraction — spatial finder + artifact-rejection quality gate
# --------------------------------------------------------------------------- #
@dataclass
class CoherentFeatures:
    """Result of :func:`find_coherent_features` — spatially-coherent, artifact-screened
    features plus the drop count at each pipeline stage (the detection funnel)."""
    peaks: list                  # enriched dicts, descending composite quality
    n_candidates: int            # candidate peaks detected
    n_after_spatial: int         # survived find_spatial_features (freq + Moran's + isotope)
    n_after_quality: int         # survived the artifact-rejection quality gate (== len(peaks))
    suggested_ppm: float         # median auto-width of the survivors (NaN if none)
    params: dict


def _composite_quality(morans, chaos, hotspot) -> float:
    """Fuse the per-ion spatial metrics into one quality score in ``[0, 1]``.

    ``structure`` is the mean of the available structure signals — Moran's I
    (autocorrelation, clipped to ``[0, 1]``) and the level-set spatial-chaos ρ
    (morphology; ``NaN`` when the ion has too few pixels, then dropped from the mean).
    The **hotspot** fraction (share of signal in the brightest ~1% of pixels; high = a
    delocalization / matrix-crystallization artifact whose signal collapses onto a few
    pixels) multiplies the structure down: a well-spread ion (hotspot≈0) keeps its
    structure score, while an ion that clears Moran's yet dumps its signal into a handful
    of pixels (hotspot→1) is driven toward 0. Returns ``structure * (1 - hotspot)``.
    """
    m = float(np.clip(morans, 0.0, 1.0))
    parts = [m]
    if chaos is not None and np.isfinite(chaos):
        parts.append(float(np.clip(chaos, 0.0, 1.0)))
    structure = float(np.mean(parts))
    h = 0.0 if (hotspot is None or not np.isfinite(hotspot)) else float(np.clip(hotspot, 0.0, 1.0))
    return structure * (1.0 - h)


def find_coherent_features(ds, *, snr: float = 3.0, min_rel_intensity: float = 0.0,
                           max_candidates: int = 2000, min_frequency: float = 0.01,
                           min_morans: float = 0.0, min_quality: float = 0.15,
                           max_hotspot: float = 0.80, tol_ppm: float = DEFAULT_TOL_PPM,
                           norm: str = "tic", reduce: str = "sum", mask=None,
                           projection: str = "mean", collapse_isotopes: bool = True,
                           charge: int = 1, rescue_frequency: float | None = None,
                           iso_tol_ppm: float | None = None, progress=None) -> CoherentFeatures:
    """Coherent feature extraction — a spatially-aware peak picker that adds an
    **artifact-rejection quality gate** on top of :func:`find_spatial_features`.

    Independent implementation composed from published primitives (spatial
    autocorrelation, frequency gating, level-set spatial-chaos, hotspot concentration,
    isotope collapse); it is *not* a re-implementation of, and shares no code with, any
    vendor's feature finder.

    Pipeline:

    1. :func:`find_spatial_features` — candidate detection (mean / skyline / both),
       reproducibility (frequency) gate, Moran's-I spatial denoise, and isotope collapse
       (one compound → one feature). This yields the spatially-coherent feature set.
    2. **Artifact rejection** — score each survivor's *morphology* with the level-set
       spatial-chaos ρ (:func:`spatial_chaos`; high = organized) and its **hotspot
       concentration** (:func:`hotspot_fraction`; high = a delocalization / matrix
       artifact). A smoothly-delocalized matrix ion can *pass* Moran's I — it is spatially
       autocorrelated — yet be junk; hotspot/chaos catch what autocorrelation alone misses.
    3. **Composite quality** in ``[0, 1]`` (:func:`_composite_quality`) fuses Moran's I,
       chaos and the hotspot penalty; features below ``min_quality`` or above
       ``max_hotspot`` are dropped.

    Surviving ``peaks`` carry ``quality``, ``spatial_chaos`` and ``hotspot_fraction``
    alongside every field :func:`find_spatial_features` sets, sorted by descending
    quality and extracted **region-complete** over every pixel (the cached dense
    ``pixels × features`` matrix — no missing values). Returns a :class:`CoherentFeatures`.
    """
    sf = find_spatial_features(
        ds, snr=snr, min_rel_intensity=min_rel_intensity, max_candidates=max_candidates,
        min_frequency=min_frequency, min_morans=min_morans, tol_ppm=tol_ppm, norm=norm,
        reduce=reduce, mask=mask, projection=projection, rescue_frequency=rescue_frequency,
        collapse_isotopes=collapse_isotopes, charge=charge, iso_tol_ppm=iso_tol_ppm,
        progress=_stage(progress, 0, 70))
    params = dict(sf.params)
    params.update(min_quality=min_quality, max_hotspot=max_hotspot, stage="coherent")
    survivors = sf.peaks
    if not survivors:
        return CoherentFeatures([], sf.n_candidates, 0, 0, float("nan"), params)
    mzs = [float(p["mz"]) for p in survivors]
    chaos = spatial_chaos(ds, mzs, tol_ppm=tol_ppm, norm=norm)
    if progress is not None:
        progress(88, 100)
    hot = hotspot_fraction(ds, mzs, tol_ppm=tol_ppm, reduce=reduce)
    if progress is not None:
        progress(96, 100)
    out, widths = [], []
    for p, c, h in zip(survivors, chaos, hot):
        q = _composite_quality(p.get("morans_i", 0.0), c, h)
        if q < min_quality or (np.isfinite(h) and h > max_hotspot):
            continue
        d = dict(p)
        d["spatial_chaos"] = float(c) if np.isfinite(c) else float("nan")
        d["hotspot_fraction"] = float(h)
        d["quality"] = float(q)
        out.append(d)
        if "width_ppm" in d and np.isfinite(d["width_ppm"]):
            widths.append(float(d["width_ppm"]))
    out.sort(key=lambda d: d["quality"], reverse=True)
    suggested = float(np.median(widths)) if widths else sf.suggested_ppm
    if progress is not None:
        progress(100, 100)
    return CoherentFeatures(out, sf.n_candidates, len(survivors), len(out), suggested, params)


def _silhouette(scores, labels):
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(scores, labels))
    except Exception:  # noqa: BLE001
        return float("nan")


# --------------------------------------------------------------------------- #
# Discriminating features (one-vs-rest per region)
# --------------------------------------------------------------------------- #
def discriminating_features(ds, labels, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                            top_n: int = 15, present_frac: float = 0.5):
    """Top discriminating ions for each region in a labelling (one-vs-rest).

    For every cluster, contrasts its pixels against all others and returns the
    peaks with the highest |AUC-0.5|, as a dict ``{cluster: DataFrame}`` with
    columns ``mz, AUC, p_value, q_value, log2_fc, mean_in, mean_out, present``
    sorted by descending discrimination. AUC > 0.5 means enriched in that cluster.
    ``log2_fc`` is the signed log2 fold-change in/out (+ = enriched in the cluster,
    − = depleted, 0 = no change), pseudocount-regularized (see :func:`_fold_tau`).

    **Presence gate.** Mann–Whitney AUC is rank-based and scale-free, so an ion
    sitting at the noise floor can post a high AUC from a tiny *systematic* rank
    offset (e.g. the TIC amp-cap lifting a region's low-TIC pixels) — it would rank
    as a "top discriminator" with no real signal in the region, and image flat. The
    ``present`` column flags ions detected in at least ``present_frac`` of the
    cluster's pixels above a data-scaled noise floor (a low percentile of the nonzero
    matrix). Present ions are ranked ahead of absent ones so they fill ``top_n``
    first; absent ions are flagged, not dropped, so a region with no real markers
    still returns its (flagged) best separators.
    """
    import pandas as pd

    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    labels = np.asarray(labels)
    peaks = np.asarray(peaks, dtype=float)
    if (labels < 0).any():                       # drop unassigned pixels (e.g. ROI regions
        keep = labels >= 0                       # that don't tile the whole slide)
        X, labels = X[keep], labels[keep]
    # Data-scaled noise floor: a low percentile of the nonzero (TIC-normalized)
    # intensities, so "present" means measurably above background — not merely nonzero.
    nz = X[X > 0]
    floor = float(np.percentile(nz, 5)) if nz.size else 0.0
    out = {}
    for cl in np.unique(labels):
        inmask = labels == cl
        if inmask.sum() == 0 or (~inmask).sum() == 0:
            continue
        Xin, Xout = X[inmask], X[~inmask]
        # _auc_mwu(A, B) returns AUC high in B (its 2nd arg), so pass the cluster
        # ("in") group second → AUC > 0.5 means enriched in the cluster, matching
        # the docstring and the log2_fc direction below.
        auc, p = _auc_mwu(Xout, Xin)
        mi, mo = Xin.mean(0), Xout.mean(0)
        present = (Xin > floor).mean(0) >= present_frac   # detected across the region
        # signed log2 fold-change with a data-scaled pseudocount (the same noise floor)
        # so a pair of sub-floor means gives log2 FC ≈ 0 rather than a spurious ratio
        # (cf. KNOWN_ISSUES.md artifact #2); + = enriched in the cluster, − = depleted.
        tau = floor if floor > 0 else 1.0
        log2_fc = np.log2((mi + tau) / (mo + tau))
        df = pd.DataFrame({"mz": peaks, "AUC": auc, "p_value": p, "q_value": _bh_fdr(p),
                           "log2_fc": log2_fc, "mean_in": mi, "mean_out": mo,
                           "present": present})
        # Present ions first, then by discrimination, so noise-floor ions can't crowd
        # real markers out of the top-N (they remain, flagged, only as filler).
        df["_disc"] = df["AUC"].sub(0.5).abs()
        df = df.sort_values(["present", "_disc"], ascending=[False, False], kind="stable")
        df = df.drop(columns="_disc").head(top_n).reset_index(drop=True)
        df.attrs["n_in"], df.attrs["n_out"] = int(inmask.sum()), int((~inmask).sum())
        out[int(cl)] = df
    return out


def shrunken_centroids(ds, labels, peaks, shrink: float = 2.0, tol_ppm: float = DEFAULT_TOL_PPM,
                       norm: str = "tic", top_n: int = 15):
    """Nearest-shrunken-centroid feature selection per region (Cardinal's SSC stance).

    For each segment, the per-feature class centroid is expressed as a t-like
    statistic vs the grand centroid (pooled within-class SD + a median offset ``s0``),
    then **soft-thresholded** by ``shrink``: features with ``|t| <= shrink`` collapse to
    zero and drop out, so each segment keeps only the ions that actually distinguish it —
    automatic discriminating-m/z selection that gets sparser as ``shrink`` grows.

    Returns ``{cluster: DataFrame(mz, statistic, shrunken, mean_in, mean_out)}`` sorted by
    ``|shrunken|`` (only non-zero, i.e. retained, ions), plus ``df.attrs['n_selected']``.

    Refs: nearest shrunken centroids — Tibshirani et al. (2002), PNAS 99(10):6567,
    doi:10.1073/pnas.082099299; spatial shrunken centroids — Bemis et al. (2016),
    doi:10.1074/mcp.O115.053918.
    """
    import pandas as pd

    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    labels = np.asarray(labels)
    peaks = np.asarray(peaks, dtype=float)
    if (labels < 0).any():
        keep = labels >= 0
        X, labels = X[keep], labels[keep]
    n, p = X.shape
    grand = X.mean(0)
    classes = np.unique(labels)
    # pooled within-class standard error per feature + s0 offset (median, like PAM)
    ss = np.zeros(p)
    for cl in classes:
        Xc = X[labels == cl]
        if Xc.shape[0] > 1:
            ss += ((Xc - Xc.mean(0)) ** 2).sum(0)
    sj = np.sqrt(ss / max(1, n - len(classes)))
    s0 = float(np.median(sj))
    out = {}
    for cl in classes:
        inmask = labels == cl
        nk = int(inmask.sum())
        if nk == 0 or nk == n:
            continue
        mk = np.sqrt(1.0 / nk - 1.0 / n) if nk < n else 0.0
        denom = mk * (sj + s0)
        d = np.divide(X[inmask].mean(0) - grand, denom, out=np.zeros(p), where=denom > 0)
        shr = np.sign(d) * np.clip(np.abs(d) - shrink, 0.0, None)   # soft-threshold
        sel = shr != 0
        df = pd.DataFrame({"mz": peaks[sel], "statistic": d[sel], "shrunken": shr[sel],
                           "mean_in": X[inmask].mean(0)[sel], "mean_out": X[~inmask].mean(0)[sel]})
        df = df.reindex(df["shrunken"].abs().sort_values(ascending=False).index).head(top_n)
        df = df.reset_index(drop=True)
        df.attrs["n_selected"] = int(sel.sum())
        df.attrs["shrink"] = float(shrink)
        out[int(cl)] = df
    return out


def roi_localization(ds, mask_in, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic"):
    """How confined each ion is to a region, vs. the rest of the *on-tissue* slide.

    Picking peaks inside an ROI only tells you what is *present* there; it never
    asks whether an ion is actually *higher* inside than out. This contrasts the
    ROI's pixels against every other on-tissue pixel and returns, per peak:

    * ``roi_auc``  — rank-based ROC AUC; > 0.5 means higher inside the ROI
      (0.5 = no spatial preference, ~1.0 = strongly confined to the ROI).
    * ``roi_log2_fc`` — signed log2 fold-change inside vs. outside (+ = enriched
      inside, − = depleted, 0 = no change), pseudocount-regularized.

    'On-tissue' excludes empty background (TIC below 5% of the median on-tissue
    TIC) so the contrast isn't inflated by void pixels — comparing an ROI to bare
    slide would make almost everything look enriched. Returns a DataFrame with
    columns ``mz, roi_auc, roi_log2_fc, mean_in, mean_out`` (one row per peak); the
    statistic columns are NaN if the ROI or the remaining tissue is empty.
    """
    import pandas as pd

    # accept peak dicts ({"mz": ...}) or bare m/z floats, like build_feature_list
    peaks = np.asarray([p["mz"] if isinstance(p, dict) else p for p in peaks],
                       dtype=float)
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    mask_in = np.asarray(mask_in, dtype=bool)
    tic = np.asarray(ds.tic(), dtype=float)
    pos = tic[tic > 0]
    tissue = tic > (0.05 * float(np.median(pos)) if len(pos) else 0.0)
    inside = mask_in & tissue
    outside = (~mask_in) & tissue
    if int(inside.sum()) == 0 or int(outside.sum()) == 0:
        nan = np.full(len(peaks), np.nan)
        return pd.DataFrame({"mz": peaks, "roi_auc": nan, "roi_log2_fc": nan,
                             "mean_in": nan, "mean_out": nan})
    Xin, Xout = X[inside], X[outside]
    auc, _ = _auc_mwu(Xout, Xin)             # AUC > 0.5 ⇒ enriched inside the ROI
    mi, mo = Xin.mean(0), Xout.mean(0)
    # signed log2 fold-change with a data-scaled pseudocount = 5th percentile of the
    # nonzero on-tissue intensities, so two near-noise means give log2 FC ≈ 0 instead of
    # exploding (KNOWN_ISSUES.md #2); + = enriched inside the ROI, − = depleted.
    tau = _fold_tau(np.vstack([Xin, Xout]))
    roi_log2 = np.log2((mi + tau) / (mo + tau))
    df = pd.DataFrame({"mz": peaks, "roi_auc": auc, "roi_log2_fc": roi_log2,
                       "mean_in": mi, "mean_out": mo})
    df.attrs["n_in"], df.attrs["n_out"] = int(inside.sum()), int(outside.sum())
    return df


# --------------------------------------------------------------------------- #
# Co-localization
# --------------------------------------------------------------------------- #
def _foreground_mask(ds):
    """Union of detected tissue samples → an on-tissue pixel mask, or ``None`` if no
    tissue is found (caller then falls back to all pixels).

    Used to restrict co-localization away from the off-tissue background: a large shared
    block of (0, 0) pixels otherwise pulls both ion vectors' means to zero and inflates
    Pearson *r* toward a spurious positive (two uncorrelated on-tissue images can read
    r ≈ +0.65 once padded with background). Cosine/Manders/Dice are invariant to shared
    zeros, but masking is harmless for them and fixes the correlation measures."""
    try:
        masks = detect_samples(ds)
    except Exception:
        return None
    if not masks:
        return None
    fg = np.zeros(ds.n_pixels, dtype=bool)
    for m in masks:
        fg |= np.asarray(m, dtype=bool)
    return fg if fg.any() else None


def colocalize(ds, target_mz: float, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
               method: str = "pearson", mask=None, *, embedding=None):
    """Rank ``peaks`` by spatial similarity of their ion image to ``target_mz``.
    Returns a list of ``{mz, score}`` sorted by descending similarity.

    ``mask`` (a boolean per-pixel array) restricts the comparison to a region's pixels,
    so the ranking reflects co-localization *within* that region rather than over the
    whole slide. When ``mask`` is None the comparison is restricted to detected on-tissue
    pixels (see :func:`_foreground_mask`) so off-tissue background does not inflate the
    correlation; pass an explicit mask to override.

    ``method="learned"`` ranks by cosine in a **learned** ion-embedding space instead of
    a raw-pixel statistic: pass the trained ``embedding`` (a
    :class:`smile_msi.ionembed.IonEmbedding` from
    :func:`smile_msi.ionembed.train_ion_encoder`) and the call delegates to
    :func:`smile_msi.ionembed.colocalize_learned`. ``mask`` is ignored in this path (the
    embedding already encodes the whole-slide distribution). The default stays
    ``"pearson"``; ``"learned"`` is opt-in and needs the optional ``torch`` extra at
    train time (not here). Raises :class:`ValueError` if ``embedding is None``.

    Refs: MSI co-localization — Ovchinnikova et al. (2020), doi:10.1093/bioinformatics/btaa085.
    """
    if method == "learned":
        if embedding is None:
            raise ValueError(
                "learned colocalization needs a trained ion embedding — train one first")
        from . import ionembed
        return ionembed.colocalize_learned(ds, target_mz, peaks, embedding)
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    peaks = np.asarray(peaks, dtype=float)
    j = int(np.argmin(np.abs(peaks - target_mz)))
    if abs(peaks[j] - target_mz) <= target_mz * tol_ppm / 1e6:
        target = X[:, j]
    else:
        target = ds.ion_vector(target_mz, tol_ppm=tol_ppm, norm=norm)
    if mask is None:
        mask = _foreground_mask(ds)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        X, target = X[mask], np.asarray(target)[mask]
    out = [{"mz": float(peaks[c]), "score": float(_similarity(target, X[:, c], method))}
           for c in range(X.shape[1])]
    out.sort(key=lambda d: (np.isnan(d["score"]), -d["score"]))
    return out


def _similarity(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Spatial similarity of two ion images. Measures mirror Cardinal's
    ``colocalized()``: ``pearson`` (cor), ``cosine``, ``moc`` (Manders overlap),
    ``m1``/``m2`` (Manders colocalization coefficients), ``dice`` (overlap of the
    at-or-above-median footprints). Manders/Dice threshold each image at the median of its
    positive pixels (Costes 2004-style auto-threshold; Manders 1993 used a zero/presence
    threshold) — the fraction of one ion's signal that falls where the other is present,
    which Pearson (a global linear fit) cannot express.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if method == "cosine":
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan
    if method == "moc":                                  # Manders overlap coefficient
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / denom) if denom > 0 else np.nan
    if method in ("m1", "m2", "dice"):
        # Footprint = at-or-above the median of positive pixels. Use ">=" (not ">") so a
        # binary/uniform ion — whose positive pixels all equal that median — keeps a
        # non-empty footprint instead of collapsing to empty (which reported M2/Dice =
        # 0/NaN even for perfectly co-localized ions). An all-zero image gets threshold
        # +inf → empty footprint; positive thresholds are > 0, so zeros stay excluded.
        ta = np.median(a[a > 0]) if np.any(a > 0) else np.inf
        tb = np.median(b[b > 0]) if np.any(b > 0) else np.inf
        ma, mb = a >= ta, b >= tb
        if method == "m1":                               # frac of a's signal where b present
            return float(a[mb].sum() / a.sum()) if a.sum() > 0 else np.nan
        if method == "m2":                               # frac of b's signal where a present
            return float(b[ma].sum() / b.sum()) if b.sum() > 0 else np.nan
        inter, sz = int((ma & mb).sum()), int(ma.sum() + mb.sum())
        return float(2 * inter / sz) if sz > 0 else np.nan
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Exact two-region statistics (rank-based ROC AUC + Mann-Whitney)
# --------------------------------------------------------------------------- #
def sample_labels(ds, rel_threshold: float = 0.10, min_frac: float = 0.004,
                  smooth: float = 1.0) -> np.ndarray:
    """Per-pixel replicate IDs from :func:`detect_samples` (−1 = off-tissue / unassigned).

    A convenience for the sample-summarized statistics: each detected tissue piece
    becomes a distinct integer ID so :func:`roi_comparison` can treat the pieces as
    independent replicates instead of pooling correlated pixels."""
    masks = detect_samples(ds, rel_threshold=rel_threshold, min_frac=min_frac, smooth=smooth)
    out = np.full(ds.n_pixels, -1, dtype=int)
    for sid, m in enumerate(masks):
        out[np.asarray(m, dtype=bool)] = sid
    return out


def _summarize_by_sample(X, group, samples):
    """Mean of each replicate's in-group pixels → (n_samples, n_peaks) matrix.

    Only samples with at least one in-group pixel contribute a row."""
    rows, ids = [], []
    for sid in np.unique(samples[group]):
        if sid < 0:
            continue
        sel = group & (samples == sid)
        if sel.any():
            rows.append(X[sel].mean(0))
            ids.append(int(sid))
    return (np.vstack(rows) if rows else np.empty((0, X.shape[1]))), ids


def roi_comparison(ds, mask_a, mask_b, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                   a_label: str = "Group A", b_label: str = "Group B", samples=None,
                   method: str = "mwu", effect: str = "match"):
    """Per-peak **exact** discrimination between two pixel regions.

    Returns a pandas DataFrame (one row per peak) with ``mz, mean_A, sd_A, mean_B,
    sd_B, AUC, log2_fc, p_value, q_value``. ``AUC`` is the exact rank-based ROC
    AUC (= Mann-Whitney U / n_a / n_b); > 0.5 means higher in region B. ``log2_fc``
    is the signed log2 fold-change (B over A): + = higher in B, − = higher in A, 0 =
    no change. Column names match :func:`smile_msi.pipeline.annotate_df` / ``build_report``.

    ``method`` picks the per-peak significance test (``AUC`` is always the rank-based
    effect size regardless): ``'mwu'`` (default) = exact rank-based Mann-Whitney U,
    the non-parametric default that's robust to MSI's skewed pixel intensities;
    ``'welch'`` = Welch's two-sample t-test (unequal variance); ``'student'`` =
    Student's pooled-variance t-test. The parametric tests assume roughly normal
    per-unit values, so prefer them on sample-summarized replicates rather than raw
    pixels. The chosen test is recorded in ``df.attrs['test']``.

    **Unit of replication.** By default every *pixel* is a unit. Thousands of
    spatially-correlated pixels are **not** independent replicates, so per-pixel
    p-values are wildly overconfident (pseudoreplication) — a standard reviewer
    objection to MSI statistics. Pass ``samples`` — a per-pixel integer array of
    biological/technical replicate IDs (e.g. from :func:`sample_labels`) — to instead
    summarize each replicate to one value per peak and test **across replicates**,
    the per-sample-summary stance Cardinal's ``meansTest`` takes. The result then
    carries ``df.attrs['unit'] == 'sample'`` with ``n_a``/``n_b`` = replicate counts,
    and a ``df.attrs['warning']`` when a group has < 2 replicates (underpowered).

    Pass ``effect='pixel'`` to keep the ``AUC``, means and ``log2_fc`` on the
    per-pixel scale (the rank-based discrimination effect size, consistent with a
    per-pixel ROC) while the ``p_value``/``q_value`` still summarize across
    ``samples`` — recorded as ``df.attrs['effect_unit'] == 'pixel'``. The default
    ``effect='match'`` computes every column on the unit being tested.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if int(a.sum()) == 0 or int(b.sum()) == 0:
        raise ValueError("both regions must contain at least one pixel")

    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    return _two_group_table(X, a, b, "mz", np.asarray(peaks, dtype=float),
                            a_label, b_label, samples, method, effect=effect)


def _two_group_table(X, a, b, id_col, ids, a_label, b_label, samples, method, effect="match"):
    """Build the A-vs-B per-feature statistics table over a ``(pixels × features)`` matrix.

    ``a``/``b`` are boolean pixel masks; ``id_col``/``ids`` name the per-feature identifier
    column (``'mz'`` for ions, ``'class'`` for lipid classes). Computes the exact rank-based
    AUC (> 0.5 ⇒ higher in B), the chosen significance test, per-group means/SDs, signed
    log2 fold-change and BH q-values, and stamps the ``df.attrs`` :func:`roi_comparison`
    documents. Honours the
    ``samples`` pseudoreplication summary when given. Shared by :func:`roi_comparison` and
    :func:`class_comparison` so both report identical statistics."""
    import pandas as pd

    if samples is not None:
        samples = np.asarray(samples, dtype=int)
        XA, ids_a = _summarize_by_sample(X, a, samples)
        XB, ids_b = _summarize_by_sample(X, b, samples)
        unit = "sample"
    else:
        XA, XB = X[a], X[b]
        ids_a, ids_b = None, None
        unit = "pixel"
    na, nb = XA.shape[0], XB.shape[0]         # units driving the p-value inference
    if na == 0 or nb == 0:
        raise ValueError("each region must contain at least one replicate with pixels")

    # Effect-size / descriptive arrays. ``effect="pixel"`` keeps the AUC, means, SDs and
    # fold change on the per-pixel scale — the rank-based discrimination effect size, and
    # exactly what a per-pixel ROC panel draws — even when the p-value summarizes to
    # replicates (Cardinal's meansTest pseudoreplication stance). ``effect="match"``
    # (default, legacy) computes every column on the same arrays the test runs on, so the
    # existing callers are unchanged.
    decouple = effect == "pixel" and samples is not None
    XE_a, XE_b = (X[a], X[b]) if decouple else (XA, XB)

    auc, mwu_p = _auc_mwu(XA, XB)             # rank-based AUC + MWU p on the inference units
    if decouple:
        auc, _ = _auc_mwu(XE_a, XE_b)        # AUC stays a per-pixel effect size
    pval, test_name = _two_group_p(XA, XB, method, mwu_p)
    mA, mB = XE_a.mean(0), XE_b.mean(0)
    sA = XE_a.std(0, ddof=1) if XE_a.shape[0] > 1 else np.zeros(X.shape[1])
    sB = XE_b.std(0, ddof=1) if XE_b.shape[0] > 1 else np.zeros(X.shape[1])
    # Signed log2 fold-change (B over A, matching AUC's ">0.5 ⇒ higher in B" convention
    # and the cohort log2_fc) with a data-scaled pseudocount τ = 5th percentile of the
    # nonzero intensities — so two near-noise means give log2 FC ≈ 0 instead of a spurious
    # ratio (KNOWN_ISSUES.md artifact #2). The ``present`` flag marks features above the
    # floor in either group, so a large |log2 FC| on a pair of sub-floor (effectively
    # absent) means can be discounted.
    tau = _fold_tau(X)
    log2_fc = np.log2((mB + tau) / (mA + tau))
    present = (mA > tau) | (mB > tau)
    df = pd.DataFrame({
        id_col: ids,
        "mean_A": mA, "sd_A": sA, "mean_B": mB, "sd_B": sB,
        "AUC": auc, "log2_fc": log2_fc, "p_value": pval, "q_value": _bh_fdr(pval),
        "present": present,
    })
    df.attrs["n_a"], df.attrs["n_b"] = na, nb
    df.attrs["a_label"], df.attrs["b_label"] = a_label, b_label
    df.attrs["unit"] = unit
    df.attrs["auc_method"] = "exact rank-based (Mann-Whitney)"
    df.attrs["test"] = test_name
    if decouple:
        df.attrs["effect_unit"] = "pixel"    # AUC / means / fold per-pixel; p / q across replicates
    if unit == "sample":
        df.attrs["samples_a"], df.attrs["samples_b"] = ids_a, ids_b
        if na < 2 or nb < 2:
            df.attrs["warning"] = (f"only {na} vs {nb} replicate(s) — the across-sample "
                                   "test is underpowered; collect more tissue replicates")
    return df


def _auc_mwu(XA: np.ndarray, XB: np.ndarray):
    """Rank-based AUC (high in B) + two-sided Mann-Whitney U p per feature column.

    ``AUC = U_B/(n_a·n_b)`` is the **exact** rank-based effect size regardless of n
    (Bamber 1975; Hanley & McNeil 1982). The **p-value** is computed per column via
    ``scipy.stats.mannwhitneyu(method='auto')`` — the exact permutation null when one
    sample is small and tie-free (e.g. the sample-summarized replicate path, where the
    old asymptotic normal approximation was anti-conservative: 3-vs-3 gave p=0.0495 vs an
    exact 0.10), and the tie-corrected normal approximation **with continuity correction**
    otherwise. Refs: Mann & Whitney (1947), doi:10.1214/aoms/1177730491.
    """
    from scipy.stats import mannwhitneyu

    na, nb = XA.shape[0], XB.shape[0]
    ncol = XA.shape[1] if XA.ndim == 2 else XB.shape[1]
    auc = np.full(ncol, np.nan)
    pval = np.ones(ncol)
    if na == 0 or nb == 0:
        return auc, pval
    for j in range(ncol):
        a, b = XA[:, j], XB[:, j]
        try:                                                   # U for group B → AUC high in B
            res = mannwhitneyu(b, a, alternative="two-sided", method="auto",
                               use_continuity=True)
            ub, p = float(res.statistic), float(res.pvalue)
        except ValueError:                                     # e.g. an all-NaN column
            ub, p = float("nan"), 1.0
        auc[j] = ub / (na * nb)
        pval[j] = p if np.isfinite(p) else 1.0
    return auc, pval


def _ttest_p(XA: np.ndarray, XB: np.ndarray, equal_var: bool) -> np.ndarray:
    """Two-sided independent two-sample t-test p per feature column.

    ``equal_var=False`` is Welch's t (Welch 1947, doi:10.1093/biomet/34.1-2.28);
    ``True`` is Student's pooled-variance t (Student 1908). Degenerate columns
    (a group with < 2 values, or zero variance both sides) yield p = 1.
    """
    from scipy.stats import ttest_ind

    if XA.shape[0] < 2 or XB.shape[0] < 2:
        return np.ones(XA.shape[1])
    p = np.asarray(ttest_ind(XA, XB, axis=0, equal_var=equal_var).pvalue, dtype=float)
    return np.where(np.isfinite(p), p, 1.0)


def _two_group_p(XA: np.ndarray, XB: np.ndarray, method: str, mwu_p: np.ndarray):
    """Resolve ``method`` to a per-column p-value vector + a human-readable test name.

    Shared by :func:`roi_comparison` (and reusable elsewhere): ``mwu_p`` is the
    already-computed Mann-Whitney p so the rank path costs nothing extra.
    """
    m = (method or "mwu").lower()
    if m in ("mwu", "mannwhitney", "mann-whitney", "rank"):
        return mwu_p, "exact rank-based AUC; Mann-Whitney U p (exact for small n, asymptotic otherwise)"
    if m in ("welch", "t", "ttest", "t-test"):
        return _ttest_p(XA, XB, equal_var=False), "Welch's t-test (unequal variance)"
    if m in ("student", "ttest_equal", "pooled"):
        return _ttest_p(XA, XB, equal_var=True), "Student's t-test (pooled variance)"
    raise ValueError(f"unknown method {method!r}; use 'mwu', 'welch', or 'student'")


def roc_curve(values_a, values_b):
    """ROC curve for a single ion used as a classifier of region **B** (positive) vs
    region **A** (negative), with the ion's per-pixel intensity as the score.

    Sweeping an intensity threshold from high to low traces the true-positive rate
    (fraction of B pixels above it) against the false-positive rate (fraction of A
    pixels above it). Returns ``(fpr, tpr, auc)`` with the vertices already ordered for
    plotting. The diagonal ``fpr == tpr`` is the no-discrimination chance line; the
    curve bows toward the top-left when the ion is enriched in B (``auc > 0.5``) and
    toward the bottom-right when enriched in A (``auc < 0.5``) — so ``auc`` equals the
    rank-based AUC :func:`roi_comparison` reports (> 0.5 ⇒ higher in B).

    Ref: Hanley & McNeil (1982), doi:10.1148/radiology.143.1.7063747.
    """
    a = np.asarray(values_a, dtype=float).ravel()
    b = np.asarray(values_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    scores = np.concatenate([b, a])
    pos = np.concatenate([np.ones(b.size, bool), np.zeros(a.size, bool)])
    order = np.argsort(-scores, kind="mergesort")          # highest score first
    scores, pos = scores[order], pos[order]
    tps = np.cumsum(pos)
    fps = np.cumsum(~pos)
    # one ROC vertex at the end of each run of equal scores, so tied pixels don't draw a
    # spurious staircase (correct tie handling, matching sklearn's roc_curve).
    last = np.r_[np.where(np.diff(scores))[0], scores.size - 1]
    tpr = np.r_[0.0, tps[last] / tps[-1]]
    fpr = np.r_[0.0, fps[last] / fps[-1]]
    # np.trapz was renamed to np.trapezoid in numpy 2.0 and *removed* in 2.4 — don't reference
    # np.trapz unless it's actually there, or merely importing this on numpy ≥2.4 raises.
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values).

    Reference: Benjamini & Hochberg (1995), J. R. Stat. Soc. B 57(1):289–300,
    doi:10.1111/j.2517-6161.1995.tb02031.x.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


# --------------------------------------------------------------------------- #
# Lipid-class roll-ups (compare whole classes, not individual ions)
# --------------------------------------------------------------------------- #
def _aggregate_by_class(X, classes):
    """Collapse a ``(pixels × ions)`` matrix to ``(pixels × classes)`` by summing each lipid
    class's ion columns. ``classes[j]`` is ion ``j``'s class label (``''``/``None`` =
    unannotated, dropped). Returns ``(Xc, names, n_ions)`` where ``names`` are the sorted
    classes kept and ``n_ions[k]`` is how many ions fed class ``names[k]``."""
    classes = ["" if c is None else str(c) for c in classes]
    if X.shape[1] != len(classes):
        raise ValueError(f"classes ({len(classes)}) must be parallel to peaks ({X.shape[1]})")
    names = sorted({c for c in classes if c})
    if not names:
        raise ValueError("no annotated lipids to roll up — annotate the peaks (or widen the "
                         "identification tolerance) first")
    cols, n_ions = [], []
    for c in names:
        idx = [j for j, cc in enumerate(classes) if cc == c]
        cols.append(X[:, idx].sum(axis=1))
        n_ions.append(len(idx))
    return np.column_stack(cols), names, n_ions


def class_comparison(ds, mask_a, mask_b, peaks, classes, tol_ppm: float = DEFAULT_TOL_PPM,
                     norm: str = "tic", a_label: str = "Group A", b_label: str = "Group B",
                     samples=None, method: str = "mwu"):
    """A-vs-B comparison rolled up to whole lipid **classes** instead of individual ions.

    Each class's ion intensities are summed per pixel, then the same exact rank-based AUC
    (> 0.5 ⇒ higher in B), signed log2 fold-change and significance test as
    :func:`roi_comparison` run on those per-class totals. ``classes`` is a per-peak list of
    class labels (parallel to ``peaks``; ``''`` = unannotated, excluded) — e.g. from
    :meth:`smile_msi.match.Annotator.classes_for`. Returns a DataFrame ``class, n_ions,
    mean_A, sd_A, mean_B, sd_B, AUC, log2_fc, p_value, q_value`` carrying the same
    ``df.attrs`` (labels, unit, test,
    pseudoreplication ``samples``) :func:`roi_comparison` documents, plus
    ``df.attrs['kind'] = 'class'``."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if int(a.sum()) == 0 or int(b.sum()) == 0:
        raise ValueError("both regions must contain at least one pixel")
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    Xc, names, n_ions = _aggregate_by_class(X, classes)
    df = _two_group_table(Xc, a, b, "class", np.asarray(names, dtype=object),
                          a_label, b_label, samples, method)
    df.insert(1, "n_ions", n_ions)
    df.attrs["kind"] = "class"
    return df


def class_composition(ds, masks, peaks, classes, names=None, tol_ppm: float = DEFAULT_TOL_PPM,
                      norm: str = "tic"):
    """Lipid-class **composition** of one or more regions: each region's share of total
    annotated-lipid signal contributed by each class.

    For every region mask, each class's ion intensities are summed per pixel and averaged
    over the region's pixels, then expressed as a percentage of that region's annotated-lipid
    total (each region column sums to 100%). ``classes`` is the per-peak class list (as in
    :func:`class_comparison`). Returns a percentage DataFrame indexed by class with one column
    per region; ``df.attrs['intensity']`` holds the matching mean-intensity table,
    ``df.attrs['n_ions']`` the ion count per class, and ``df.attrs['regions']`` the kept region
    names (regions with no pixels are dropped). Region names must be unique."""
    import pandas as pd

    masks = [np.asarray(m, dtype=bool) for m in masks]
    names = list(names) if names is not None else [f"region {i}" for i in range(len(masks))]
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    Xc, cls_names, n_ions = _aggregate_by_class(X, classes)
    pct, inten, kept = {}, {}, []
    for nm, m in zip(names, masks):
        if not m.any():
            continue
        mean_per_class = Xc[m].mean(0)
        total = float(mean_per_class.sum())
        inten[nm] = mean_per_class
        pct[nm] = 100.0 * mean_per_class / total if total > 0 else np.zeros_like(mean_per_class)
        kept.append(nm)
    if not kept:
        raise ValueError("no region contained any pixels")
    comp = pd.DataFrame(pct, index=cls_names, columns=kept)
    comp.index.name = "class"
    comp.attrs["intensity"] = pd.DataFrame(inten, index=cls_names, columns=kept)
    comp.attrs["n_ions"] = dict(zip(cls_names, n_ions))
    comp.attrs["regions"] = kept
    return comp


# --------------------------------------------------------------------------- #
# Multi-group statistics (>2 regions) and co-localization matrix
# --------------------------------------------------------------------------- #
def multigroup_features(ds, labels, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                        method: str = "kruskal", samples=None, names=None):
    """Per-peak test across all regions in ``labels``.

    ``method='kruskal'`` (default) runs the non-parametric Kruskal-Wallis H test and
    the statistic column is named ``H``; ``method='anova'`` runs a one-way ANOVA and
    the column is named ``F`` (parametric — assumes roughly normal, equal-variance
    groups). Returns a DataFrame ``mz, <stat>, p_value, q_value, top_region`` (the
    region with the highest mean for that peak), sorted by ascending p — the ions
    whose spatial abundance differs most between regions. The test name is recorded
    in ``df.attrs['test']``.

    Pass ``names`` — the group name aligned to each label id — to label ``top_region``
    with the readable group name (e.g. 'Synk') instead of the raw integer cluster id; a
    label id outside ``names`` falls back to ``'region <n>'``. When ``names`` is None the
    column keeps the raw integer ids.

    **Unit of replication.** By default every *pixel* is a unit, but MSI pixels are
    strongly spatially autocorrelated and are **not** independent replicates, so the
    per-pixel p/q are anti-conservative (pseudoreplication) and ``df.attrs['warning']``
    says so. Pass ``samples`` — a per-pixel integer array of replicate IDs (e.g. from
    :func:`sample_labels`) — to summarize each region to one value per replicate and
    test **across replicates** (the per-sample stance of Cardinal's ``meansTest``); the
    result then carries ``df.attrs['unit'] == 'sample'`` and a warning when any group has
    < 2 replicates. ``df.attrs['unit']`` and ``df.attrs['group_sizes']`` record the unit.
    """
    import pandas as pd

    m = (method or "kruskal").lower()
    if m in ("kruskal", "kw", "kruskal-wallis"):
        from scipy.stats import kruskal as _test
        stat_name, test_name = "H", "Kruskal-Wallis H"
    elif m in ("anova", "f_oneway", "f-oneway", "oneway", "one-way"):
        from scipy.stats import f_oneway as _test
        stat_name, test_name = "F", "one-way ANOVA F"
    else:
        raise ValueError(f"unknown method {method!r}; use 'kruskal' or 'anova'")

    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    labels = np.asarray(labels)
    samples = None if samples is None else np.asarray(samples, dtype=int)
    if (labels < 0).any():                       # drop unassigned pixels (ROI regions)
        keep = labels >= 0
        X, labels = X[keep], labels[keep]
        if samples is not None:
            samples = samples[keep]
    uniq = np.unique(labels)

    if samples is not None:
        # summarize each region to one row per replicate → test across replicates
        unit = "sample"
        mats = [_summarize_by_sample(X, labels == c, samples)[0] for c in uniq]
        sizes = {int(c): int(mt.shape[0]) for c, mt in zip(uniq, mats)}
        underpowered = any(mt.shape[0] < 2 for mt in mats)

        def _vals(j):
            return [mt[:, j] for mt in mats]
    else:
        unit = "pixel"
        groups = [np.flatnonzero(labels == c) for c in uniq]
        sizes = {int(c): int(len(g)) for c, g in zip(uniq, groups)}
        underpowered = False

        def _vals(j):
            return [X[g, j] for g in groups]

    stat = np.empty(X.shape[1]); pval = np.empty(X.shape[1]); topr = np.empty(X.shape[1], dtype=int)
    for j in range(X.shape[1]):
        vals = _vals(j)
        try:
            s, p = _test(*vals)
        except ValueError:                       # identical values / empty group -> no variation
            s, p = 0.0, 1.0
        stat[j] = s if np.isfinite(s) else 0.0
        pval[j] = p if np.isfinite(p) else 1.0
        means = [v.mean() if len(v) else -np.inf for v in vals]
        topr[j] = uniq[int(np.argmax(means))]
    if names is not None:                        # readable group name, not the raw cluster id
        names = list(names)
        top_col = [names[int(t)] if 0 <= int(t) < len(names) else f"region {int(t)}"
                   for t in topr]
    else:
        top_col = topr
    df = pd.DataFrame({"mz": np.asarray(peaks, float), stat_name: stat, "p_value": pval,
                       "q_value": _bh_fdr(pval), "top_region": top_col})
    df = df.sort_values("p_value").reset_index(drop=True)
    df.attrs["group_sizes"] = sizes
    df.attrs["unit"] = unit
    df.attrs["test"] = test_name
    if unit == "pixel":
        df.attrs["warning"] = ("per-pixel test: spatially-correlated pixels are not "
                               "independent replicates, so p/q are anti-conservative "
                               "(pseudoreplication) — pass samples= to test across replicates")
    elif underpowered:
        df.attrs["warning"] = ("a group has < 2 replicates — the across-sample test is "
                               "underpowered; collect more tissue replicates")
    return df


# --------------------------------------------------------------------------- #
# Distinct & shared features across N regions (Venn compartments)
# --------------------------------------------------------------------------- #
def region_membership(ds, masks, peaks, names=None, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                      min_prevalence: float = 0.5, detect_quantile: float = 0.75,
                      max_q: float | None = None, samples=None):
    """Partition ``peaks`` into Venn compartments across two or more pixel regions —
    which ions are **distinct** to one region and which are **shared**.

    A peak is *present* in a region when it is detected (normalized intensity above
    its per-peak global detection floor) in at least ``min_prevalence`` of that
    region's pixels. The floor is the ``detect_quantile`` quantile of the peak's
    intensity over all pixels (0.75 = the slide-wide top quartile / hotspot; 0.5 =
    above the median; 0 = any positive signal). Each peak is then assigned to the
    compartment given by the **exact set**
    of regions that contain it (standard Venn semantics): present in one region only
    = distinct to it; present in several = shared by exactly those.

    When ``max_q`` is given, peaks are first restricted to those that differ
    significantly across the regions (Kruskal-Wallis BH-FDR ``q < max_q``) — the
    optional statistical gate layered on top of the prevalence rule. Pass ``samples``
    (a per-pixel integer replicate-ID array) to compute that gate **across replicates**
    rather than across pseudoreplicated pixels; without it the gate is per-pixel and
    therefore anti-conservative.

    ``masks`` is a list of boolean per-pixel arrays; ``names`` labels them (defaults
    A, B, C…). Returns a dict::

        {names, n_pixels, compartments, peaks}

    where ``compartments`` is ``[{key, regions, label, exclusive, mzs, count}]`` (key
    is the tuple of region indices) ordered distinct-first, and ``peaks`` is a
    DataFrame with per-region prevalence, the assigned compartment, and (if gated) q.
    """
    import pandas as pd

    masks = [np.asarray(m, dtype=bool) for m in masks]
    if len(masks) < 2:
        raise ValueError("need at least two regions to find distinct/shared features")
    n_reg = len(masks)
    if names is None:
        names = [chr(ord("A") + i) if i < 26 else f"R{i + 1}" for i in range(n_reg)]
    names = [str(x) for x in names]
    peaks = np.asarray(peaks, dtype=float)

    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    npk = X.shape[1]
    q = float(np.clip(detect_quantile, 0.0, 1.0))
    floor = np.quantile(X, q, axis=0) if q > 0 else np.zeros(npk)
    on = X > floor[None, :]                                       # per-pixel "detected" map
    prevalence = np.vstack([on[m].mean(0) if m.any() else np.zeros(npk) for m in masks])
    present = prevalence >= float(min_prevalence)

    qvals = np.full(npk, np.nan)
    if max_q is not None:
        from scipy.stats import kruskal
        pvals = np.ones(npk)
        live = [m for m in masks if m.any()]
        samp = None if samples is None else np.asarray(samples, dtype=int)
        # summarize each region to one value per replicate so the gate tests across
        # replicates, not pseudoreplicated pixels
        summ = ([_summarize_by_sample(X, m, samp)[0] for m in live]
                if samp is not None else None)
        for j in range(npk):
            if len(live) < 2:
                continue
            try:
                if summ is not None:
                    cols = [S[:, j] for S in summ if S.shape[0] > 0]
                    if len(cols) >= 2:
                        _, pvals[j] = kruskal(*cols)
                else:
                    _, pvals[j] = kruskal(*[X[m, j] for m in live])
            except ValueError:                                   # identical values
                pvals[j] = 1.0
        qvals = _bh_fdr(pvals)
        sig = qvals < float(max_q)
    else:
        sig = np.ones(npk, dtype=bool)

    def _label(members):
        return (f"{names[members[0]]} only" if len(members) == 1
                else " ∩ ".join(names[i] for i in members))

    comp_map: dict[tuple, list] = {}
    rows = []
    for j in range(npk):
        members = tuple(i for i in range(n_reg) if present[i, j])
        assigned = bool(members) and bool(sig[j])
        if assigned:
            comp_map.setdefault(members, []).append(float(peaks[j]))
        row = {"mz": float(peaks[j]), "compartment": _label(members) if assigned else "",
               "n_regions": len(members) if assigned else 0}
        for i in range(n_reg):
            row[f"prev_{names[i]}"] = round(float(prevalence[i, j]), 3)
        if max_q is not None:
            row["q"] = round(float(qvals[j]), 4)
        rows.append(row)

    compartments = [{"key": members, "regions": [names[i] for i in members],
                     "label": _label(members), "exclusive": len(members) == 1,
                     "mzs": sorted(mzs), "count": len(mzs)}
                    for members, mzs in comp_map.items()]
    # distinct (single region) first, ordered by region; then shared by size
    compartments.sort(key=lambda c: (len(c["key"]) > 1, len(c["key"]), c["key"]))
    return {"names": names, "n_pixels": [int(m.sum()) for m in masks],
            "compartments": compartments, "peaks": pd.DataFrame(rows)}


def coloc_matrix(ds, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic", method: str = "pearson",
                 mask=None, *, embedding=None):
    """Pairwise spatial-similarity matrix between all peaks (co-localization network
    backbone). Returns ``(matrix, peaks)`` with matrix shape (n_peaks, n_peaks).

    ``mask`` (a boolean per-pixel array) restricts the correlation to a region's pixels;
    when None it defaults to detected on-tissue pixels (:func:`_foreground_mask`) so the
    off-tissue background does not inflate the pairwise Pearson correlations.

    ``method="learned"`` returns the learned ion-embedding cosine matrix instead: pass the
    trained ``embedding`` and the result is ``(np.nan_to_num(embedding.similarity_matrix()),
    embedding.peaks)``, preserving the ``(matrix, peaks)`` contract :func:`coloc_modules`
    depends on. Raises :class:`ValueError` if ``embedding is None``."""
    if method == "learned":
        if embedding is None:
            raise ValueError(
                "learned colocalization needs a trained ion embedding — train one first")
        return np.nan_to_num(embedding.similarity_matrix()), np.asarray(embedding.peaks, float)
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    if mask is None:
        mask = _foreground_mask(ds)
    if mask is not None:
        X = np.asarray(X)[np.asarray(mask, dtype=bool)]
    if method == "cosine":
        Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
        M = Xn.T @ Xn
    else:
        M = np.corrcoef(X, rowvar=False)
    M = np.atleast_2d(M)                          # corrcoef returns 0-d for a single column
    return np.nan_to_num(M), np.asarray(peaks, float)


@dataclass
class ColocModules:
    matrix: np.ndarray          # correlation matrix reordered by module
    peaks: np.ndarray           # peak m/z, reordered to match ``matrix``
    labels: np.ndarray          # module id per reordered peak
    order: np.ndarray           # indices that reorder the original peak list

    def members(self):
        """{module_id: [m/z, …]} grouping co-localized ions."""
        out = {}
        for mz, lab in zip(self.peaks, self.labels):
            out.setdefault(int(lab), []).append(float(mz))
        return out


def coloc_modules(ds, peaks, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic", method: str = "pearson",
                  n_modules: int | None = None, threshold: float = 0.5, mask=None) -> ColocModules:
    """Group ions into **co-localization modules** — sets whose ion images share a
    spatial distribution — by hierarchical clustering of the co-localization matrix.

    Distance is ``1 - correlation``; clusters use average linkage. Give ``n_modules``
    for a fixed count, else ions merge while their correlation exceeds ``threshold``.
    The returned matrix/peaks are reordered by module so a heatmap shows blocks.
    ``mask`` restricts the correlation to a region's pixels.
    """
    from sklearn.cluster import AgglomerativeClustering

    M, pk = coloc_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm, method=method, mask=mask)
    n = M.shape[0]
    if n < 2:
        return ColocModules(M, pk, np.zeros(n, dtype=int), np.arange(n))
    dist = np.clip(1.0 - M, 0.0, 2.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    if n_modules:
        model = AgglomerativeClustering(n_clusters=int(min(n_modules, n)), metric="precomputed",
                                        linkage="average")
    else:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - threshold,
                                        metric="precomputed", linkage="average")
    labels = model.fit_predict(dist)
    order = np.argsort(labels, kind="stable")
    return ColocModules(M[np.ix_(order, order)], pk[order], labels[order], order)


# --------------------------------------------------------------------------- #
# Region-to-region correlation (which regions share a lipid profile)
# --------------------------------------------------------------------------- #
@dataclass
class RegionCorrelation:
    names: list             # region names, in matrix order
    matrix: np.ndarray      # (n_regions x n_regions) symmetric similarity, diagonal 1
    fingerprints: np.ndarray  # (n_regions x n_peaks) mean profile per region
    peaks: np.ndarray       # the m/z axis the fingerprints/correlation were built on
    method: str

    def best_match(self, name: str, among: list | None = None) -> tuple | None:
        """The region most similar to ``name`` (excluding itself), as ``(name, score)``.
        ``among`` restricts the candidates to that subset of region names."""
        if name not in self.names:
            return None
        i = self.names.index(name)
        pool = [j for j, nm in enumerate(self.names)
                if nm != name and (among is None or nm in among)]
        if not pool:
            return None
        j = max(pool, key=lambda k: self.matrix[i, k])
        return self.names[j], float(self.matrix[i, j])


def _profile_similarity(F: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Pairwise similarity of the rows of ``F`` (one fingerprint per row).

    ``pearson`` mean-centres each row first (correlation of the profiles);
    ``cosine`` skips centring (angle between the raw intensity profiles). Both
    reduce to a dot product of L2-normalised rows, so the result is symmetric
    with a unit diagonal. Zero-norm rows (an all-zero region) score 0.
    """
    F = np.asarray(F, dtype=np.float64)
    if method == "pearson":
        F = F - F.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    U = F / nrm
    M = U @ U.T
    return np.clip(M, -1.0, 1.0)


def region_correlation(ds, region_masks: dict, peaks, tol_ppm: float = DEFAULT_TOL_PPM,
                       norm: str = "tic", method: str = "pearson") -> RegionCorrelation:
    """Collapse each region to its **mean feature profile** (one intensity per peak,
    averaged over the region's pixels), then score every region pair by the
    similarity of those profiles — so you can ask which regions share a lipid
    composition even when they live on different slides / have no shared pixels.

    ``region_masks`` maps a region name → boolean per-pixel mask (same pixel order
    as the dataset's feature matrix). All fingerprints are built on the one shared
    ``peaks`` axis, so the comparison is apples-to-apples. ``method`` is ``pearson``
    (profile correlation, intensity-scale invariant) or ``cosine``.
    """
    names = [nm for nm, m in region_masks.items()
             if m is not None and np.asarray(m, bool).any()]
    if len(names) < 2:
        raise ValueError("region correlation needs at least two non-empty regions")
    X = feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)   # pixels x peaks
    fps = np.vstack([X[np.asarray(region_masks[nm], bool)].mean(axis=0) for nm in names])
    M = _profile_similarity(fps, method=method)
    pk = np.asarray([float(p["mz"]) if isinstance(p, dict) else float(p) for p in peaks],
                    dtype=float)
    return RegionCorrelation(names=names, matrix=M, fingerprints=fps, peaks=pk, method=method)
