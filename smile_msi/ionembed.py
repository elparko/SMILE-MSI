"""Learned ion-image representations — a self-supervised contrastive encoder over a
single slide's ion images, the learned analogue of the hand-crafted Pearson/cosine
co-localization in :mod:`smile_msi.spatial`.

Every co-localization decision in SMILE MSI is, by default, a hand-crafted pixel
statistic on the *raw* ion-image vectors (:func:`spatial.colocalize` →
:func:`spatial._similarity`). Tissue ion images are sparse and Poisson-noisy with a
large shared on/off-tissue block, so raw-pixel correlation is easily inflated. The
literature consensus (ColocML, Ovchinnikova 2020; DeepION, Wang 2024) is that
co-localization computed on a *learned/denoised* representation agrees with expert
judgment substantially better than raw-pixel correlation.

This module adds a small **self-supervised contrastive ion-image encoder**
(DeepION-style) trainable on the user's own slide on CPU, producing a per-ion
embedding vector. Co-localization is then cosine of the learned per-ion embeddings, a
denoised alternative reached via ``spatial.colocalize(..., method="learned",
embedding=...)``.

Design rules (deliberate, and different from :mod:`smile_msi.explain`):

* **No ``torch`` at import time.** ``torch`` is lazy-imported inside the train/encode
  functions only; module-level imports stay ``numpy`` + ``from . import spatial``. The
  GUI and the rest of the engine never pull ``torch`` in.
* **No silent fallback, no auto-install.** :func:`_ensure_torch` raises a *clear*
  ``ImportError`` naming ``uv pip install torch`` if ``torch`` is absent — it does
  **not** ``pip install`` on demand (unlike :func:`explain._ensure_shap`) and it does
  **not** fall back to a non-DL method. A hard project rule is "make the absence
  apparent; no silent fail/fallback": deep learning is opt-in, never the default, and
  its absence must be visible.
* **CPU-only + deterministic.** A fixed ``random_state`` seeds ``numpy`` and
  ``torch.manual_seed``; ``torch.set_num_threads`` is pinned so a given seed + thread
  count reproduces the embedding (bit-reproducibility across *differing* thread counts
  is not promised).

Refs:

* Wang, X. et al. (2024). DeepION: a deep learning–based low-dimensional
  representation model of ion images for mass spectrometry imaging. *Analytical
  Chemistry*, 96. doi:10.1021/acs.analchem.3c05002. The MSI-specific augmentation set
  (Poisson abundance resampling, random pixel missingness) + NT-Xent objective are
  reimplemented **clean-room from the publication**, not copied from its source.
* Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020). A simple framework for
  contrastive learning of visual representations (SimCLR). *ICML*. arXiv:2002.05709 —
  the NT-Xent / InfoNCE loss.
* Ovchinnikova, K. et al. (2020). ColocML. *Bioinformatics*, 36(10).
  doi:10.1093/bioinformatics/btaa085 — learned-feature cosine beats raw-pixel cosine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_TOL_PPM

from . import spatial


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class IonEmbedding:
    """A trained per-ion embedding produced by :func:`train_ion_encoder`.

    ``vectors[i]`` is the L2-normalized embedding of the ion at ``peaks[i]``, so the
    cosine similarity of two ions is just their dot product, and
    :meth:`similarity_matrix` is ``vectors @ vectors.T``.
    """

    peaks: np.ndarray          # (n_peaks,) m/z, column-aligned to ``vectors``
    vectors: np.ndarray        # (n_peaks, dim) L2-normalized embedding per ion
    dim: int
    epochs: int
    loss: float                # final-epoch mean NT-Xent loss (training sanity)
    random_state: int
    settings: dict = field(default_factory=dict)   # tol_ppm/norm/peaks-hash/etc. for provenance

    def similarity_matrix(self) -> np.ndarray:
        """Cosine similarity between every pair of ions (rows are unit-norm, so this is
        ``vectors @ vectors.T``). Symmetric with a unit diagonal."""
        V = np.asarray(self.vectors, dtype=np.float64)
        M = V @ V.T
        return np.clip(M, -1.0, 1.0)

    def rank(self, target_mz: float, *, n: int | None = None) -> list:
        """Rank every ion by embedding-cosine to ``target_mz`` → ``[{mz, score}]`` sorted
        descending, matching :func:`spatial.colocalize`'s output shape. The target ion
        (cosine 1) is included, as :func:`spatial.colocalize` includes the target column."""
        peaks = np.asarray(self.peaks, dtype=float)
        if peaks.size == 0:
            return []
        j = int(np.argmin(np.abs(peaks - float(target_mz))))
        sims = self.similarity_matrix()[j]
        out = [{"mz": float(peaks[c]), "score": float(sims[c])} for c in range(peaks.size)]
        out.sort(key=lambda d: (np.isnan(d["score"]), -d["score"]))
        return out if n is None else out[: int(n)]


# --------------------------------------------------------------------------- #
# Lazy torch import — clear failure, NO auto-install, NO fallback
# --------------------------------------------------------------------------- #
def _ensure_torch(progress=None):
    """Import ``torch``, or raise a clear :class:`ImportError` naming the install command.

    Deliberately different from :func:`explain._ensure_shap`: this does **not**
    auto-``pip install`` and it does **not** fall back to a non-DL method. Learned
    embeddings are an opt-in ``torch`` extra; if it is missing, the feature must fail
    loudly so the absence is apparent (a hard project rule), never silently degrade to
    Pearson.
    """
    try:
        import torch
        return torch
    except ImportError as e:                        # pragma: no cover - exercised when torch absent
        raise ImportError(
            "Learned ion-image embeddings need the optional 'torch' package, which is not "
            "installed. Install it with `uv pip install torch` (or `uv sync --extra ion-embed`). "
            "This is an opt-in deep-learning extra; raw-pixel co-localization "
            "(method='pearson'/'cosine'/…) needs no extra dependency."
        ) from e


# --------------------------------------------------------------------------- #
# Ion-image stack
# --------------------------------------------------------------------------- #
def ion_image_stack(ds, peaks, *, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                    fg_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-ion image tensor for the contrastive encoder.

    Returns ``(stack, mask)`` where ``stack`` is ``(n_peaks, H, W)`` float32 with
    ``H == ds.height``, ``W == ds.width``, and ``mask`` is the ``(H, W)`` bool on-tissue
    grid. Each ion image is built from the cached feature matrix scattered onto the grid
    via :meth:`MSIDataset.to_image` (so orientation/rotation composes for free),
    per-image min-max scaled to ``[0, 1]``, with off-tissue (and non-acquired) pixels set
    to ``0``.

    ``fg_mask`` (a per-pixel bool array) restricts the on-tissue support; it defaults to
    :func:`spatial._foreground_mask` (detected tissue), falling back to all acquired
    pixels when no tissue is detected.
    """
    peaks = np.asarray(peaks, dtype=float)
    X = spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)   # (n_pixels, n_peaks)
    n_pix = X.shape[0]

    if fg_mask is None:
        fg_mask = spatial._foreground_mask(ds)
    if fg_mask is None:
        fg = np.ones(n_pix, dtype=bool)
    else:
        fg = np.asarray(fg_mask, dtype=bool)

    # On-tissue grid: scatter the per-pixel foreground onto the image grid (acquired-but-
    # off-tissue and non-acquired cells are both False).
    mask_img = ds.to_image(fg.astype(float), fill=0.0) > 0.5            # (H, W) bool
    H, W = mask_img.shape

    stack = np.zeros((peaks.size, H, W), dtype=np.float32)
    for k in range(peaks.size):
        vec = np.where(fg, X[:, k], 0.0)                               # zero off-tissue pixels
        img = ds.to_image(vec, fill=0.0)                              # (H, W); non-acquired → 0
        img = np.where(mask_img, img, 0.0)
        on = img[mask_img]
        if on.size:
            lo = float(on.min())
            hi = float(on.max())
            if hi > lo:
                img = np.where(mask_img, (img - lo) / (hi - lo), 0.0)
            else:                                                      # flat image → all zero
                img = np.zeros_like(img)
        stack[k] = img.astype(np.float32)
    return stack, mask_img


# --------------------------------------------------------------------------- #
# DeepION-style MSI-specific augmentations (clean-room from Anal. Chem. 2024)
# --------------------------------------------------------------------------- #
def _aug_poisson(img: np.ndarray, rng) -> np.ndarray:
    """Poisson abundance resampling — model the ion image's counts as a Poisson draw so
    each augmented view has independent shot noise, then rescale back to ``[0, 1]``.

    The image is in ``[0, 1]``; scale it up to a modest count regime, draw
    ``Poisson(scaled)``, and min-max renormalize. Off-tissue zeros (Poisson(0) = 0)
    stay zero, so the support is preserved. Reimplemented from DeepION's MSI-specific
    augmentation set (Wang 2024)."""
    img = np.asarray(img, dtype=np.float64)
    scale = 40.0                                       # counts at full intensity (modest, CPU-cheap)
    lam = np.clip(img, 0.0, None) * scale
    drawn = rng.poisson(lam).astype(np.float64)
    m = float(drawn.max())
    if m > 0:
        drawn = drawn / m
    return drawn.astype(np.float32)


def _aug_missing(img: np.ndarray, rng, p: float) -> np.ndarray:
    """Random pixel missingness — independently zero a fraction ``p`` of the on-tissue
    (nonzero) pixels, the detector-dropout augmentation from DeepION. Off-tissue zeros
    are untouched (they carry no signal to drop)."""
    img = np.asarray(img, dtype=np.float32).copy()
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return img
    on = img > 0.0
    drop = (rng.random(img.shape) < p) & on
    img[drop] = 0.0
    return img


def _augment_pair(img: np.ndarray, rng, p_missing: float) -> tuple[np.ndarray, np.ndarray]:
    """Two independent augmented views of one ion image (Poisson resample then random
    missingness, drawn independently per view) — the positive pair the contrastive loss
    pulls together."""
    v1 = _aug_missing(_aug_poisson(img, rng), rng, p_missing)
    v2 = _aug_missing(_aug_poisson(img, rng), rng, p_missing)
    return v1, v2


# --------------------------------------------------------------------------- #
# Encoder + NT-Xent contrastive training
# --------------------------------------------------------------------------- #
def _build_encoder(torch, dim: int, proj_dim: int):
    """Small CNN encoder (3 conv blocks → global average pool → ``dim``-d embedding head)
    with a projection head used only for the contrastive loss (SimCLR: project for the
    loss, embed for downstream). Returns ``(encoder_module,)`` as one ``nn.Module`` whose
    ``forward`` returns ``(embedding, projection)``."""
    import torch.nn as nn

    class _IonEncoder(nn.Module):
        def __init__(self, dim, proj_dim):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.embed = nn.Linear(64, dim)
            self.proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.ReLU(),
                                      nn.Linear(proj_dim, proj_dim))

        def forward(self, x):
            h = self.features(x).flatten(1)            # (B, 64)
            z = self.embed(h)                          # (B, dim) embedding
            p = self.proj(z)                           # (B, proj_dim) projection for the loss
            return z, p

    return _IonEncoder(dim, proj_dim)


def _nt_xent(torch, z1, z2, temperature: float):
    """NT-Xent / InfoNCE contrastive loss (SimCLR, Chen 2020) for a batch of ``B`` ion
    images, each contributing two augmented views. The two views of an image are the
    positive pair; the other ``2B - 2`` views are negatives.

    ``z1``, ``z2`` are ``(B, d)`` projection outputs; they are L2-normalized so the dot
    product is cosine. Returns the mean loss over all ``2B`` anchors."""
    import torch.nn.functional as F

    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2B, d)
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / float(temperature)             # (2B, 2B) cosine / T
    # Mask self-similarity so an anchor never matches itself.
    diag = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, float("-inf"))
    # Positive index: view i (0..B-1) pairs with i+B, and vice versa.
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def train_ion_encoder(ds, peaks, *, dim: int = 32, epochs: int = 30, batch_size: int = 64,
                      proj_dim: int = 64, temperature: float = 0.5, p_missing: float = 0.2,
                      tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic", random_state: int = 0,
                      n_threads: int | None = None, progress=None) -> IonEmbedding:
    """Train a self-supervised contrastive ion-image encoder over this slide's ``peaks``.

    Builds the per-ion image stack (:func:`ion_image_stack`), then trains a small CNN
    encoder with an NT-Xent / InfoNCE objective (SimCLR; Chen 2020) over MSI-specific
    augmentations (Poisson abundance resampling + random pixel missingness; DeepION,
    Wang 2024): two augmented views of an ion image are pulled together while other ions
    are pushed apart, so an ion's embedding captures its *spatial distribution*
    independent of shot noise. Returns L2-normalized per-ion embeddings in an
    :class:`IonEmbedding`.

    CPU-only and deterministic: ``random_state`` seeds ``numpy`` + ``torch.manual_seed``,
    ``n_threads`` pins ``torch.set_num_threads`` (defaults to torch's current setting).
    A fixed seed + thread count reproduces the vectors; bit-reproducibility across
    differing thread counts is not promised.
    """
    torch = _ensure_torch(progress)

    peaks = np.asarray(peaks, dtype=float)
    if peaks.size == 0:
        raise ValueError("training an ion embedding needs at least one peak")

    if n_threads is not None:
        torch.set_num_threads(int(n_threads))
    torch.manual_seed(int(random_state))
    rng = np.random.default_rng(int(random_state))

    if progress:
        progress(f"building ion-image stack ({peaks.size} ions)…")
    stack, _mask = ion_image_stack(ds, peaks, tol_ppm=tol_ppm, norm=norm)
    n_ions, H, W = stack.shape

    model = _build_encoder(torch, dim=int(dim), proj_dim=int(proj_dim))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    bs = int(min(max(2, batch_size), n_ions))          # NT-Xent needs ≥ 2 distinct ions
    final_loss = float("nan")
    for ep in range(int(epochs)):
        order = rng.permutation(n_ions)
        losses = []
        for s in range(0, n_ions, bs):
            idx = order[s : s + bs]
            if idx.size < 2:                           # a trailing singleton has no negatives
                continue
            v1 = np.empty((idx.size, 1, H, W), dtype=np.float32)
            v2 = np.empty((idx.size, 1, H, W), dtype=np.float32)
            for r, i in enumerate(idx):
                a, b = _augment_pair(stack[i], rng, p_missing)
                v1[r, 0] = a
                v2[r, 0] = b
            t1 = torch.from_numpy(v1)
            t2 = torch.from_numpy(v2)
            _, p1 = model(t1)
            _, p2 = model(t2)
            loss = _nt_xent(torch, p1, p2, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        final_loss = float(np.mean(losses)) if losses else float("nan")
        if progress:
            progress(f"epoch {ep + 1}/{int(epochs)} — loss {final_loss:.4f}")

    # Embed every ion's *clean* (un-augmented) image; L2-normalize so cosine == dot.
    model.eval()
    with torch.no_grad():
        clean = stack.reshape(n_ions, 1, H, W).astype(np.float32)
        z, _ = model(torch.from_numpy(clean))
        vecs = z.detach().numpy().astype(np.float64)
    nrm = np.linalg.norm(vecs, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    vecs = (vecs / nrm).astype(np.float32)

    settings = dict(tol_ppm=float(tol_ppm), norm=str(norm), batch_size=int(batch_size),
                    proj_dim=int(proj_dim), temperature=float(temperature),
                    p_missing=float(p_missing), n_ions=int(n_ions),
                    peaks_hash=_peaks_hash(peaks))
    return IonEmbedding(peaks=peaks.astype(float), vectors=vecs, dim=int(dim),
                        epochs=int(epochs), loss=final_loss, random_state=int(random_state),
                        settings=settings)


def _peaks_hash(peaks) -> str:
    """Stable short hash of the peak m/z list (provenance / cache key)."""
    import hashlib

    arr = np.asarray(peaks, dtype=np.float64)
    return hashlib.sha1(np.round(arr, 4).tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Embedding-space co-localization (learned analogues of spatial.*)
# --------------------------------------------------------------------------- #
def colocalize_learned(ds, target_mz: float, peaks, embedding: IonEmbedding, *,
                       n: int | None = None) -> list:
    """Embedding-space co-localization ranking — the learned analogue of
    :func:`spatial.colocalize`. Cosine of the target ion's learned embedding vs every
    other ion; returns ``[{mz, score}]`` sorted descending, matching
    :func:`spatial.colocalize`'s output shape.

    ``ds`` / ``peaks`` are accepted for signature parity with :func:`spatial.colocalize`
    (the ranking is read straight from the trained ``embedding``, whose ``peaks`` axis is
    authoritative)."""
    if embedding is None:
        raise ValueError("learned colocalization needs a trained ion embedding — train one first")
    return embedding.rank(target_mz, n=n)


def embedding_modules(embedding: IonEmbedding, *, n_modules: int | None = None,
                      threshold: float = 0.5) -> "spatial.ColocModules":
    """Group ions into co-localization / isotope / adduct modules by clustering the
    embedding cosine matrix — the learned analogue of :func:`spatial.coloc_modules`.

    Distance is ``1 - cosine`` with average linkage (same as
    :func:`spatial.coloc_modules`). Give ``n_modules`` for a fixed count, else ions merge
    while their cosine exceeds ``threshold``. Returns a :class:`spatial.ColocModules`
    (matrix/peaks reordered by module) so it drops into the same downstream.
    """
    if embedding is None:
        raise ValueError("learned modules need a trained ion embedding — train one first")
    from sklearn.cluster import AgglomerativeClustering

    M = np.nan_to_num(embedding.similarity_matrix())
    pk = np.asarray(embedding.peaks, dtype=float)
    n = M.shape[0]
    if n < 2:
        return spatial.ColocModules(M, pk, np.zeros(n, dtype=int), np.arange(n))
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
    return spatial.ColocModules(M[np.ix_(order, order)], pk[order], labels[order], order)


def validate_against_pearson(ds, peaks, embedding: IonEmbedding, *, sample_pairs: int = 2000,
                             random_state: int = 0) -> dict:
    """User-tissue sanity check: on random ion pairs, correlate the learned-cosine
    similarity with the raw-pixel similarity (:func:`spatial._similarity`, Pearson) and
    report the rank agreement.

    No external benchmark is needed — this measures whether the learned ordering tracks
    (or departs from) raw Pearson on *this* tissue. Returns a dict with keys
    ``spearman`` (rank agreement of learned-cosine vs raw Pearson over the sampled pairs),
    ``pearson`` (the linear correlation of the two similarity series), ``n_pairs``
    (pairs actually scored), and ``disagreements`` (a few pairs where the two most
    disagree, for inspection)."""
    if embedding is None:
        raise ValueError("validation needs a trained ion embedding — train one first")
    from scipy.stats import spearmanr, pearsonr

    pk = np.asarray(embedding.peaks, dtype=float)
    n = pk.size
    learned = np.nan_to_num(embedding.similarity_matrix())

    X = spatial.feature_matrix(ds, peaks, tol_ppm=float(embedding.settings.get("tol_ppm", 50.0)),
                               norm=str(embedding.settings.get("norm", "tic")))
    fg = spatial._foreground_mask(ds)
    if fg is not None:
        X = np.asarray(X)[np.asarray(fg, dtype=bool)]

    rng = np.random.default_rng(int(random_state))
    if n < 2:
        return {"spearman": float("nan"), "pearson": float("nan"), "n_pairs": 0,
                "disagreements": []}
    # All unique unordered pairs, subsampled for speed on large feature sets.
    iu, ju = np.triu_indices(n, k=1)
    if iu.size > sample_pairs:
        sel = rng.choice(iu.size, size=int(sample_pairs), replace=False)
        iu, ju = iu[sel], ju[sel]

    learned_s, raw_s, pairs = [], [], []
    for i, j in zip(iu, ju):
        raw = spatial._similarity(X[:, i], X[:, j], "pearson")
        if not np.isfinite(raw):
            continue
        ls = float(learned[i, j])
        learned_s.append(ls)
        raw_s.append(float(raw))
        pairs.append((float(pk[i]), float(pk[j]), ls, float(raw)))

    learned_s = np.asarray(learned_s)
    raw_s = np.asarray(raw_s)
    if learned_s.size < 2 or np.ptp(learned_s) == 0 or np.ptp(raw_s) == 0:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "n_pairs": int(learned_s.size), "disagreements": []}
    rho = float(spearmanr(learned_s, raw_s)[0])
    r = float(pearsonr(learned_s, raw_s)[0])
    # Largest rank-gaps between the two similarity series → most-disagreeing pairs.
    lr = _rankdata(learned_s)
    rr = _rankdata(raw_s)
    gap = np.abs(lr - rr)
    worst = np.argsort(-gap)[:5]
    disagreements = [{"mz_a": pairs[k][0], "mz_b": pairs[k][1],
                      "learned": pairs[k][2], "pearson": pairs[k][3]} for k in worst]
    return {"spearman": rho if np.isfinite(rho) else float("nan"),
            "pearson": r if np.isfinite(r) else float("nan"),
            "n_pairs": int(learned_s.size), "disagreements": disagreements}


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of ``a`` (ties share the mean rank) — a tiny scipy-free rankdata for
    the disagreement gap."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # Average ties.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
