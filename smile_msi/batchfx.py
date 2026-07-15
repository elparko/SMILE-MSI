"""Batch-effect correction and QC for multi-sample MSI cohorts.

Clean-room reimplementation of **ComBat** empirical-Bayes batch adjustment
(Johnson, Li & Rabinovic 2007, doi:10.1093/biostatistics/kxj037) with an optional
*protected-covariate* design — biology is preserved by regressing it alongside batch
in ComBat's own model matrix and adding it back, so a biological contrast survives
while cross-batch technical shifts are removed. This is ComBat's native covariate
preservation; it is **not** reComBat (Adler et al. 2022 add regularized covariate
regression and explicit unwanted-confounder terms, which are *not* implemented here —
cf. doi:10.1093/bioadv/vbac016).

Plus batch-mixing **QC metrics**: a kBET-like neighbourhood χ² rejection rate
(Büttner et al. 2019, doi:10.1038/s41592-018-0254-1) and silhouette-by-batch vs
-by-biology (Rousseeuw 1987) — the before/after acceptance read that guards against
over-correction.

Pure NumPy/SciPy/scikit-learn — no new dependency, offline. Reimplemented from the
publication; the GPL ``sva`` ComBat source was **not** copied (copyright protects
code, not algorithms — see ``THIRD_PARTY_LICENSES.md``).

All functions take a dense ``X`` of shape ``(n_rows, n_features)`` where a row is a
sample / region / pixel, plus label vectors — no :class:`~smile_msi.msi.MSIDataset`
coupling, so they unit-test on synthetic data and serve both the summarized and
pixel-level correction paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CorrectionReport",
    "MixingState",
    "MixingReport",
    "combat",
    "correction_summary",
    "batch_mixing",
    "compare_mixing",
]


@dataclass
class CorrectionReport:
    """Per-feature batch-variance fractions before/after correction."""

    var_fraction_before: np.ndarray   # (p,) SS_batch/SS_total per feature, pre
    var_fraction_after: np.ndarray    # (p,) post
    var_reduction: float              # (mean_before - mean_after) / mean_before
    n_features_improved: int          # count of features whose batch-variance dropped


@dataclass
class MixingState:
    """Batch-mixing QC for one matrix state (before *or* after)."""

    rejection: float                   # kBET-like neighbourhood rejection rate in [0,1]
    silhouette_batch: float | None     # silhouette of the batch labelling (lower = better mixed)
    silhouette_biology: float | None   # silhouette of the biology labelling (higher = preserved)


@dataclass
class MixingReport:
    """Before/after batch-mixing QC (the acceptance read)."""

    rejection_before: float
    rejection_after: float
    silhouette_batch_before: float | None
    silhouette_batch_after: float | None
    silhouette_biology_before: float | None
    silhouette_biology_after: float | None


def _as_codes(labels):
    labels = np.asarray(labels)
    uniq, codes = np.unique(labels, return_inverse=True)
    return uniq, codes.astype(int)


def combat(X, batch, *, covariates=None, mean_only=False, parametric=True,
           ref_batch=None, eps=1e-8):
    """Empirical-Bayes ComBat batch adjustment.

    Parameters
    ----------
    X : array (n_rows, n_features)
        Feature matrix; rows are samples / regions / pixels.
    batch : array (n_rows,)
        Batch label per row. With a single batch the input is returned unchanged.
    covariates : array (n_rows, k), optional
        Design of *protected* biological factors (e.g. a drop-first one-hot of the
        group plus numeric meta) regressed alongside batch so their effect is kept
        (ComBat's native covariate model matrix — not reComBat's regularized
        regression). Should not duplicate an intercept (the batch dummies already
        provide one).
    mean_only : bool
        Location-only correction (skip the per-batch scale term).
    parametric : bool
        Parametric empirical-Bayes shrinkage (default). When ``False`` the unshrunk
        per-batch location/scale estimates are used (no EB pooling).
    ref_batch : label, optional
        If given, that batch is the reference: it passes through unchanged and every
        *other* batch is standardized against — and adjusted back onto — the reference
        batch's own location and scale (sva reference-batch semantics), not the
        batch-size-weighted grand mean. Needs >= 2 reference rows to estimate the
        reference scale; with a single reference row it falls back to the pooled
        (all-batch) variance for the scale term.
    eps : float
        Numerical floor for variances.

    Returns
    -------
    array (n_rows, n_features)
        Batch-corrected matrix, same shape as ``X``.

    Notes
    -----
    Degenerate inputs are handled: a single batch is a no-op; a one-row batch falls
    back to location-only for that batch; a zero-variance feature column is left
    unchanged (no NaN/inf).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_rows, n_features)")
    n, p = X.shape
    batch = np.asarray(batch)
    if batch.shape[0] != n:
        raise ValueError("batch length must match X rows")

    uniq_b, b_codes = _as_codes(batch)
    n_batch = len(uniq_b)
    if n_batch < 2:
        return X.copy()

    ref_code = None
    if ref_batch is not None:
        match = np.where(uniq_b == ref_batch)[0]
        if len(match):
            ref_code = int(match[0])

    Y = X.T.copy()  # (p, n) feature-by-sample

    # full batch dummy design (no dropped level) + optional covariates
    batchmod = np.zeros((n, n_batch))
    batchmod[np.arange(n), b_codes] = 1.0
    batch_sizes = batchmod.sum(axis=0)

    if covariates is not None:
        C = np.asarray(covariates, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError("covariates rows must match X rows")
        design = np.hstack([batchmod, C])
        n_cov = C.shape[1]
    else:
        C = None
        design = batchmod
        n_cov = 0

    # OLS fit per feature (min-norm via lstsq, robust to rank deficiency)
    B, *_ = np.linalg.lstsq(design, Y.T, rcond=None)  # (cols, p)

    if ref_code is not None:
        # Reference-batch ComBat (sva semantics): standardize against — and adjust the
        # other batches back onto — the *reference* batch's location/scale, not the
        # batch-size-weighted grand mean. Under the full one-hot design (no dropped
        # level) B[ref_code] is the reference batch's own covariate-adjusted mean.
        grand_mean = B[ref_code, :]
    else:
        grand_mean = (batch_sizes / n) @ B[:n_batch, :]   # batch-size-weighted intercept
    stand_mean = np.tile(grand_mean[:, None], (1, n))  # (p, n)
    if n_cov:
        stand_mean = stand_mean + (C @ B[n_batch:, :]).T

    resid = Y - (design @ B).T
    if ref_code is not None and batch_sizes[ref_code] >= 2:
        # Scale to the reference batch's variance (its residuals only), so corrected
        # batches match the reference spread, not the pooled-over-all spread. Needs
        # >= 2 reference rows; otherwise fall back to the all-batch pooled variance.
        ref_idx = b_codes == ref_code
        var_pooled = (resid[:, ref_idx] ** 2).mean(axis=1)
    else:
        var_pooled = (resid ** 2).mean(axis=1)            # (p,)
    zero_var = var_pooled <= eps
    sd = np.sqrt(np.where(zero_var, 1.0, var_pooled))[:, None]

    s_data = (Y - stand_mean) / sd                    # standardized (p, n)

    # per-batch location/scale estimates on standardized data
    gamma_hat = np.zeros((n_batch, p))
    delta_hat = np.ones((n_batch, p))
    batch_n = np.zeros(n_batch, dtype=int)
    for i in range(n_batch):
        idx = b_codes == i
        ni = int(idx.sum())
        batch_n[i] = ni
        gamma_hat[i] = s_data[:, idx].mean(axis=1)
        if ni > 1 and not mean_only:
            delta_hat[i] = s_data[:, idx].var(axis=1, ddof=1)
    delta_hat = np.where(delta_hat <= eps, 1.0, delta_hat)

    # EB hyperparameters across features, per batch
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)
    t2 = np.where(t2 <= eps, eps, t2)

    def _aprior(d):
        m, s2 = float(d.mean()), float(d.var())
        return (2 * s2 + m * m) / s2 if s2 > eps else 1e6

    def _bprior(d):
        m, s2 = float(d.mean()), float(d.var())
        return (m * s2 + m ** 3) / s2 if s2 > eps else m

    gamma_star = np.zeros((n_batch, p))
    delta_star = np.ones((n_batch, p))

    for i in range(n_batch):
        ni = batch_n[i]
        if ref_code is not None and i == ref_code:
            gamma_star[i] = 0.0
            delta_star[i] = 1.0
            continue
        idx = b_codes == i
        zi = s_data[:, idx]
        if mean_only or ni <= 1:
            gamma_star[i] = (t2[i] * ni * gamma_hat[i] + gamma_bar[i]) / (t2[i] * ni + 1.0)
            delta_star[i] = 1.0
            continue
        if not parametric:
            gamma_star[i] = gamma_hat[i]
            delta_star[i] = delta_hat[i]
            continue
        a_pri = _aprior(delta_hat[i])
        b_pri = _bprior(delta_hat[i])
        g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
        for _ in range(500):
            g_new = (t2[i] * ni * gamma_hat[i] + d_old * gamma_bar[i]) / (t2[i] * ni + d_old)
            sum2 = ((zi - g_new[:, None]) ** 2).sum(axis=1)
            d_new = (0.5 * sum2 + b_pri) / (ni / 2.0 + a_pri - 1.0)
            d_new = np.where(d_new <= eps, eps, d_new)
            change = max(
                float(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + eps))),
                float(np.max(np.abs(d_new - d_old) / (np.abs(d_old) + eps))),
            )
            g_old, d_old = g_new, d_new
            if change < 1e-4:
                break
        gamma_star[i] = g_old
        delta_star[i] = d_old

    bayes = s_data.copy()
    for i in range(n_batch):
        idx = b_codes == i
        bayes[:, idx] = (s_data[:, idx] - gamma_star[i][:, None]) / np.sqrt(delta_star[i])[:, None]
    out = bayes * sd + stand_mean

    if zero_var.any():
        out[zero_var, :] = Y[zero_var, :]

    return out.T


def _batch_var_fraction(X, b_codes, n_batch):
    """One-way ANOVA SS_batch / SS_total per feature."""
    X = np.asarray(X, dtype=float)
    grand = X.mean(axis=0)
    ss_total = ((X - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(X.shape[1])
    for i in range(n_batch):
        idx = b_codes == i
        ni = int(idx.sum())
        if ni == 0:
            continue
        mi = X[idx].mean(axis=0)
        ss_between += ni * (mi - grand) ** 2
    return np.where(ss_total > 0, ss_between / ss_total, 0.0)


def correction_summary(X, Xc, batch) -> CorrectionReport:
    """Per-feature batch-variance fraction before/after, aggregate reduction, and
    the count of features whose batch variance dropped."""
    _, b_codes = _as_codes(batch)
    n_batch = int(b_codes.max()) + 1 if b_codes.size else 0
    before = _batch_var_fraction(X, b_codes, n_batch)
    after = _batch_var_fraction(Xc, b_codes, n_batch)
    mb, ma = float(before.mean()), float(after.mean())
    red = (mb - ma) / mb if mb > 0 else 0.0
    improved = int(np.sum(after < before - 1e-12))
    return CorrectionReport(before, after, red, improved)


def _silhouette(data, codes):
    if len(np.unique(codes)) < 2:
        return None
    if data.shape[0] - 1 < len(np.unique(codes)):
        return None
    try:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(data, codes))
    except Exception:
        return None


def batch_mixing(X, batch, *, biology=None, k=None, n_repeats=100,
                 random_state=0) -> MixingState:
    """kBET-like batch-mixing QC for a single matrix state.

    For ``n_repeats`` random points, the local batch composition of the ``k`` nearest
    neighbours is χ²-tested against the global batch frequencies; the rejection rate
    (fraction with p < 0.05) is ~0 when batches are well mixed and ~1 when batch-
    segregated. Also returns silhouette-by-batch (lower = better mixed) and, if
    ``biology`` is given, silhouette-by-biology (higher = preserved) — the over-
    correction guard. ``random_state`` makes the point subsampling deterministic.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    _, b_codes = _as_codes(batch)
    n_batch = int(b_codes.max()) + 1 if b_codes.size else 0

    rejection = 0.0
    if n_batch >= 2 and n >= 4:
        from scipy.stats import chi2
        from sklearn.neighbors import NearestNeighbors

        kk = k or max(3, min(n - 1, int(round(0.25 * n))))
        kk = int(min(kk, n - 1))
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        _, idxs = nn.kneighbors(X)
        global_freq = np.bincount(b_codes, minlength=n_batch) / n
        exp = np.where(global_freq * kk <= 0, 1e-9, global_freq * kk)
        dof = n_batch - 1
        rng = np.random.default_rng(random_state)
        pts = rng.integers(0, n, size=int(n_repeats))
        rejects = 0
        for pt in pts:
            neigh = idxs[pt, 1:kk + 1]
            obs = np.bincount(b_codes[neigh], minlength=n_batch).astype(float)
            stat = float(((obs - exp) ** 2 / exp).sum())
            if chi2.sf(stat, dof) < 0.05:
                rejects += 1
        rejection = rejects / len(pts)

    sil_batch = _silhouette(X, b_codes)
    sil_bio = None
    if biology is not None:
        _, g_codes = _as_codes(biology)
        sil_bio = _silhouette(X, g_codes)
    return MixingState(rejection, sil_batch, sil_bio)


def compare_mixing(X, Xc, batch, *, biology=None, k=None, n_repeats=100,
                   random_state=0) -> MixingReport:
    """Run :func:`batch_mixing` on the uncorrected and corrected matrices and pack the
    before/after acceptance read."""
    a = batch_mixing(X, batch, biology=biology, k=k, n_repeats=n_repeats,
                     random_state=random_state)
    b = batch_mixing(Xc, batch, biology=biology, k=k, n_repeats=n_repeats,
                     random_state=random_state)
    return MixingReport(a.rejection, b.rejection, a.silhouette_batch, b.silhouette_batch,
                        a.silhouette_biology, b.silhouette_biology)
