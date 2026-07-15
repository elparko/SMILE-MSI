"""Cohort-global TIC normalization for the pooled UMAP embedding (multivariate.pooled_embedding,
``cohort_norm=True``).

The default per-slide normalization anchors each slide to *its own* mean intensity, which makes
intensities incomparable across slides (Farrow et al. 2025). ``cohort_norm`` instead anchors every
slide's per-pixel TIC scaling to **one shared reference — the MEAN TIC of all pooled pixels** — so
same-biology slides line up while genuine cross-slide abundance differences survive. The mean (not
the paper's median) is used deliberately: the median recipe is a withheld patented method.
"""
import numpy as np

from smile_msi import multivariate


class _TinyDS:
    """Minimal dataset double: hands back controlled raw counts for given pixel rows and a
    controlled per-pixel full-spectrum TIC (``_pix['tic']``), so the cohort-global anchor math
    can be checked exactly without a real imzML slide."""

    def __init__(self, counts, tic):
        self._counts = np.asarray(counts, dtype=float)
        self._pix = {"tic": np.asarray(tic, dtype=float)}
        self.n_pixels = int(self._counts.shape[0])

    def prime(self):                              # cohort path calls it; already "primed"
        pass

    def features_for_rows(self, peaks, rows, tol_ppm=20.0, reduce="sum", norm="none"):
        # the cohort path always extracts raw (norm="none"); return the requested rows verbatim
        return self._counts[np.asarray(rows, dtype=int)].astype(np.float32)


def _sample(name, group="g"):
    return type("S", (), {"name": name, "group": group})()


def test_cohort_norm_anchors_to_global_mean_tic():
    """Every pixel's cohort-normalized value is ``raw * G / tic`` where ``G`` is the MEAN TIC over
    all pooled pixels (then log1p'd). A skewed TIC (one big outlier) makes mean ≠ median, so the
    result matches the mean anchor and NOT the median anchor — proving we don't ship the patented
    median recipe."""
    targets = np.array([100.0, 200.0, 300.0])
    counts_a = np.array([[10, 20, 30], [5, 5, 5], [1, 2, 3],
                         [40, 10, 0], [2, 2, 2], [7, 3, 1]], dtype=float)
    tic_a = np.array([100.0, 50.0, 10.0, 200.0, 30.0, 5.0])
    counts_b = np.array([[3, 3, 3], [8, 1, 1], [0, 5, 5],
                         [2, 2, 2], [9, 9, 0], [1, 1, 1]], dtype=float)
    tic_b = np.array([1000.0, 20.0, 15.0, 8.0, 60.0, 25.0])   # 1000 → mean >> median
    dss = {"A": _TinyDS(counts_a, tic_a), "B": _TinyDS(counts_b, tic_b)}

    emb = multivariate.pooled_embedding(
        [_sample("A"), _sample("B")], targets, loader=lambda s: (dss[s.name], None),
        method="tsne", per_sample_cap=None, cohort_norm=True,
        standardize_per_sample=True,             # must be IGNORED — cohort_norm forces it off
        keep_features=True)

    all_tic = np.concatenate([tic_a, tic_b])
    g_mean = all_tic.mean()
    g_median = np.median(all_tic)
    assert not np.isclose(g_mean, g_median)      # skewed enough for the two anchors to differ

    def normed(counts, tic, g):
        return np.log1p(np.clip(counts * (g / tic)[:, None], 0, None))

    expect_mean = np.vstack([normed(counts_a, tic_a, g_mean), normed(counts_b, tic_b, g_mean)])
    expect_median = np.vstack([normed(counts_a, tic_a, g_median), normed(counts_b, tic_b, g_median)])

    # emb.features are the pre-standardize values, stacked in sample order (A then B)
    assert emb.features.shape == (12, 3)
    assert np.allclose(emb.features, expect_mean)
    assert not np.allclose(emb.features, expect_median)


def test_cohort_norm_makes_same_biology_slides_comparable():
    """Two slides with identical biology but a global brightness difference (B = A × K, TIC and
    counts both scaled) become element-wise equal after cohort-global normalization — the
    cross-slide comparability the default per-slide normalization destroys."""
    targets = np.array([100.0, 200.0, 300.0])
    counts_a = np.array([[10, 20, 30], [5, 1, 8], [2, 9, 4],
                         [40, 10, 5], [3, 3, 7]], dtype=float)
    tic_a = np.array([100.0, 40.0, 60.0, 200.0, 30.0])
    k = 3.0                                       # B is globally 3× brighter, same shape
    dss = {"A": _TinyDS(counts_a, tic_a), "B": _TinyDS(counts_a * k, tic_a * k)}

    emb = multivariate.pooled_embedding(
        [_sample("A"), _sample("B")], targets, loader=lambda s: (dss[s.name], None),
        method="tsne", per_sample_cap=None, cohort_norm=True, keep_features=True)

    feats_a = emb.features[emb.sample_id == 0]
    feats_b = emb.features[emb.sample_id == 1]
    assert feats_a.shape == feats_b.shape == (5, 3)
    assert np.allclose(feats_a, feats_b)          # brightness removed, biology preserved → equal
