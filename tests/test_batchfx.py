"""Tests for smile_msi.batchfx — ComBat batch correction + QC metrics."""
import numpy as np
import pytest

from smile_msi import batchfx


def _make(rng, n_per=40, p=24, n_batch=3, group_effect=0.0, confound=0.5):
    """Synthetic cohort: a biology signal (two groups) plus per-batch additive
    location and multiplicative scale shifts. ``confound`` skews group membership by
    batch (0.5 = balanced, →1 = group confounded with batch)."""
    base = rng.normal(0, 1, p)
    geff = rng.normal(0, 1, p) * group_effect
    rows, batch, group = [], [], []
    for b in range(n_batch):
        loc = rng.normal(0, 2.0, p)
        scale = rng.uniform(0.6, 1.8, p)
        pg1 = confound if b % 2 == 0 else (1.0 - confound)
        for _ in range(n_per):
            g = 1 if rng.random() < pg1 else 0
            x = base + g * geff + rng.normal(0, 1, p) * scale + loc
            rows.append(x)
            batch.append(f"B{b}")
            group.append(g)
    return np.asarray(rows), np.asarray(batch), np.asarray(group)


def _group_sep(M, group):
    g = np.asarray(group)
    m0 = M[g == 0].mean(0)
    m1 = M[g == 1].mean(0)
    return float(np.abs(m1 - m0).mean())


def test_combat_removes_batch_shift():
    rng = np.random.default_rng(0)
    X, batch, _ = _make(rng, group_effect=0.0)
    Xc = batchfx.combat(X, batch)
    rep = batchfx.correction_summary(X, Xc, batch)
    assert rep.var_fraction_before.mean() > rep.var_fraction_after.mean()
    assert rep.var_reduction > 0.5
    assert rep.var_fraction_after.mean() < 0.1
    assert rep.n_features_improved >= X.shape[1] // 2
    assert np.isfinite(Xc).all()


def test_biology_preserved_with_covariates():
    # group confounded with batch so naive ComBat erases real biology
    rng = np.random.default_rng(1)
    X, batch, group = _make(rng, n_batch=2, group_effect=3.0, confound=0.85)
    g = np.asarray(group)
    cov = g.reshape(-1, 1).astype(float)  # drop-first one-hot of the 2-group factor

    Xc = batchfx.combat(X, batch, covariates=cov)   # protected
    Xn = batchfx.combat(X, batch)                   # naive

    # covariate protection keeps the group contrast; naive shrinks it more
    assert _group_sep(Xc, group) > _group_sep(Xn, group)
    assert _group_sep(Xc, group) > 1.0


def test_qc_metric_improves_after_correction():
    rng = np.random.default_rng(2)
    X, batch, group = _make(rng, group_effect=1.0)
    Xc = batchfx.combat(X, batch)
    rpt = batchfx.compare_mixing(X, Xc, batch, biology=group, random_state=0)
    # batch becomes better mixed: rejection rate and silhouette-by-batch drop
    assert rpt.rejection_after <= rpt.rejection_before
    assert rpt.silhouette_batch_after <= rpt.silhouette_batch_before + 1e-9


def test_single_batch_is_noop():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 5))
    Xc = batchfx.combat(X, np.array(["A"] * 20))
    assert np.allclose(Xc, X)


def test_one_sample_batch_no_nan():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(11, 5))
    batch = np.array(["A"] * 10 + ["B"])  # B has a single row
    Xc = batchfx.combat(X, batch)
    assert np.isfinite(Xc).all()


def test_zero_variance_feature_unchanged():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 4))
    X[:, 0] = 5.0  # constant feature
    batch = np.array(["A"] * 15 + ["B"] * 15)
    Xc = batchfx.combat(X, batch)
    assert np.isfinite(Xc).all()
    assert np.allclose(Xc[:, 0], 5.0)


def test_mean_only_skips_scale():
    rng = np.random.default_rng(6)
    X, batch, _ = _make(rng, group_effect=0.0)
    Xc = batchfx.combat(X, batch, mean_only=True)
    rep = batchfx.correction_summary(X, Xc, batch)
    assert rep.var_reduction > 0.0  # location correction still helps
    assert np.isfinite(Xc).all()


def test_ref_batch_passes_through():
    rng = np.random.default_rng(7)
    X, batch, _ = _make(rng, n_batch=2, group_effect=0.0)
    Xc = batchfx.combat(X, batch, ref_batch="B0")
    ref = np.asarray(batch) == "B0"
    assert np.allclose(Xc[ref], X[ref])  # reference batch untouched


def test_ref_batch_aligns_others_to_reference():
    """Non-reference batches are corrected ONTO the reference batch's location, not the
    grand mean (sva reference-batch semantics; KNOWN_ISSUES.md). Was previously wrong:
    others landed on the batch-size-weighted grand mean instead of the reference."""
    rng = np.random.default_rng(11)
    X, batch, _ = _make(rng, n_batch=2, group_effect=0.0)
    batch = np.asarray(batch)
    ref, other = batch == "B0", batch == "B1"
    Xc = batchfx.combat(X, batch, ref_batch="B0")
    ref_mean = X[ref].mean(0)                 # reference is untouched, so == Xc[ref].mean(0)
    grand_mean = X.mean(0)                     # the old (wrong) target
    other_after = Xc[other].mean(0)
    # the corrected non-reference batch lands on the reference mean, clearly closer to it
    # than to the grand mean (the two targets are well separated by the batch shift)...
    assert np.abs(other_after - ref_mean).mean() < np.abs(other_after - grand_mean).mean()
    # ...and is close to the reference in absolute terms (location difference removed)
    assert np.abs(other_after - ref_mean).mean() < 0.25


def test_mixing_determinism():
    rng = np.random.default_rng(8)
    X, batch, _ = _make(rng)
    r1 = batchfx.batch_mixing(X, batch, random_state=42)
    r2 = batchfx.batch_mixing(X, batch, random_state=42)
    assert r1.rejection == r2.rejection
