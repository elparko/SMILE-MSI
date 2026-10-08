"""Pseudoreplication guard for region comparison (smile_msi.spatial.roi_comparison).

Pixels within a region are spatially autocorrelated, so treating each pixel as an
independent replicate inflates significance. The engine supports summarizing by a
per-pixel ``samples`` array (each ROI = one replicate) and testing *across* replicates.
These tests pin that behavior; the GUI (gui/stats.do_compare_regions) defaults to it.
"""
import numpy as np

from smile_msi import demo, spatial


def _ds_and_peaks():
    ds = demo.make_synthetic()
    ds.prime()
    peaks = [p["mz"] for p in ds.pick_peaks(max_peaks=12)]
    return ds, peaks


def test_pixel_mode_is_the_unsafe_default_unit():
    ds, peaks = _ds_and_peaks()
    q = ds.n_pixels // 5
    a = np.zeros(ds.n_pixels, bool); a[:q] = True
    b = np.zeros(ds.n_pixels, bool); b[2 * q:3 * q] = True
    df = spatial.roi_comparison(ds, a, b, peaks)            # no samples → per-pixel
    assert df.attrs["unit"] == "pixel"


def test_roi_replicate_summary_tests_across_regions():
    ds, peaks = _ds_and_peaks()
    q = ds.n_pixels // 5
    samples = np.full(ds.n_pixels, -1, int)
    samples[:q] = 0; samples[q:2 * q] = 1                   # A: 2 replicate ROIs
    samples[2 * q:3 * q] = 2; samples[3 * q:4 * q] = 3      # B: 2 replicate ROIs
    a = (samples == 0) | (samples == 1)
    b = (samples == 2) | (samples == 3)
    df = spatial.roi_comparison(ds, a, b, peaks, samples=samples)
    assert df.attrs["unit"] == "sample"
    assert df.attrs["n_a"] == 2 and df.attrs["n_b"] == 2   # n = ROIs, not pixels
    assert "warning" not in df.attrs                        # ≥2 per side → adequately powered


def test_single_replicate_per_side_warns_underpowered():
    ds, peaks = _ds_and_peaks()
    q = ds.n_pixels // 5
    samples = np.full(ds.n_pixels, -1, int)
    samples[:q] = 0; samples[2 * q:3 * q] = 1              # one ROI per side
    df = spatial.roi_comparison(ds, (samples == 0), (samples == 1), peaks, samples=samples)
    assert df.attrs["n_a"] == 1 and df.attrs["n_b"] == 1
    assert df.attrs.get("warning")                         # flags the underpowered comparison


def _two_roi_per_side(ds):
    """A/B masks plus a per-pixel ``samples`` array of 2 replicate ROIs per side."""
    q = ds.n_pixels // 5
    samples = np.full(ds.n_pixels, -1, int)
    samples[:q] = 0; samples[q:2 * q] = 1                   # A: 2 replicate ROIs
    samples[2 * q:3 * q] = 2; samples[3 * q:4 * q] = 3      # B: 2 replicate ROIs
    return (samples == 0) | (samples == 1), (samples == 2) | (samples == 3), samples


def test_effect_pixel_keeps_per_pixel_auc_with_replicate_p():
    """effect='pixel' is the SCiLS-style split the ROI-stats panel uses: the AUC + means
    stay the per-pixel discrimination effect size (identical to a pure per-pixel run, so
    they match the per-pixel ROC curve), while p/q summarize across the ROI replicates."""
    ds, peaks = _ds_and_peaks()
    a, b, samples = _two_roi_per_side(ds)
    px = spatial.roi_comparison(ds, a, b, peaks)                              # per-pixel everything
    dec = spatial.roi_comparison(ds, a, b, peaks, samples=samples, effect="pixel")
    assert dec.attrs["unit"] == "sample" and dec.attrs["effect_unit"] == "pixel"
    assert dec.attrs["n_a"] == 2 and dec.attrs["n_b"] == 2                    # n = ROIs, not pixels
    # AUC + means are the per-pixel effect size — identical to the pure per-pixel run …
    assert np.allclose(dec["AUC"].to_numpy(), px["AUC"].to_numpy(), equal_nan=True)
    assert np.allclose(dec["mean_A"].to_numpy(), px["mean_A"].to_numpy())
    assert np.allclose(dec["mean_B"].to_numpy(), px["mean_B"].to_numpy())
    # … while the p-value is the across-replicate inference, far less extreme than per-pixel
    assert dec["p_value"].min() >= px["p_value"].min()
    assert not np.allclose(dec["p_value"].to_numpy(), px["p_value"].to_numpy())


def test_effect_match_summarizes_the_auc_too():
    """The legacy default (effect='match') summarizes *every* column to the replicate unit,
    so its AUC differs from the per-pixel effect size and no effect_unit tag is stamped."""
    ds, peaks = _ds_and_peaks()
    a, b, samples = _two_roi_per_side(ds)
    px = spatial.roi_comparison(ds, a, b, peaks)
    m = spatial.roi_comparison(ds, a, b, peaks, samples=samples)              # effect='match'
    assert m.attrs["unit"] == "sample" and "effect_unit" not in m.attrs
    assert not np.allclose(m["AUC"].to_numpy(), px["AUC"].to_numpy(), equal_nan=True)


def test_pixel_level_summaries_say_the_p_values_describe_pixels():
    """A per-pixel test's one-line summary ("N ions q<0.05") must not read as N findings:
    with one ROI per group its p-values describe pixels, not replicates."""
    from smile_msi import registry

    ds, peaks = _ds_and_peaks()
    q = ds.n_pixels // 5
    labels = np.full(ds.n_pixels, -1, int)
    labels[:q], labels[2 * q:3 * q], labels[3 * q:4 * q] = 0, 1, 2      # one ROI per group
    multi = spatial.multigroup_features(ds, labels, peaks)
    assert "pseudoreplication" in registry.REGISTRY["multigroup_features"].summary(multi)
    cmp = registry.REGISTRY["roi_comparison"].summary
    assert "pseudoreplication" in cmp(spatial.roi_comparison(ds, labels == 0, labels == 1, peaks))
    a, b, samples = _two_roi_per_side(ds)
    assert "pseudoreplication" not in cmp(spatial.roi_comparison(ds, a, b, peaks, samples=samples))
