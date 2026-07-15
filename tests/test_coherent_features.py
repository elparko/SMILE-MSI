"""Tests for spatial.find_coherent_features — the spatial finder + artifact-rejection
quality gate (spatial-chaos morphology + hotspot concentration on top of Moran's I)."""
import numpy as np

from smile_msi import demo, spatial


def test_returns_quality_scored_features():
    ds = demo.make_synthetic(width=28, height=22, seed=11)
    res = spatial.find_coherent_features(ds, snr=3.0, min_frequency=0.02, min_quality=0.1)

    # funnel is monotone: quality survivors <= spatial survivors <= candidates
    assert res.n_candidates > 0
    assert res.n_after_quality == len(res.peaks)
    assert res.n_after_quality <= res.n_after_spatial <= res.n_candidates
    assert len(res.peaks) >= 1                                   # a real section has coherent ions

    for p in res.peaks:
        assert 0.0 <= p["quality"] <= 1.0
        assert 0.0 <= p["hotspot_fraction"] <= 1.0
        c = p["spatial_chaos"]
        assert np.isnan(c) or 0.0 <= c <= 1.0
        assert "morans_i" in p and "mz" in p                    # carries the spatial-finder fields

    qs = [p["quality"] for p in res.peaks]
    assert qs == sorted(qs, reverse=True)                       # sorted by descending quality


def test_stricter_gate_keeps_fewer():
    ds = demo.make_synthetic(width=28, height=22, seed=7)
    loose = spatial.find_coherent_features(ds, min_quality=0.0, max_hotspot=1.0)
    strict = spatial.find_coherent_features(ds, min_quality=0.5, max_hotspot=0.5)
    assert strict.n_after_quality <= loose.n_after_quality


def test_composite_quality_penalizes_hotspots():
    # a structured, well-spread ion keeps its score...
    assert spatial._composite_quality(0.8, 0.9, 0.0) > 0.5
    # ...but the same ion collapsed onto a few pixels (hotspot -> 1) is driven to ~0
    assert spatial._composite_quality(0.8, 0.9, 0.98) < 0.1
    # NaN chaos falls back to Moran's I alone (no hotspot penalty)
    assert abs(spatial._composite_quality(0.6, float("nan"), 0.0) - 0.6) < 1e-9
    # output is always within [0, 1]
    assert 0.0 <= spatial._composite_quality(2.0, 2.0, -1.0) <= 1.0


def test_impossible_quality_gate_yields_none():
    ds = demo.make_synthetic(width=20, height=16, seed=9)
    res = spatial.find_coherent_features(ds, min_quality=1.01)   # quality maxes at 1.0
    assert res.peaks == []
    assert res.n_after_quality == 0


def test_registered_analysis_runs():
    from smile_msi import registry

    sd = registry.REGISTRY["find_coherent_features"]
    ds = demo.make_synthetic(width=20, height=16, seed=3)
    params = {p.name: p.default for p in sd.params}
    res = sd.run(ds, {}, params)
    assert res.n_after_quality == len(res.peaks)
    # the registry adapters work on the result
    assert isinstance(sd.produce(res), dict) and "peaks" in sd.produce(res)
    assert isinstance(sd.summary(res), str)
