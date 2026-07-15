"""Absolute m/z calibration: engine-computed anchor resolution + offset-measurement
round-trip (measure a known injected offset, then verify a lock-mass correction removes it).

The measurement compares the mean spectrum's apexes to *known* reference ion m/z — the
absolute-offset check that neither ``intake.mass_drift`` (per-pixel spread) nor
``preprocess.auto_recalibrate`` (self-referential) provides."""
import numpy as np

from smile_msi import intake


def test_default_anchors_resolve_to_known_masses():
    """Anchors are computed from the in-silico DB (no hardcoded lock masses) and match the
    literature ion m/z the user calibrates against."""
    neg = dict(intake.default_calibration_anchors("negative"))
    by = {lbl.rsplit(" ", 1)[0]: mz for lbl, mz in neg.items()}   # strip the trailing adduct
    assert abs(by["FA 18:1"] - 281.2486) < 1e-3
    assert abs(by["Sulfatide 42:2;O2"] - 888.6240) < 2e-3
    assert abs(by["PI 38:4"] - 885.5499) < 2e-3
    assert len(intake.default_calibration_anchors("positive")) >= 3


class _StubDS:
    """Minimal stand-in — measure_calibration_offset only needs mean_spectrum()."""
    def __init__(self, axis, spec):
        self._axis, self._spec = axis, spec

    def mean_spectrum(self, mask=None):
        return self._axis, self._spec


def _synthetic(offset_ppm):
    """A profile mean spectrum with a Gaussian at each anchor's *observed* (offset) m/z."""
    anchors = intake.default_calibration_anchors("negative")
    axis = np.arange(200.0, 950.0, 0.001)
    spec = np.zeros_like(axis)
    for _lbl, true_mz in anchors:
        obs = true_mz * (1 + offset_ppm * 1e-6)
        spec += np.exp(-0.5 * ((axis - obs) / 0.003) ** 2)
    return axis, spec


def test_measure_recovers_injected_offset():
    axis, spec = _synthetic(-6.0)
    r = intake.measure_calibration_offset(_StubDS(axis, spec), mode="negative")
    assert r["n"] >= 6
    assert abs(r["median_ppm"] - (-6.0)) < 0.5
    assert abs(r["slope_ppm_per_da"]) < 0.05                 # constant offset → flat slope
    assert r["factor"] > 1.0                                 # masses low → correction scales up


def test_recalibration_round_trip_under_2ppm():
    axis, spec = _synthetic(-6.0)
    r = intake.measure_calibration_offset(_StubDS(axis, spec), mode="negative")
    axis_corr = axis * r["factor"]                           # apply the multiplicative fix
    r2 = intake.measure_calibration_offset(_StubDS(axis_corr, spec), mode="negative")
    assert abs(r2["median_ppm"]) < 2.0                       # target: sulfatides land <2 ppm


def test_no_anchor_match_returns_neutral():
    axis = np.arange(200.0, 210.0, 0.001)                    # no anchors in this window
    r = intake.measure_calibration_offset(_StubDS(axis, np.ones_like(axis)), mode="negative")
    assert r["n"] == 0 and r["factor"] == 1.0
